from dataloader import Dataset
import numpy as np
import vector
import matplotlib.pyplot as plt
import awkward as ak
import fastjet
import subprocess
import uproot
import argparse 
import itertools
import os
from omnifold import  Multifold
import horovod.tensorflow as hvd
import tensorflow as tf
hvd.init()
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a PET model using Pythia and Herwig data.")
    parser.add_argument("--data_dir", type=str, default="/global/cfs/cdirs/m3246/H1/h5/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/global/homes/r/rmilton/m3246/rmilton/H1Unfold/jet_data/", help="Folder to store jet data")
    parser.add_argument("--outfile_name", type=str, default="saved_jets.root", help="Name of output ROOT file")
    parser.add_argument("--num_data", type=int, default=-1, help="Number of data to analyze")
    parser.add_argument("--frame", type=str, default="breit", help="breit/lab frames")
    parser.add_argument("--clustering", type=str, default="centauro", help="kt/centauro clustering")
    parser.add_argument("--dataset", type=str, default="H1", help="H1/Rapgap/Djangoh will be used. Rapgap and Djangoh will be saved with unfolded weights")

    args = parser.parse_args()
    return args


def load_model(model_path, dataloader):
    #Load the trained model        
    model_name = '{}/checkpoint'.format(model_path)
    if hvd.rank()==0:
        print("Loading model {}".format(model_name))

    mfold = Multifold(verbose = hvd.rank()==0)
    mfold.PrepareModel()
    mfold.model2.load_weights(model_name).expect_partial() #Doesn't matter which model is loaded since both have the same architecture
    unfolded_weights = mfold.reweight(dataloader.gen,mfold.model2_ema,batch_size=1000)
    #return unfolded_weights
    return hvd.allgather(tf.constant(unfolded_weights)).numpy()

def _convert_kinematics(part, event, mask):
    #return particles in cartesian coordinates
    new_particles = np.zeros((part.shape[0],part.shape[1],4))
    # part[:,:,i] = [eta_part-eta_e, phi_part-phi_e-pi, log(pT), log(pT/Q), log(E/Q), log(E), sqrt(eta_part - eta_e)^2 + (phi_part - phi_e)^2), charge]
    # event[:,i,None] = [log(Q^2), y, pT_e/Q, eta_e, phi_e]
    new_particles[:,:,0] = np.ma.exp(part[:,:,2])*np.cos((np.pi + part[:,:,1] + event[:,4,None]))
    new_particles[:,:,1] = np.ma.exp(part[:,:,2])*np.sin((np.pi + part[:,:,1] + event[:,4,None]))
    new_particles[:,:,2] = np.ma.exp(part[:,:,2])*np.sinh((part[:,:,0] + event[:,3,None]))
    new_particles[:,:,3] = np.ma.exp(part[:,:,5])
    return new_particles*mask[:,:,None]

def _convert_electron_kinematics(event_list):
    pt = event_list[:, 2]*np.sqrt(np.exp(event_list[:, 0]))
    phi = event_list[:, 4]
    eta = event_list[:, 3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2)
    vectorize_func = np.vectorize(vector.obj, otypes=[object])
    electron_cartesian = vectorize_func(px=px, py=py, pz=pz, energy=E)
    return electron_cartesian

def boost_particles(final_states, scattered_electron):
    particle_vectors, particle_sigma_tot = [], []
    # Putting all particles into vectors and calculating sigma_tot
    for i, event in enumerate(final_states):
        sigma_h = np.sum([part[3] - part[2] for part in event if np.abs(part[0])!=0]) # sum_i(E_i - pz_i)
        sigma_eprime = scattered_electron[i].energy*(1-np.cos(scattered_electron[i].theta))
        sigma_tot = sigma_h + sigma_eprime
        particle_vectors.append([vector.obj(px=part[0], py=part[1], pz=part[2], energy=part[3]) for part in event if np.abs(part[0]!=0)])
        particle_sigma_tot.append(sigma_tot)
    
    # Getting the effictive lepton beam momentum
    particle_sigma_tot = np.asarray(particle_sigma_tot)
    vectorize_func = np.vectorize(vector.obj, otypes=[object])
    beam_electron_momentum = vectorize_func(px=0, py=0, pz=-particle_sigma_tot/2., energy=particle_sigma_tot/2.)

    # Boosting the particles to the Breit frame
    boosted_vectors = []
    for i, (beam_e, scattered_e, event_particles) in enumerate(zip(beam_electron_momentum, scattered_electron, particle_vectors)):
        # Calculating boost vector
        # Calculation taken from https://doi.org/10.1140/epjc/s10052-024-13003-1
        z_hat = vector.obj(px=0, py=0, pz=1, energy=1)
        photon = beam_e - scattered_e
        Q = np.sqrt(-1*photon.dot(photon))
        Sigma = photon.energy - photon.pz
        boostvector = photon - Q*Q/Sigma*(-1*z_hat)
        xBoost = vector.obj(px = photon.px/photon.pt, py = photon.py/photon.pt, pz = photon.pt/Sigma, energy = photon.pt/Sigma)
        yBoost = vector.obj(px = -photon.py/photon.pt, py = photon.px/photon.pt, pz = 0, energy = 0)
        boosted_event_vectors = []
        for vec in event_particles:
            breit_vec = vector.obj(px = xBoost.dot(vec), py = yBoost.dot(vec), pz = photon.dot(vec)/Q, energy = boostvector.dot(vec)/Q)
            boosted_event_vectors.append(breit_vec)
        boosted_vectors.append(boosted_event_vectors)
    return boosted_vectors

if __name__ == "__main__":

    flags = parse_arguments()
    file_data_dict = {"H1": 'data_prep.h5', "Rapgap": 'Rapgap_Eplus0607_prep.h5', "Djangoh": 'Djangoh_Eplus0607_prep.h5'}
    model_dict = {"Rapgap": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_pretrained_step2", "Djangoh": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_pretrained_step2"}
    dataset = flags.dataset
    if dataset != "H1" and dataset != "Rapgap" and dataset != "Djangoh":
        print("Invalid dataset. Please use H1, Rapgap, or Djangoh.")
        exit()
    if dataset == "H1":
        MC = False
    elif dataset == "Rapgap" or dataset == "Djangoh":
        MC = True
        model_path = model_dict[dataset]

    
    # Taking all events, not just the ones that pass fiduucial cuts
    dataloader = Dataset([file_data_dict[dataset]],
                         flags.data_dir,
                         rank=hvd.rank(),
                         size=hvd.size(),
                         is_mc=MC,
                         pass_fiducial=MC,
                         pass_reco = not MC,
                         nmax=flags.num_data)
    num_events = dataloader.gen[0].shape[0] if MC else dataloader.reco[0].shape[0]
    print(f"Loaded {num_events} data events")
    if MC:
        unfolded_weights = load_model(model_path, dataloader)
    #Undo the preprocessing
    if MC:
        particles, events = dataloader.revert_standardize(dataloader.gen[0], dataloader.gen[1], dataloader.gen[-1])
        mask = dataloader.gen[-1]
    else:
        particles, events = dataloader.revert_standardize(dataloader.reco[0], dataloader.reco[1], dataloader.reco[-1])
        mask = dataloader.reco[-1]
    
    calibration_weights = dataloader.weight # These weights are calibration constants that should be applied

    if not MC:
        pass_reco_data = dataloader.pass_reco
        particles = particles[pass_reco_data]
        events = events[pass_reco_data]
        mask = mask[pass_reco_data]
        calibration_weights = calibration_weights[pass_reco_data]

    num_valid_events = len(events)
    # Getting the four vectors of our hadronic final states
    # cartesian_particles will be shape (N_events, N_particles, 4)
    # The 4 will be [px, py, pz, E]
    cartesian_particles = _convert_kinematics(particles, events, mask)
    
    events_for_clustering = []
    # Boosting particles to Breit frame
    if flags.frame == "breit":
        print("Boosting to the Breit frame.")
        # Getting scattered electron kinematics
        # Shape is (N_events,)
        # Each entry will be a MomentumObject4D containing the lab frame 4-momenta
        electron_momentum = _convert_electron_kinematics(events)
        # boosted_vectors has shape (N_events, N_particles_in_event,)
        # Each event contains the 4-momenta of the particles
        boosted_vectors = boost_particles(cartesian_particles, electron_momentum)
        #Extracting individual components of 4-vectors
        for event in boosted_vectors:
            events_for_clustering.append([{"px": part_vec.px, "py": part_vec.py, "pz": part_vec.pz, "E": part_vec.E} for part_vec in event])
            
    elif flags.frame == "lab":
        for event in cartesian_particles:
            events_for_clustering.append([{"px": part[0], "py": part[1], "pz": part[2], "E": part[3]} for part in event if np.abs(part[0])!=0])

    if flags.clustering == "centauro":
        px, py, pz, energy = [], [], [], []
        for event in events_for_clustering:
            px.append([particle_vector["px"] for particle_vector in event])
            py.append([particle_vector["py"] for particle_vector in event])
            pz.append([particle_vector["pz"] for particle_vector in event])
            energy.append([particle_vector["E"] for particle_vector in event])
        px = ak.Array(px)
        py = ak.Array(py)
        pz = ak.Array(pz)
        energy = ak.Array(energy)
        # Saving 4-vectors and passing it to Centauro jet-clustering
        with uproot.recreate("./breit_particles.root") as file:
            file["particles"] = {"px": px, "py": py, "pz": pz, "energy": energy}
        subprocess.run(["./run_centauro.sh"], shell=True)
        with uproot.open("{file}:jets".format(file="./centauro_jets.root")) as out:
            jets = out.arrays(["eta", "px", "py", "pz", "E", "pT", "phi"])
    elif flags.clustering == "kt":
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)

        array = ak.Array(events_for_clustering)
        cluster = fastjet.ClusterSequence(array, jetdef)
        jets = cluster.inclusive_jets(min_pt=10)
        jets["pt"] = -np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arctanh(jets["pz"]/np.sqrt(jets["px"]**2 + jets["py"]**2 + jets["pz"]**2))
        jets=fastjet.sorted_by_pt(jets)

        def _take_leading_jet(jets):
            jet = {}
            jet["pT"] = -np.array(list(itertools.zip_longest(*jets.pt.to_list(), fillvalue=0))).T[:,0]
            jet["eta"] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T[:,0]
            jet["phi"] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T[:,0]
            jet["E"] =   np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T[:,0]
            jet["px"] = np.array(list(itertools.zip_longest(*jets.px.to_list(), fillvalue=0))).T[:,0]
            jet["py"] = np.array(list(itertools.zip_longest(*jets.py.to_list(), fillvalue=0))).T[:,0]
            jet["pz"] = np.array(list(itertools.zip_longest(*jets.pz.to_list(), fillvalue=0))).T[:,0]
            return jet
        jets = _take_leading_jet(jets)
    # Saving clustered jets and event kinematics.
    output_path = flags.save_dir + flags.outfile_name
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    with uproot.recreate(output_path) as file:
        file["jets"] = {"eta": jets["eta"], "px": jets["px"], "py": jets["py"], "pz": jets["pz"], "E": jets["E"], "pT": jets["pT"], "phi": jets["phi"]}
        output_event_dict = {"Q2": np.exp(events[:, 0]), "y": events[:,1], "elec_pT": events[:,2]*np.sqrt(np.exp(events[:, 0])), "elec_eta": events[:,3], "elec_phi": events[:,4], "weight":calibration_weights}
        if MC:
            output_event_dict["unfolded_weights"] = unfolded_weights
        file["event"] = output_event_dict
        