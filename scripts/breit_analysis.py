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
import os, gc
from omnifold_vinny import  Multifold
from omnifold import PET
import horovod.tensorflow as hvd
import tensorflow as tf
import h5py as h5
import utils
import time
hvd.init()
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a PET model using Pythia and Herwig data.")
    parser.add_argument("--data_dir", type=str, default="/global/cfs/cdirs/m3246/H1/h5/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/global/homes/r/rmilton/m3246/rmilton/H1Unfold/jet_data/", help="Folder to store jet data")
    parser.add_argument("--outfile_name", type=str, default="saved_jets.root", help="Name of output ROOT file")
    parser.add_argument("--num_data", type=int, default=-1, help="Number of data to analyze")
    parser.add_argument("--frame", type=str, default="breit", help="breit/lab frames")
    parser.add_argument("--clustering", type=str, default="centauro", help="kt/centauro clustering")
    parser.add_argument("--jet_radius", type=float, default=0.8, help="Jet radius for clustering")
    parser.add_argument("--dataset", type=str, default="H1", help="H1/Rapgap/Djangoh will be used. Rapgap and Djangoh will be saved with unfolded weights")
    parser.add_argument("--use_vinny_models", action='store_true', default=False,help='Use unfolded models from Vinny')
    parser.add_argument("--vinny_version", type=str, default='H1_July_closure', help="Version of Vinny model. Only used when use_vinny_models is True")
    parser.add_argument("--load_breit", action='store_true', default=False, help = "Option to load boosted Breit particles from file")
    parser.add_argument("--breit_gen_path", type=str, default="breit_particles_Rapgap_gen.root", help="Location of saved gen Breit particles ROOT file")
    parser.add_argument("--breit_reco_path", type=str, default="breit_particles_Rapgap_reco.root", help="Location of saved reco Breit particles ROOT file")
    parser.add_argument("--saved_step1_weights_path", type=str, default="", help="Location of saved step 1 weights")
    parser.add_argument("--saved_step2_weights_path", type=str, default="", help="Location of saved step 2 weights")
    parser.add_argument("--saved_weights_name", type=str, default="", help="Location and name of file to save weights to")
    args = parser.parse_args()
    return args


def load_model(model_path, events, use_vinny_models, version = None):
    def expit(x):
        return 1. / (1. + np.exp(-x))
    def reweight(events,model,batch_size=None):
        f = expit(model.predict(events,batch_size=batch_size))
        weights = f / (1. - f)  # this is the crux of the reweight, approximates likelihood ratio
        weights = np.nan_to_num(weights[:,0],posinf=1)
        return weights
    
    if hvd.rank()==0:
        print("Loading model {}".format(model_path))
    if use_vinny_models:
        mfold = Multifold(version = version,verbose = hvd.rank()==0)
        mfold.PrepareModel()
        mfold.model2.load_weights(model_path).expect_partial() #Doesn't matter which model is loaded since both have the same architecture
        unfolded_weights = mfold.reweight(events,mfold.model2_ema,batch_size=1000)
    else:
        events = np.asarray(events[0])
        PET_model = PET(events.shape[2], num_part=events.shape[1], num_heads = 4, num_transformer = 4, local = True, projection_dim = 128, K = 10)
        PET_model.load_weights(model_path)
        unfolded_weights =  reweight(events, PET_model, batch_size=1000)
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
    electron_cartesian_dict = {"px":px, "py":py, "pz":pz, "E":E}
    # vectorize_func = np.vectorize(vector.obj, otypes=[object])
    # electron_cartesian = vectorize_func(px=px, py=py, pz=pz, energy=E)
    return electron_cartesian_dict

def boost_particles(final_states, scattered_electron):
    particle_vectors = []
    # Putting all particles into vectors and calculating sigma_tot
    for event in final_states:
        nonzero_particles = np.array([part for part in event if np.abs(part[0])!=0])
        particle_vectors.append([vector.obj(px=part[0], py=part[1], pz=part[2], energy=part[3]) for part in nonzero_particles])
    
    sigma_h = np.array([np.sum(event[:, 3] - event[:, 2]) for event in final_states if any(np.abs(event[:, 0]) != 0)])

    scattered_electron_momentum = np.sqrt(scattered_electron["px"]**2 + scattered_electron["py"]**2 + scattered_electron["pz"]**2)
    scattered_electron_theta = np.arccos(scattered_electron["pz"]/scattered_electron_momentum)
    sigma_eprime = scattered_electron["E"] * (1 - np.cos(scattered_electron_theta))
    sigma_tot = sigma_h + sigma_eprime

    beam_electron_momentum = {"px":np.zeros(len(sigma_tot)), "py":np.zeros(len(sigma_tot)), "pz":-sigma_tot/2., "E":sigma_tot/2.}
    q_x = beam_electron_momentum["px"] - scattered_electron["px"]
    q_y = beam_electron_momentum["py"] - scattered_electron["py"]
    q_z = beam_electron_momentum["pz"] - scattered_electron["pz"]
    q_E = beam_electron_momentum["E"] - scattered_electron["E"]
    q_list = np.stack((q_x, q_y, q_z, q_E), axis=1)
    # Boosting the particles to the Breit frame
    boosted_vectors = []
    z_hat = vector.obj(px=0, py=0, pz=1, energy=1)

    for q, event_particles in zip(q_list, particle_vectors):
        # Calculating boost vector
        # Calculation taken from https://doi.org/10.1140/epjc/s10052-024-13003-1
        q = vector.obj(px=q[0], py=q[1], pz=q[2], energy=q[3])
        Q = np.sqrt(-1*q.dot(q))
        Sigma = q.energy - q.pz
        boostvector = q - Q*Q/Sigma*(-1*z_hat)
        xBoost = vector.obj(px = q.px/q.pt, py = q.py/q.pt, pz = q.pt/Sigma, energy = q.pt/Sigma)
        yBoost = vector.obj(px = -q.py/q.pt, py = q.px/q.pt, pz = 0, energy = 0)
        boosted_event_vectors = [
            vector.obj(
                px = xBoost.dot(vec),
                py = yBoost.dot(vec),
                pz = q.dot(vec)/Q,
                energy = boostvector.dot(vec)/Q
                )
                for vec in event_particles
        ]

        boosted_vectors.append(boosted_event_vectors)
    return boosted_vectors

def get_q_and_y(final_states, scattered_electron):
    sigma_h = np.array([np.sum(event[:, 3] - event[:, 2]) for event in final_states if any(np.abs(event[:, 0]) != 0)])

    scattered_electron_momentum = np.sqrt(scattered_electron["px"]**2 + scattered_electron["py"]**2 + scattered_electron["pz"]**2)
    scattered_electron_theta = np.arccos(scattered_electron["pz"]/scattered_electron_momentum)
    sigma_eprime = scattered_electron["E"] * (1 - np.cos(scattered_electron_theta))
    sigma_tot = sigma_h + sigma_eprime

    y = sigma_h/sigma_tot
    beam_electron_momentum = {"px":np.zeros(len(sigma_tot)), "py":np.zeros(len(sigma_tot)), "pz":-sigma_tot/2., "E":sigma_tot/2.}
    q_x = beam_electron_momentum["px"] - scattered_electron["px"]
    q_y = beam_electron_momentum["py"] - scattered_electron["py"]
    q_z = beam_electron_momentum["pz"] - scattered_electron["pz"]
    q_E = beam_electron_momentum["E"] - scattered_electron["E"]
    q_list = np.stack((q_x, q_y, q_z, q_E), axis=1)
    return q_list, y

def clustering_procedure(cartesian_particles, dataloader, dataset, reco, jet_radius, load_breit, breit_gen_path, breit_reco_path):
    clustering_start = time.time()
    events_for_clustering = []
    
    # Boosting particles to Breit frame
    if flags.frame == "breit":
        # Getting scattered electron kinematics
        # Shape is (N_events,)
        # Each entry will be a MomentumObject4D containing the lab frame 4-momenta
        electron_momentum = _convert_electron_kinematics(dataloader.reco_events if reco else dataloader.gen_events)

        # boosted_vectors has shape (N_events, N_particles_in_event,)
        # Each event contains the 4-momenta of the particles
        if not load_breit:
            print("Boosting to the Breit frame.")

            boosted_vectors = boost_particles(cartesian_particles, electron_momentum)
            for event in boosted_vectors:
                events_for_clustering.append([{"px": part_vec.px, "py": part_vec.py, "pz": part_vec.pz, "E": part_vec.E} for part_vec in event])
        # Need to add support for breit + kt
        q, y = get_q_and_y(cartesian_particles, electron_momentum)
        # Extracting individual components of 4-vectors
        
    elif flags.frame == "lab":
        events_for_clustering = [
            [fastjet.PseudoJet(*part) for part in event[event[:, 0] != 0]] # Removing particles that are zero padded
            for event in cartesian_particles
        ]
        electron_momentum = _convert_electron_kinematics(dataloader.reco_events if reco else dataloader.gen_events)
        print("Configuring events time: ", time.time()-clustering_start)
        q, y = get_q_and_y(cartesian_particles, electron_momentum)
        print("q time: ", time.time()-clustering_start)
    if reco:
        dataloader.reco_q = q
        dataloader.reco_y = y
    else:
        dataloader.gen_q = q
        dataloader.gen_y = y

    if flags.clustering == "centauro":
        if not load_breit:
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
        if reco:
            if load_breit:
                breit_file_name = breit_reco_path
                print(f"Loading reco from pre-boosted Breit files: {breit_file_name}")
            else:
                breit_file_name = f"breit_particles_{dataset}_reco.root"
            jet_file_name = f"centauro_jets_{dataset}_reco.root"
        else:
            if load_breit:
                breit_file_name = breit_gen_path
                print(f"Loading gen from pre-boosted Breit files: {breit_file_name}")
            else:
                breit_file_name = f"breit_particles_{dataset}_gen.root"
            jet_file_name = f"centauro_jets_{dataset}_gen.root"
        if not load_breit:
            with uproot.recreate(breit_file_name) as file:
                file["particles"] = {"px": px, "py": py, "pz": pz, "energy": energy}
        print(f"./run_centauro.sh --input {breit_file_name} --output {jet_file_name} --jet_radius {jet_radius}")
        subprocess.run([f"./run_centauro.sh --input {breit_file_name} --output {jet_file_name}"], shell=True)
        with uproot.open(f"{jet_file_name}:jets") as out:
            jets = out.arrays(["pT", "eta", "phi", "E", "px", "py", "pz"])
            max_num_jets = max([len(jet_pt) for jet_pt in jets["pT"]])
            def _take_all_jet(jets, max_num_jets):
                dataloader.reco_events if reco else dataloader.gen_events
                if reco:
                    jet = np.zeros((dataloader.reco_events.shape[0],7, max_num_jets))
                else:
                    jet = np.zeros((dataloader.gen_events.shape[0],7, max_num_jets))
                # jet = np.zeros_like(jets["pt"])
                jet[:,0] = np.array(list(itertools.zip_longest(*jets.pT.to_list(), fillvalue=0))).T
                jet[:,1] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T
                jet[:,2] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T
                jet[:,3] = np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T
                jet[:,4] = np.array(list(itertools.zip_longest(*jets.px.to_list(), fillvalue=0))).T
                jet[:,5] = np.array(list(itertools.zip_longest(*jets.py.to_list(), fillvalue=0))).T
                jet[:,6] = np.array(list(itertools.zip_longest(*jets.pz.to_list(), fillvalue=0))).T
                return jet
            
        if reco:
            dataloader.reco_jet = _take_all_jet(jets, max_num_jets)
        else:
            dataloader.gen_jet = _take_all_jet(jets, max_num_jets)

    elif flags.clustering == "kt":
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, jet_radius)
        jet_px, jet_py, jet_pz, jet_E = [], [], [], []
        for event in events_for_clustering:
            event_px, event_py, event_pz, event_E = [], [], [], []
            cluster = fastjet.ClusterSequence(event, jetdef)
            event_jets = cluster.inclusive_jets(0)
            event_jets = fastjet.sorted_by_E(event_jets)
            for jet in event_jets:
                num_constituents = len(jet.constituents())
                if num_constituents > 1:
                    event_px.append(jet.px())
                    event_py.append(jet.py())
                    event_pz.append(jet.pz())
                    event_E.append(jet.E())
            jet_px.append(event_px)
            jet_py.append(event_py)
            jet_pz.append(event_pz)
            jet_E.append(event_E)
        print("Actual clustering time: ", time.time()-clustering_start)
        jets = {"px":ak.Array(jet_px), "py":ak.Array(jet_py), "pz":ak.Array(jet_pz),"E":ak.Array(jet_E)}
        jets["pt"] = np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arctanh(jets["pz"]/np.sqrt(jets["px"]**2 + jets["py"]**2 + jets["pz"]**2))
        jets = ak.Record(jets)
        max_num_jets = max([len(jet_pt) for jet_pt in jets["pt"]])
        def _take_all_jet(jets, max_num_jets):
            dataloader.reco_events if reco else dataloader.gen_events
            if reco:
                jet = np.zeros((dataloader.reco_events.shape[0],7, max_num_jets))
            else:
                jet = np.zeros((dataloader.gen_events.shape[0],7, max_num_jets))
            jet[:,0] = np.array(list(itertools.zip_longest(*jets.pt.to_list(), fillvalue=0))).T
            jet[:,1] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T
            jet[:,2] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T
            jet[:,3] =   np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T
            jet[:,4] = np.array(list(itertools.zip_longest(*jets.px.to_list(), fillvalue=0))).T
            jet[:,5] = np.array(list(itertools.zip_longest(*jets.py.to_list(), fillvalue=0))).T
            jet[:,6] = np.array(list(itertools.zip_longest(*jets.pz.to_list(), fillvalue=0))).T
            return jet
        if reco:
            dataloader.reco_jet = _take_all_jet(jets, max_num_jets)
        else:
            dataloader.gen_jet = _take_all_jet(jets, max_num_jets)

if __name__ == "__main__":

    start_time = time.time()
    flags = parse_arguments()
    use_vinny_models = flags.use_vinny_models
    version = flags.vinny_version

    file_data_dict = {"H1": 'data_Eplus0607_prep.h5', "Rapgap": 'Rapgap_Eplus0607_prep.h5', "Djangoh": 'Djangoh_Eplus0607_prep.h5'}
    if use_vinny_models:
        model_dict_step1 = {"Rapgap": f"/global/cfs/cdirs/m3246/H1/weights/OmniFold_{version}_pretrained_iter0_step1/checkpoint"}
        model_dict_step2 = {"Rapgap": f"/global/cfs/cdirs/m3246/H1/weights/OmniFold_{version}_pretrained_iter4_step2/checkpoint"}
    else:
        # model_dict_step1 = {"Rapgap": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_Djangoh_Rapgap_closure_12_09_iter0_step1.weights.h5"}
        # model_dict_step2 = {"Rapgap": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_Djangoh_Rapgap_closure_12_09_iter4_step2.weights.h5"}
        model_dict_step1 = {"Rapgap": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_Rapgap_H1_noelectron_standardized_iter0_step1.weights.h5", "Djangoh":None}
        model_dict_step2 = {"Rapgap": "/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/OmniFold_Rapgap_H1_noelectron_standardized_iter3_step2.weights.h5", "Djangoh":None}
    dataset = flags.dataset
    if dataset != "H1" and dataset != "Rapgap" and dataset != "Djangoh":
        print("Invalid dataset. Please use H1, Rapgap, or Djangoh.")
        exit()
    if dataset == "H1":
        MC = False
    elif dataset == "Rapgap" or dataset=="Djangoh":
        MC = True
        model_path_step2 = model_dict_step2[dataset]
        model_path_step1 = model_dict_step1[dataset]

    
    # Taking all events, not just the ones that pass fiduucial cuts
    if flags.num_data == -1:
        nmax = None
    else:
        nmax = flags.num_data
        
    dataloader = Dataset([file_data_dict[dataset]],
                        flags.data_dir,
                        rank=hvd.rank(),
                        size=hvd.size(),
                        is_mc=MC,
                        nmax=nmax)
    
    print("Files loaded time: ", time.time()-start_time)
    dataloader.masked_reco = [dataloader.reco[0][dataloader.pass_reco], dataloader.reco[1][dataloader.pass_reco], dataloader.reco[2][dataloader.pass_reco]]
    dataloader.reco_weight = dataloader.weight[dataloader.pass_reco]
    # dataloader.masked_reco = [dataloader.reco[0], dataloader.reco[1], dataloader.reco[2]]
    # dataloader.reco_weight = dataloader.weight
    del dataloader.reco
    if MC:
        # dataloader.masked_gen = [dataloader.gen[0][dataloader.pass_gen], dataloader.gen[1][dataloader.pass_gen], dataloader.gen[2][dataloader.pass_gen]]
        # dataloader.gen_weight = dataloader.weight[dataloader.pass_gen]
        dataloader.masked_gen = [dataloader.gen[0], dataloader.gen[1], dataloader.gen[2]]
        dataloader.gen_weight = dataloader.weight
        del dataloader.gen
    del dataloader.weight
    gc.collect()

    #Undo the preprocessing
    # The mask obtained from masked_X[-1] removes events that have pT = 0
    if MC:
        if model_path_step1 is not None:
            if flags.saved_step1_weights_path == "":
                step1_weights = load_model(model_path_step1, dataloader.masked_reco, use_vinny_models, version)
            else:
                step1_weights = np.load(flags.saved_step1_weights_path)
        if flags.saved_weights_name == "":
            np.save('./step1.npy', step1_weights)
        else:
            np.save(flags.saved_weights_name+"_step1.npy", step1_weights)
        dataloader.reco_particles, dataloader.reco_events = dataloader.revert_standardize(dataloader.masked_reco[0], dataloader.masked_reco[1], dataloader.masked_reco[-1])
        dataloader.reco_mask = dataloader.masked_reco[-1]
        del dataloader.masked_reco

        print("Step 1 time: ", time.time()-start_time)
        if model_path_step2 is not None:
            if flags.saved_step2_weights_path == "":
                step2_weights = load_model(model_path_step2, dataloader.masked_gen, use_vinny_models, version)
            else:
                step2_weights = np.load(flags.saved_step2_weights_path)
        if flags.saved_weights_name == "":
            np.save('./step2.npy', step2_weights)
        else:
            np.save(flags.saved_weights_name+"_step2.npy", step2_weights)
        dataloader.gen_particles, dataloader.gen_events= dataloader.revert_standardize(dataloader.masked_gen[0], dataloader.masked_gen[1], dataloader.masked_gen[-1])
        dataloader.gen_mask = dataloader.masked_gen[-1]
        del dataloader.masked_gen
        print("Step 2 time: ", time.time()-start_time)
    else:
        dataloader.reco_particles, dataloader.reco_events = dataloader.revert_standardize(dataloader.masked_reco[0], dataloader.masked_reco[1], dataloader.masked_reco[-1])
        dataloader.reco_mask = dataloader.masked_reco[-1]
        del dataloader.masked_reco
    gc.collect()

    # Getting the four vectors of our hadronic final states
    # cartesian_particles will be shape (N_events, N_particles, 4)
    # The 4 will be [px, py, pz, E]
    if MC:
        gen_cartesian_particles = _convert_kinematics(dataloader.gen_particles, dataloader.gen_events, dataloader.gen_mask)
    reco_cartesian_particles = _convert_kinematics(dataloader.reco_particles, dataloader.reco_events, dataloader.reco_mask)
    load_breit = flags.load_breit
    if load_breit and flags.frame == "lab":
        print("load_breit is used, but frame is selected as lab. Lab frame will be used instead.")
    jet_radius = flags.jet_radius
    if MC:
        print("Clustering gen jets")
        clustering_procedure(
            cartesian_particles=gen_cartesian_particles,
            dataloader=dataloader,
            dataset=dataset,
            reco=False,
            jet_radius = jet_radius,
            load_breit=load_breit,
            breit_gen_path=flags.breit_gen_path,
            breit_reco_path=flags.breit_reco_path
        )
        print("Gen clustering time: ", time.time()-start_time)
    print("Clustering reco jets")
    clustering_procedure(
            cartesian_particles=reco_cartesian_particles,
            dataloader=dataloader,
            dataset=dataset,
            reco=True,
            jet_radius = jet_radius,
            load_breit=load_breit,
            breit_gen_path=flags.breit_gen_path,
            breit_reco_path=flags.breit_reco_path
        )
    print("Reco clustering time: ", time.time()-start_time)
    def gather_data(dataloader, is_MC):
        dataloader.reco_mask = np.reshape(dataloader.reco_mask,(-1))
        dataloader.reco_particles = hvd.allgather(tf.constant(dataloader.reco_particles.reshape(
            (-1,dataloader.reco_particles.shape[-1]))[dataloader.reco_mask])).numpy()
        
        dataloader.reco_events = hvd.allgather(tf.constant(dataloader.reco_events)).numpy()
        dataloader.reco_jet = hvd.allgather(tf.constant(dataloader.reco_jet)).numpy()
        dataloader.reco_weight = hvd.allgather(tf.constant(dataloader.reco_weight)).numpy()
        dataloader.reco_mask = hvd.allgather(tf.constant(dataloader.reco_mask)).numpy()
        dataloader.reco_q = hvd.allgather(tf.constant(dataloader.reco_q)).numpy()
        dataloader.reco_y = hvd.allgather(tf.constant(dataloader.reco_y)).numpy()
        if is_MC:
            dataloader.gen_mask = np.reshape(dataloader.gen_mask,(-1))
            dataloader.gen_particles = hvd.allgather(tf.constant(dataloader.gen_particles.reshape(
                (-1,dataloader.gen_particles.shape[-1]))[dataloader.gen_mask])).numpy()
            
            dataloader.gen_events = hvd.allgather(tf.constant(dataloader.gen_events)).numpy()
            dataloader.gen_jet = hvd.allgather(tf.constant(dataloader.gen_jet)).numpy()
            dataloader.gen_weight = hvd.allgather(tf.constant(dataloader.gen_weight)).numpy()
            dataloader.gen_mask = hvd.allgather(tf.constant(dataloader.gen_mask)).numpy()
            dataloader.gen_q = hvd.allgather(tf.constant(dataloader.gen_q)).numpy()
            dataloader.gen_y = hvd.allgather(tf.constant(dataloader.gen_y)).numpy()

    gather_data(dataloader, MC)
    print("Gathering data time: ", time.time()-start_time)
    if MC:
        gen_jets = dataloader.gen_jet
        gen_jet_dict = {"pT":gen_jets[:, 0], "eta":gen_jets[:, 1], "phi":gen_jets[:, 2], "E":gen_jets[:, 3], "px":gen_jets[:, 4], "py":gen_jets[:, 5], "pz":gen_jets[:, 6]}
        gen_q = dataloader.gen_q
        gen_y = dataloader.gen_y
        gen_q_dict = {"qx":gen_q[:,0], "qy":gen_q[:,1], "qz":gen_q[:,2],"E":gen_q[:,3]}

    reco_jets = dataloader.reco_jet
    reco_jet_dict = {"pT":reco_jets[:, 0], "eta":reco_jets[:, 1], "phi":reco_jets[:, 2], "E":reco_jets[:, 3], "px":reco_jets[:, 4], "py":reco_jets[:, 5], "pz":reco_jets[:, 6]}
    reco_q = dataloader.reco_q
    reco_y = dataloader.reco_y
    reco_q_dict = {"qx":reco_q[:,0], "qy":reco_q[:,1], "qz":reco_q[:,2],"E":reco_q[:,3]}
    
    # These weights are calibration constants that should be applied
    reco_events = dataloader.reco_events
    reco_calibration_weights = dataloader.reco_weight
    if MC:
        gen_events = dataloader.gen_events
        gen_calibration_weights = dataloader.gen_weight
     
    # Saving clustered jets and event kinematics.
    output_path = flags.save_dir + flags.outfile_name
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    with uproot.recreate(output_path) as file:
        if MC:
            file["gen_jets"] = {"eta": gen_jet_dict["eta"], "px": gen_jet_dict["px"], "py": gen_jet_dict["py"], "pz": gen_jet_dict["pz"], "E": gen_jet_dict["E"], "pT": gen_jet_dict["pT"], "phi": gen_jet_dict["phi"]}
            output_event_dict = {"q_x":gen_q_dict["qx"], "q_y":gen_q_dict["qy"], "q_z":gen_q_dict["qz"], "q_E":gen_q_dict["E"], "Q2": np.exp(gen_events[:, 0]), "y": dataloader.gen_y,"elec_pT": gen_events[:,2]*np.sqrt(np.exp(gen_events[:, 0])), "elec_eta": gen_events[:,3], "elec_phi": gen_events[:,4], "weight":gen_calibration_weights}
            if model_path_step2 is not None:
                output_event_dict["step2_weights"] = step2_weights
            file["gen_event"] = output_event_dict

        reco_event_dict = {}
        if MC:
            if model_path_step1 is not None:
                reco_event_dict["step1_weights"] = step1_weights
            
        file["reco_jets"] = {"eta": reco_jet_dict["eta"], "px": reco_jet_dict["px"], "py": reco_jet_dict["py"], "pz": reco_jet_dict["pz"], "E": reco_jet_dict["E"], "pT": reco_jet_dict["pT"], "phi": reco_jet_dict["phi"]}
        reco_event_dict["y"] =  dataloader.reco_y
        reco_event_dict["Q2"] = np.exp(reco_events[:,0])
        reco_event_dict["elec_pT"] = reco_events[:,2]*np.sqrt(np.exp(reco_events[:, 0]))
        reco_event_dict["elec_eta"] = reco_events[:,3]
        reco_event_dict["elec_phi"] = reco_events[:,4]
        reco_event_dict["q"] = dataloader.reco_q
        reco_event_dict["weight"] = reco_calibration_weights
        reco_event_dict["q_x"] = reco_q_dict["qx"]
        reco_event_dict["q_y"] = reco_q_dict["qy"]
        reco_event_dict["q_z"] = reco_q_dict["qz"]
        reco_event_dict["q_E"] = reco_q_dict["E"]
        file["reco_event"] = reco_event_dict
