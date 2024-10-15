from dataloader import Dataset
import numpy as np
import vector
import matplotlib.pyplot as plt
import awkward as ak
import itertools
import uproot
import subprocess

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
    base_path = '/global/cfs/cdirs/m3246/H1/h5/'
    file_data = ['Rapgap_Eplus0607_prep.h5']
    # Taking all MC events, not just the ones that pass fiduucial cuts
    dataloader_MC = Dataset(file_data, base_path, is_mc=True, pass_fiducial=False, nmax=50000)

    print("Loaded {} data events".format(dataloader_MC.gen[0].shape[0]))
        
    #Undo the preprocessing
    particles_MC, events_MC, mask_MC  = dataloader_MC.gen
    particles_MC, events_MC = dataloader_MC.revert_standardize(particles_MC, events_MC, mask_MC)
    MC_weight = dataloader_MC.weight # These weights are calibration constants that should be applied

    # Getting the four vectors of our hadronic final states and scattered electron
    data_cartesian = _convert_kinematics(particles_MC, events_MC, mask_MC)
    data_electron_momentum = _convert_electron_kinematics(events_MC)
    # Boosting particles to Breit frame
    boosted_vectors = boost_particles(data_cartesian, data_electron_momentum)

    # Extracting individual components of 4-vectors
    boosted_px, boosted_py, boosted_pz, boosted_E = [], [], [], []
    for event in boosted_vectors:
        boosted_px.append([particle_vector.px for particle_vector in event])
        boosted_py.append([particle_vector.py for particle_vector in event])
        boosted_pz.append([particle_vector.pz for particle_vector in event])
        boosted_E.append([particle_vector.E for particle_vector in event])
    boosted_px = ak.Array(boosted_px)
    boosted_py = ak.Array(boosted_py)
    boosted_pz = ak.Array(boosted_pz)
    boosted_E = ak.Array(boosted_E)

    # Saving 4-vectors and passing it to Centauro jet-clustering
    with uproot.recreate("./breit_particles.root") as file:
        file["particles"] = {"px": boosted_px, "py": boosted_py, "pz": boosted_pz, "energy": boosted_E}
    subprocess.run(["./run_centauro.sh"], shell=True)
    with uproot.open("{file}:jets".format(file="./centauro_jets.root")) as out:
        jets = out.arrays(["eta", "px", "py", "pz", "E", "pT", "phi"])
    
    # Saving clustered jets and event kinematics
    with uproot.recreate("./breit_output.root") as file:
        file["jets"] = {"eta": jets["eta"], "px": jets["px"], "py": jets["py"], "pz": jets["pz"], "E": jets["E"], "pT": jets["pT"], "phi": jets["phi"]}
        file["event"] = {"Q2": np.exp(events_MC[:, 0]), "weight":MC_weight}