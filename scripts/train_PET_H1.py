from omnifold import DataLoader, MultiFold, PET
from dataloader import Dataset
import h5py as h5
import argparse 
import numpy as np
import gzip
import pickle
import gc
import horovod.tensorflow.keras as hvd
hvd.init()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a PET model using Pythia and Herwig data.")
    parser.add_argument("--data_dir", type=str, default="/global/homes/r/rmilton/m3246/rmilton/H1Unfold/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/global/homes/r/rmilton/m3246/rmilton/H1Unfold/weights/", help="Folder to store trained model weights")
    parser.add_argument("--synthetic_data", type=str, default="Rapgap", help="The dataset that will be the synthetic (MC) data. Rapgap/Djangoh")
    parser.add_argument("--nature_data", type=str, default="H1", help="The dataset that will be the actual data. H1/Rapgap/Djangoh")
    parser.add_argument("--num_synthetic_data", type=int, default=4_000_000, help="Number of MC data to train with")
    parser.add_argument("--num_nature_data", type=int, default=-1, help="Number of nature data to train with")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of iterations to use during training")
    parser.add_argument("--start_N", type=int, default=0, help="Number of iterations to start with")
    parser.add_argument("--model_name", type=str, default="H1_OmniFold_model", help="Name to be used for model")
    args = parser.parse_args()
    return args

# Putting electron info in arrays [eta, phi, log(pT), 0, 0, log(E), 0, charge] with shape (Nevents, 1, 8)
def extract_electron_info(events):
    pT = events[:, 2]*np.sqrt(np.exp(events[:, 0])) # pT/Q -> pT
    eta = events[:, 3]
    log_energy = np.log(np.cosh(eta)*pT)
    phi = events[:, 4]
    log_pT = np.log(pT) 
    charge = -1*np.ones(events.shape[0])
    zeros = np.zeros(events.shape[0])
    return np.expand_dims(np.stack([eta, phi, log_pT, zeros, zeros, log_energy, zeros, charge], axis=-1), axis=1)

def standardize(data):
    mean_part = [0.0 , 0.0, -0.761949, -3.663438,
                        -2.8690917,0.03239748, 3.9436243, 0.0]
    std_part =  [1.0, 1.0, 1.0133458, 1.03931,
                        1.0040112, 0.98908925, 1.2256976, 1.0] 
    new_p = data
    mask = new_p[:,:,2]!=0
    p = mask[:,:,None]*(new_p-mean_part)/std_part
    return p

def main():
    flags = parse_arguments()
    itnum = flags.num_iterations
    data_dir = flags.data_dir
    file_data_dict = {"H1": 'data_Eplus0607_prep_train.h5', "Rapgap": 'Rapgap_Eplus0607_prep_train_standardized.h5', "Djangoh": 'Djangoh_Eplus0607_prep_train_5mil_standardized.h5'}

    num_synthetic_data = flags.num_synthetic_data
    num_nature_data = flags.num_nature_data

    synthetic_file_path = data_dir + file_data_dict[flags.synthetic_data]
    nature_file_path = data_dir + file_data_dict[flags.nature_data]

    with h5.File(synthetic_file_path, 'r') as synthetic:
        synthetic_gen_parts = synthetic['gen_particle_features'][:num_synthetic_data]
        synthetic_reco_parts = synthetic['reco_particle_features'][:num_synthetic_data]
        synthetic_pass_reco  = synthetic['reco_event_features'][:num_synthetic_data, -1]
        synthetic_weights    = synthetic['reco_event_features'][:num_synthetic_data, -2]
    with h5.File(nature_file_path, 'r') as nature:
        nature_reco_parts = nature['reco_particle_features'][:num_nature_data]
        nature_pass_reco  = nature['reco_event_features'][:num_nature_data, -1]
        nature_weights = nature['reco_event_features'][:num_nature_data, -2]
    
    synthetic_parts_dataloader = DataLoader(reco = synthetic_reco_parts,
                                            gen = synthetic_gen_parts,
                                            pass_reco = synthetic_pass_reco,
                                            weight = synthetic_weights,
                                            normalize = True,
                                            normalization_factor=nature_reco_parts.shape[0],
                                            rank=hvd.rank(),
                                            size=hvd.size())
    nature_parts_dataloader = DataLoader(reco = nature_reco_parts,
                                        pass_reco = nature_pass_reco,
                                        weight = nature_weights,
                                        normalize = True,
                                        normalization_factor=nature_reco_parts.shape[0],
                                        rank=hvd.rank(),
                                        size=hvd.size())
    step1_model = PET(synthetic_reco_parts.shape[2], num_part=synthetic_reco_parts.shape[1], num_heads = 4, num_transformer = 4, local = True, projection_dim = 128, K = 10)
    step2_model = PET(synthetic_gen_parts.shape[2], num_part=synthetic_gen_parts.shape[1], num_heads = 4, num_transformer = 4, local = True, projection_dim = 128, K = 10)
    model_name = flags.model_name
    omnifold_PET = MultiFold(
        model_name,
        model_reco = step1_model,
        model_gen = step2_model,
        data = nature_parts_dataloader,
        mc = synthetic_parts_dataloader,
        niter = itnum,
        weights_folder = flags.save_dir,
        verbose=True,
        batch_size = 128,
        early_stop=3,
        start = flags.start_N,
        rank=hvd.rank(),
        size=hvd.size()
    )
    omnifold_PET.Unfold()

if __name__ == '__main__':
    main()
