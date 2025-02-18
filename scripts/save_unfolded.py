import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import os,gc
from omnifold import  Multifold
import tensorflow as tf
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
import warnings
from plot_utils import *
import h5py as h5
hvd.init()




def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
    parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
    #parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
    parser.add_argument('--file', default='Rapgap_Eplus0607_prep.h5',help='File to load')
    parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
    parser.add_argument('--nmax', type=int, default=20_000_000, help='Maximum number of events to load')
    parser.add_argument('--verbose', action='store_true', default=False,help='Increase print level')
    
    flags = parser.parse_args()

    return flags


def get_dataloaders(flags,file_names):
    #Load the data from the simulations
    dataloaders = {}

    for name in file_names:
        if 'data' in name:
            dataloaders[name] = Dataset([name],flags.data_folder,is_mc=False,
                                    rank=hvd.rank(),size=hvd.size(),
                                    nmax=None,pass_reco=True)
            
        else:    
            dataloaders[name] = Dataset([name],flags.data_folder,is_mc=True,
                                        rank=hvd.rank(),size=hvd.size(),
                                        nmax=flags.nmax,pass_fiducial= not flags.reco,
                                        pass_reco = flags.reco)
            
    return dataloaders


def get_deltaphi(jet, elec):
    delta_phi = np.abs(np.pi + jet[:, 2] - elec[:, 4])
    delta_phi[delta_phi > 2 * np.pi] -= 2 * np.pi
    return delta_phi

                                                                            
def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt=utils.LoadJson(flags.config)
    mc_files = [flags.file]
    
    if flags.verbose and hvd.rank()==0:
        print(f'Will load the following files : {mc_files.keys()}')
        
    dataloaders = get_dataloaders(flags,mc_files)

    if 'data' not in flags.file:
        weights = {}
        for dataset in dataloaders:
            if flags.verbose and hvd.rank()==0:
                print(f"Evaluating weights for dataset {dataset}")
            weights[dataset] = evaluate_model(flags,opt,dataset,dataloaders)

            if "Rapgap" in flags.file and 'sys' not in flags.file:
                weights['closure'] = evaluate_model(
                    flags,opt,dataset,dataloaders,
                    version = opt['NAME']+'_closure'+'_pretrained' if flags.load_pretrain else '')
        
    if hvd.rank()==0:
        print("Done with network evaluation")
    #Important to only undo the preprocessing after the weights are derived!
    undo_standardizing(flags,dataloaders)
    #num_part = dataloaders['Rapgap'].part.shape[1]
    
    cluster_breit(dataloaders)
    cluster_jets(dataloaders)
    del dataloaders[flags.file].part, dataloaders[flags.file].mask
    gc.collect()
    gather_data(dataloaders)
    replace_string = f"unfolded_{flags.niter}"
    if flags.reco:
        replace_string += '_reco'
    output_file_name = flags.file.replace("prep",replace_string)


    if hvd.rank()==0:
        with h5.File(os.path.join(flags.data_folder,output_file_name),'w') as fh5:
            if 'data' not in flags.file:
                dset = fh5.create_dataset('weights', data=weights[flags.file])
                dset = fh5.create_dataset('mc_weights', data=dataloaders[flags.file].weight)
                if 'closure' in weights:
                    dset = fh5.create_dataset('closure_weights', data=weights['closure'])

                
            dset = fh5.create_dataset('jet_pt', data=dataloaders[flags.file].jet[:,0])
            dset = fh5.create_dataset('jet_breit_pt', data=dataloaders[flags.file].jet_breit[:,0])
            dset = fh5.create_dataset('deltaphi', data=get_deltaphi(dataloaders[flags.file].jet, dataloaders[flags.file].event))
            dset = fh5.create_dataset('jet_tau10', data=dataloaders[flags.file].jet[:,4])



    
    

if __name__ == '__main__':
    main()



    
