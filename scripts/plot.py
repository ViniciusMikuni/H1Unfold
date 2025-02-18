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
hvd.init()
utils.SetStyle()



def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
    parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
    #parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
    parser.add_argument('--sys', action='store_true', default=False,help='Load systematic variations')
    parser.add_argument('--blind', action='store_true', default=False,help='Show the results based on closure instead of data')
    #parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
    parser.add_argument('--niter', type=int, default=0, help='Omnifold iteration to load')
    parser.add_argument('--plot_zjet', action='store_true', default=False, help='Plot zjet using all jets')
    parser.add_argument('--nmax', type=int, default=1000000, help='Maximum number of events to load')
    parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')
    parser.add_argument('--verbose', action='store_true', default=False,help='Increase print level')
    
    flags = parser.parse_args()

    if flags.blind and flags.reco:
        raise ValueError("Unable to run blinded and reco modes at the same time")
    return flags


def get_dataloaders(flags,mc_file_names):
    #Load the data from the simulations
    dataloaders = {}
    for mc in mc_file_names:
        if flags.reco:
            dataloaders[mc] = Dataset([mc_file_names[mc]],flags.data_folder,is_mc=True,
                                      rank=hvd.rank(),size=hvd.size(),nmax=flags.nmax,pass_reco=True)

            del dataloaders[mc].gen #free a bit of memory
            dataloaders[mc].evts = dataloaders[mc].reco
            del dataloaders[mc].reco
        else:
            dataloaders[mc] = Dataset([mc_file_names[mc]],flags.data_folder,is_mc=True,
                                      rank=hvd.rank(),size=hvd.size(),nmax=flags.nmax,pass_fiducial=True)

            del dataloaders[mc].reco #free a bit of memory
            dataloaders[mc].evts = dataloaders[mc].gen
            del dataloaders[mc].gen
        gc.collect()

    if flags.reco:
        dataloaders['data'] = Dataset(['data_Eplus0607_prep.h5'],flags.data_folder,is_mc=False,
                                      rank=hvd.rank(),size=hvd.size(),nmax=None,pass_reco=True)

    return dataloaders



                                                                            
def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt=utils.LoadJson(flags.config)
    mc_files = get_sample_names(use_sys=flags.sys)
    if flags.verbose and hvd.rank()==0:
        print(f'Will load the following files : {mc_files.keys()}')
        
    dataloaders = get_dataloaders(flags,mc_files)
    weights = {}
    for dataset in dataloaders:
        if flags.verbose and hvd.rank()==0:
            print(f"Evaluating weights for dataset {dataset}")
        weights[dataset] = evaluate_model(flags,opt,dataset,dataloaders)
        
    weights['closure'] = evaluate_model(
        flags,opt,'Rapgap',dataloaders,
        version = opt['NAME']+'_closure'+'_pretrained' if flags.load_pretrain else '')
        
    if hvd.rank()==0:
        print("Done with network evaluation")
    #Important to only undo the preprocessing after the weights are derived!
    undo_standardizing(flags,dataloaders)
    num_part = dataloaders['Rapgap'].part.shape[1]
    
    # cluster_breit(dataloaders, store_all_jets = flags.plot_zjet)
    cluster_jets(dataloaders, store_all_jets = flags.plot_zjet)
    cluster_breit(dataloaders, clustering_algorithm="kt", store_all_jets = flags.plot_zjet)
    gather_data(dataloaders, store_all_jets = flags.plot_zjet)
    plot_particles(flags,dataloaders,weights,opt['NAME'],num_part = num_part)
    plot_jet_pt(flags,dataloaders,weights,opt['NAME'],lab_frame=False)
    plot_jet_pt(flags,dataloaders,weights,opt['NAME'])
    
    plot_deltaphi(flags,dataloaders,weights,opt['NAME'])
    plot_tau(flags,dataloaders,weights,opt['NAME'])
    plot_zjet(flags,dataloaders,weights,opt['NAME'], frame = "lab", clustering = "kt")
    plot_zjet(flags,dataloaders,weights,opt['NAME'], frame = "breit", clustering = "centauro")
    plot_event(flags,dataloaders,weights,opt['NAME'])    


    
    

if __name__ == '__main__':
    main()



    
