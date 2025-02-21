import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import os,gc
import utils

import warnings
from plot_utils import *
import h5py as h5

utils.SetStyle()

var_names = ['weights','mc_weights','jet_pt',
             'jet_breit_pt','deltaphi','jet_tau10', 'zjet', 'zjet_breit', 'zjet_centauro', 'Delta_zjet']


def get_sample_names(niter, use_sys, sys_list = ['sys0','sys1','sys5','sys7','sys11'],
                     nominal = 'Rapgap',period = 'Eplus0607',reco=False,bootstrap=False,nboot=1):
    add_string = '_reco' if reco else ''
    mc_file_names = {
        'Rapgap':f'Rapgap_{period}_unfolded_{niter}{add_string}_chargefixed.h5',
        'Djangoh':f'Djangoh_{period}_unfolded_{niter}{add_string}_chargefixed.h5',
    }
    if reco:
        mc_file_names['data'] = f'data_{period}_unfolded_{niter}{add_string}.h5'

    if use_sys:
        for sys in sys_list:
            mc_file_names[f'{sys}'] = f'{nominal}_{period}_{sys}_unfolded_{niter}{add_string}.h5'

    if bootstrap:
        mc_file_names['bootstrap'] = f'{nominal}_{period}_unfolded_{niter}_boot.h5'
            
    return mc_file_names


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
    parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
    parser.add_argument('--sys', action='store_true', default=False,help='Load systematic variations')
    parser.add_argument('--blind', action='store_true', default=False,help='Show the results based on closure instead of data')
    parser.add_argument('--niter', type=int, default=4, help='Omnifold iteration to load')
    parser.add_argument('--bootstrap', action='store_true', default=False,help='Load models for bootstrapping')
    parser.add_argument('--nboot', type=int, default=50, help='Number of bootstrap models to load')

    parser.add_argument('--verbose', action='store_true', default=False,help='Increase print level')
    
    flags = parser.parse_args()

    if flags.blind and flags.reco:
        raise ValueError("Unable to run blinded and reco modes at the same time")
    return flags


def get_dataloaders(flags,mc_file_names):
    #Load the data from the simulations
    dataloaders = {}
    
    for mc in mc_file_names:
        if flags.verbose: print(mc)        
        dataloaders[mc] = {}
        with h5.File(os.path.join(flags.data_folder,mc_file_names[mc]),'r') as fh5:
            for var in fh5.keys():
                dataloaders[mc][var] = fh5[var][:]
    return dataloaders

                                                                            
def main():

    flags = parse_arguments()
    opt=utils.LoadJson(flags.config)
    mc_files = get_sample_names(niter=flags.niter,use_sys=flags.sys,
                                reco=flags.reco,bootstrap=flags.bootstrap,nboot=flags.nboot)
    if flags.verbose:
        print(f'Will load the following files : {mc_files.keys()}')
        
    dataloaders = get_dataloaders(flags,mc_files)
    
    for var in var_names:
        if 'weight' in var:continue
        plot_observable(flags,var,dataloaders,opt['NAME'])
        
    
if __name__ == '__main__':
    main()



    
