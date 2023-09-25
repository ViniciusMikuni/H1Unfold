import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import os,gc
from omnifold import  Multifold

from dataloader import TFDataset
import utils

utils.SetStyle()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
#parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')

parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--niter', type=int, default=1, help='Omnifold iteration to load')
parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')

flags = parser.parse_args()
opt=utils.LoadJson(flags.config)

mc_file_names = {
    'Rapgap':'Rapgap_Eplus0607.h5',
    'Djangoh':'Djangoh_Eplus0607.h5',
    # 'Rapgap':'toy2.h5',
    # 'Djangoh':'toy1.h5',
}

#Load the data from the simulations
dataloaders = {}
for mc in mc_file_names:
    dataloaders[mc] = TFDataset([mc_file_names[mc]],flags.data_folder,is_mc=True,nmax=1000000)
    del dataloaders[mc].reco #free a bit of memory
    gc.collect()

#Load the trained model
version = opt['NAME']
if flags.closure:
    version  +='_closure'
model_name = '{}/Omnifold_{}_iter{}_step2.h5'.format(flags.weights,version,flags.niter)
print("Loading model {}".format(model_name))
mfold = Multifold(version = version)
mfold.PrepareModel()
mfold.model2.load_weights(model_name)
unfolded_weights = mfold.reweight(dataloaders['Rapgap'].gen,mfold.model2,batch_size=1000)
print(unfolded_weights)
#Event level observables
for feature in range(mfold.num_event):
    feed_dict = {
        'data': dataloaders['Rapgap'].gen[1][:,feature][dataloaders['Rapgap'].pass_gen],
        'Rapgap': dataloaders['Rapgap'].gen[1][:,feature][dataloaders['Rapgap'].pass_gen],
        'Djangoh': dataloaders['Djangoh'].gen[1][:,feature][dataloaders['Djangoh'].pass_gen],
    }
    weights = {
        'data':(dataloaders['Rapgap'].weight*unfolded_weights)[dataloaders['Rapgap'].pass_gen],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].pass_gen],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].pass_gen],
    }
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=utils.event_names[str(feature)],
                               weights = weights,
                               label_loc='upper left',
                               )
    fig.savefig('../plots/{}_event_{}.pdf'.format(version,feature))
    
#Particle level observables
for feature in range(mfold.num_feat):
    feed_dict = {
        'data': dataloaders['Rapgap'].gen[0][:,:,feature][dataloaders['Rapgap'].pass_gen],
        'Rapgap': dataloaders['Rapgap'].gen[0][:,:,feature][dataloaders['Rapgap'].pass_gen],
        'Djangoh': dataloaders['Djangoh'].gen[0][:,:,feature][dataloaders['Djangoh'].pass_gen],
    }
                
    weights = {
        'data':(dataloaders['Rapgap'].weight*unfolded_weights)[dataloaders['Rapgap'].pass_gen],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].pass_gen],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].pass_gen],
    }

    
    #flatten and remove zeros
    for entry in feed_dict:
        weights[entry] = weights[entry].reshape(-1,1,1).repeat(feed_dict[entry].shape[1],1).reshape(-1,1)
        feed_dict[entry] = feed_dict[entry].reshape(-1,1)
        mask = feed_dict[entry] !=0
        # if feature ==0:
        #     mask = feed_dict[entry] !=0
        feed_dict[entry] = feed_dict[entry][mask]
        weights[entry] = weights[entry][mask]


    
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=utils.particle_names[str(feature)],
                               weights = weights,
                               label_loc='upper left',
                               )
    fig.savefig('../plots/{}_part_{}.pdf'.format(version,feature))
