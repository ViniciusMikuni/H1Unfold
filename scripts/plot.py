import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import os,gc
from omnifold import  Multifold

from dataloader import Dataset
import utils

utils.SetStyle()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
#parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
parser.add_argument('--niter', type=int, default=0, help='Omnifold iteration to load')
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
    dataloaders[mc] = Dataset([mc_file_names[mc]],flags.data_folder,is_mc=True,nmax=1000000)
    if flags.reco:
        del dataloaders[mc].gen #free a bit of memory
        dataloaders[mc].evts = dataloaders[mc].reco
        dataloaders[mc].pass_evts = dataloaders[mc].pass_reco
    else:
        del dataloaders[mc].reco #free a bit of memory
        dataloaders[mc].evts = dataloaders[mc].gen
        dataloaders[mc].pass_evts = dataloaders[mc].pass_gen
    gc.collect()

if flags.reco:
    dataloaders['data'] = Dataset(['data.h5'],flags.data_folder,is_mc=False,nmax=None)
    

#Load the trained model
version = opt['NAME']
if flags.closure:
    version +='_closure'
if flags.load_pretrain:
    version += '_pretrained'
if flags.reco:
    model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(flags.weights,version,flags.niter)
    print("Loading model {}".format(model_name))
else:
    model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(flags.weights,version,flags.niter)
print("Loading model {}".format(model_name))


mfold = Multifold(version = version)
mfold.PrepareModel()
mfold.model2.load_weights(model_name).expect_partial() #Doesn't matter which model is loaded since both have the same architecture
unfolded_weights = mfold.reweight(dataloaders['Rapgap'].evts,mfold.model2_ema,batch_size=1000)
print(unfolded_weights)

#Event level observables
for feature in range(mfold.num_event):
    feed_dict = {
        'Rapgap_unfolded': dataloaders['Rapgap'].evts[1][:,feature][dataloaders['Rapgap'].pass_evts],
        'Rapgap': dataloaders['Rapgap'].evts[1][:,feature][dataloaders['Rapgap'].pass_evts],
        'Djangoh': dataloaders['Djangoh'].evts[1][:,feature][dataloaders['Djangoh'].pass_evts],
    }
    weights = {
        'Rapgap_unfolded':(dataloaders['Rapgap'].weight*unfolded_weights)[dataloaders['Rapgap'].pass_evts],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].pass_evts],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].pass_evts],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].reco[1][:,feature][dataloaders['data'].pass_reco]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].pass_reco]
        
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=utils.event_names[str(feature)],
                               weights = weights,
                               label_loc='upper left',
                               )
    fig.savefig('../plots/{}_event_{}.pdf'.format(version,feature))
    
#Particle level observables
for feature in range(mfold.num_feat):
    feed_dict = {
        'Rapgap_unfolded': dataloaders['Rapgap'].evts[0][:,:,feature][dataloaders['Rapgap'].pass_evts],
        'Rapgap': dataloaders['Rapgap'].evts[0][:,:,feature][dataloaders['Rapgap'].pass_evts],
        'Djangoh': dataloaders['Djangoh'].evts[0][:,:,feature][dataloaders['Djangoh'].pass_evts],
    }
                
    weights = {
        'Rapgap_unfolded':(dataloaders['Rapgap'].weight*unfolded_weights)[dataloaders['Rapgap'].pass_evts],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].pass_evts],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].pass_evts],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].reco[0][:,:,feature][dataloaders['data'].pass_reco]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].pass_reco]
    
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
