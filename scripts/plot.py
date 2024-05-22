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
import horovod.tensorflow.keras as hvd
import warnings

hvd.init()
utils.SetStyle()

mc_file_names = {
    'Rapgap':'Rapgap_Eplus0607_prep.h5',
    'Djangoh':'Djangoh_Eplus0607_prep.h5',
    # 'Rapgap':'toy2.h5',
    # 'Djangoh':'toy1.h5',mask
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
    #parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
    parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
    parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
    parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
    parser.add_argument('--niter', type=int, default=0, help='Omnifold iteration to load')
    parser.add_argument('--img_fmt', default='pdf', help='Format of the output figures')
    
    flags = parser.parse_args()

    if flags.closure and flags.reco:
        raise ValueError("Unable to run both closure and reco modes at the same time")
    return flags


def get_dataloaders(flags):
    #Load the data from the simulations
    dataloaders = {}
    for mc in mc_file_names:
        if flags.reco:
            dataloaders[mc] = Dataset([mc_file_names[mc]],flags.data_folder,is_mc=True,
                                      rank=hvd.rank(),size=hvd.size(),nmax=1000000,pass_reco=True)

            del dataloaders[mc].gen #free a bit of memory
            dataloaders[mc].evts = dataloaders[mc].reco
        else:
            dataloaders[mc] = Dataset([mc_file_names[mc]],flags.data_folder,is_mc=True,
                                      rank=hvd.rank(),size=hvd.size(),nmax=5000000,pass_fiducial=True)

            del dataloaders[mc].reco #free a bit of memory
            dataloaders[mc].evts = dataloaders[mc].gen

        # dataloaders[mc].evts[0] = dataloaders[mc].evts[0][dataloaders[mc].pass_evts]
        # dataloaders[mc].evts[1] = dataloaders[mc].evts[1][dataloaders[mc].pass_evts]
        # dataloaders[mc].weight = dataloaders[mc].weight[dataloaders[mc].pass_evts]
        gc.collect()

    if flags.reco:
        dataloaders['data'] = Dataset(['data_prep.h5'],flags.data_folder,is_mc=False,
                                      rank=hvd.rank(),size=hvd.size(),nmax=None,pass_reco=True)

    return dataloaders


def get_version(flags,opt):
    version = opt['NAME']
    if flags.closure:
        version +='_closure'
        reference_name = 'Djangoh'
    else:
        reference_name = 'Rapgap_unfolded'
        
    if flags.load_pretrain:
        version += '_pretrained'
    if flags.reco:
        reference_name = 'data'
        
    return reference_name, version

def load_model(flags,opt,version,dataloaders):
    #Load the trained model        
    if flags.reco:
        if flags.niter>0:
            warnings.warn('Reco level weights are only reasonable if flags.niter == 0')
        model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(flags.weights,version,flags.niter)
    else:
        model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(flags.weights,version,flags.niter)
    if hvd.rank()==0:
        print("Loading model {}".format(model_name))

    mfold = Multifold(version = version,verbose = hvd.rank()==0)
    mfold.PrepareModel()
    mfold.model2.load_weights(model_name).expect_partial() #Doesn't matter which model is loaded since both have the same architecture
    unfolded_weights = mfold.reweight(dataloaders['Rapgap'].evts,mfold.model2_ema,batch_size=500)
    #print(unfolded_weights)
    return hvd.allgather(tf.constant(unfolded_weights)).numpy()


def undo_preprocessing(flags,dataloaders):
    #Undo preprocessing
    for mc in mc_file_names:
        dataloaders[mc].part, dataloaders[mc].event = dataloaders[mc].revert_standardize(dataloaders[mc].evts[0], dataloaders[mc].evts[1],dataloaders[mc].evts[-1])
    if flags.reco:
        dataloaders['data'].part, dataloaders['data'].event = dataloaders['data'].revert_standardize(dataloaders['data'].reco[0], dataloaders['data'].reco[1],dataloaders['data'].reco[-1])


def plot_event(flags,dataloaders,reference_name,version):
    #Event level observables

    for feature in range(dataloaders['Rapgap'].event.shape[-1]):
        feed_dict = {
            'Rapgap_unfolded': dataloaders['Rapgap'].event[:,feature],
            'Rapgap': dataloaders['Rapgap'].event[:,feature],
            'Djangoh': dataloaders['Djangoh'].event[:,feature],
        }
        weights = {
            'Rapgap_unfolded':dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights,
            'Rapgap': dataloaders['Rapgap'].weight,
            'Djangoh': dataloaders['Djangoh'].weight,
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].event[:,feature]
            weights['data'] = dataloaders['data'].weight
            
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.event_names[str(feature)],
                                   weights = weights,
                                   reference_name = reference_name,
                                   label_loc='upper left',
                                   )
        fig.savefig('../plots/{}_event_{}.pdf'.format(version,feature))

                                                                
def plot_particles(flags,dataloaders,reference_name,version,num_part):    
    #Particle level observables
    for feature in range(dataloaders['Rapgap'].part.shape[-1]):
        feed_dict = {
            'Rapgap_unfolded': dataloaders['Rapgap'].part[:,feature],
            'Rapgap': dataloaders['Rapgap'].part[:,feature],
            'Djangoh': dataloaders['Djangoh'].part[:,feature],
        }
        
        weights = {
            'Rapgap_unfolded':(dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights).reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Rapgap': dataloaders['Rapgap'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Djangoh': dataloaders['Djangoh'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Djangoh'].mask],
        }
        
        if flags.reco:
            feed_dict['data'] = dataloaders['data'].part[:,feature]
            weights['data'] = dataloaders['data'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['data'].mask]
                

        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.particle_names[str(feature)],
                                   weights = weights,
                                   reference_name = reference_name,
                                   label_loc='upper left',
                                   )
        fig.savefig('../plots/{}_part_{}.pdf'.format(version,feature))


def gather_data(dataloaders):
    for dataloader in dataloaders:
        mask = np.reshape(dataloaders[dataloader].part[:,:,0]!= 0 , (-1))
        dataloaders[dataloader].part = hvd.allgather(tf.constant(dataloaders[dataloader].part.reshape((-1,dataloaders[dataloader].part.shape[-1]))[mask])).numpy()
        dataloaders[dataloader].event = hvd.allgather(tf.constant(dataloaders[dataloader].event)).numpy()
        dataloaders[dataloader].weight = hvd.allgather(tf.constant(dataloaders[dataloader].weight)).numpy()
        dataloaders[dataloader].mask = hvd.allgather(tf.constant(mask)).numpy()

        
        
def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt=utils.LoadJson(flags.config)
    dataloaders = get_dataloaders(flags)

    reference_name, version = get_version(flags,opt)
    if flags.load:
        pass
    else:        
        weights  = load_model(flags,opt,version,dataloaders)
        

    #Important to only undo the preprocessing after the weights are derived!
    undo_preprocessing(flags,dataloaders)
    num_part = dataloaders['Rapgap'].part.shape[1]
    #if hvd.rank()==0:
    gather_data(dataloaders)
    dataloaders['Rapgap'].unfolded_weights = weights    
    plot_event(flags,dataloaders,reference_name,version)
    plot_particles(flags,dataloaders,reference_name,version,num_part = num_part)

if __name__ == '__main__':
    main()



    
