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
    parser.add_argument('--nmax', type=int, default=1000000, help='Maximum number of events to load')
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
    unfolded_weights = mfold.reweight(dataloaders['Rapgap'].evts,mfold.model2_ema,batch_size=1000)
    #return unfolded_weights
    return hvd.allgather(tf.constant(unfolded_weights)).numpy()


def undo_standardizing(flags,dataloaders):
    #Undo preprocessing
    for mc in mc_file_names:
        dataloaders[mc].part, dataloaders[mc].event  = dataloaders[mc].revert_standardize(dataloaders[mc].evts[0], dataloaders[mc].evts[1],dataloaders[mc].evts[-1])
        dataloaders[mc].mask = dataloaders[mc].evts[-1]
        del dataloaders[mc].evts
        gc.collect()
    if flags.reco:
        dataloaders['data'].part, dataloaders['data'].event = dataloaders['data'].revert_standardize(dataloaders['data'].reco[0], dataloaders['data'].reco[1],dataloaders['data'].reco[-1])
        dataloaders['data'].mask = dataloaders['data'].reco[-1]

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

def plot_jet(flags,dataloaders,reference_name,version):
    #Jet level observables

    #at least 1 jet with pT above minimum cut

    weights = {
        'Rapgap_unfolded':(dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet[:,0] > 0],
    }


    feed_dict = {
        'Rapgap_unfolded': dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,0][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,0][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"Jet $p_{T}$ [GeV]",
                               weights = weights,
                               binning = np.geomspace(10,100,6),
                               logy=True,
                               logx=True,
                               reference_name = reference_name,
                               label_loc='upper left',
                               )
    fig.savefig('../plots/{}_jet_pt.pdf'.format(version))

    feed_dict = {
        'Rapgap_unfolded': dataloaders['Rapgap'].jet[:,1][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,1][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,1][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,1][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"Jet $\eta$ [GeV]",
                               weights = weights,
                               binning = np.linspace(-1,2.,7),
                               reference_name = reference_name,
                               label_loc='upper left',
                               )
    fig.savefig('../plots/{}_jet_eta.pdf'.format(version))

    def _get_deltaphi(jet,elec):
        delta_phi = np.abs(np.pi + jet[:,2] - elec[:,4])
        delta_phi[delta_phi>2*np.pi] -=  2*np.pi
        return delta_phi
    
    feed_dict = {
        'Rapgap_unfolded': _get_deltaphi(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': _get_deltaphi(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': _get_deltaphi(dataloaders['Djangoh'].jet,dataloaders['Djangoh'].event)[dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = _get_deltaphi(dataloaders['data'].jet,dataloaders['data'].event)[dataloaders['data'].jet[:,0] > 0]
                    
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"$\Delta\phi^{jet}$ [rad]",
                               weights = weights,
                               logy=True,
                               logx=True,
                               binning = np.linspace(0,1,8),
                               reference_name = reference_name,
                               label_loc='upper left',
                               )

    ax.set_ylim(1e-2,50)
    fig.savefig('../plots/{}_jet_deltaphi.pdf'.format(version))


    def _get_qtQ(jet,elec):
        qt = np.sqrt((jet[:,0]*np.cos(jet[:,2]) + elec[:,2]*np.exp(elec[:,0]/2.)*np.cos(elec[:,4]))**2 + (jet[:,0]*np.sin(jet[:,2]) + elec[:,2]*np.exp(elec[:,0]/2.)*np.sin(elec[:,4]))**2  )
        return qt/np.exp(elec[:,0]/2.)
    
    feed_dict = {
        'Rapgap_unfolded': _get_qtQ(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': _get_qtQ(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': _get_qtQ(dataloaders['Djangoh'].jet,dataloaders['Djangoh'].event)[dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = _get_qtQ(dataloaders['data'].jet,dataloaders['data'].event)[dataloaders['data'].jet[:,0] > 0]
                    
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"$q_{T}/Q$",
                               weights = weights,
                               logy=True,
                               logx=True,
                               binning = np.geomspace(1e-2,1,8),
                               reference_name = reference_name,
                               label_loc='upper left',
                               )

    ax.set_ylim(1e-2,20)
    fig.savefig('../plots/{}_jet_qtQ.pdf'.format(version))


    

                                                                
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


def cluster_jets(dataloaders):
    import fastjet
    import awkward as ak
    import itertools
    
    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
    
    def _convert_kinematics(part,event,mask):
        #return particles in cartesian coordinates
        new_particles = np.zeros((part.shape[0],part.shape[1],4))
        new_particles[:,:,0] = np.ma.exp(part[:,:,2])*np.cos(np.pi + part[:,:,1] + event[:,4,None])
        new_particles[:,:,1] = np.ma.exp(part[:,:,2])*np.sin(np.pi + part[:,:,1] + event[:,4,None])
        new_particles[:,:,2] = np.ma.exp(part[:,:,2])*np.sinh(part[:,:,0] + event[:,3,None])
        new_particles[:,:,3] = np.ma.exp(part[:,:,5])
        
        return new_particles*mask[:,:,None]
    
    for dataloader in dataloaders:
        cartesian = _convert_kinematics(dataloaders[dataloader].part,
                                        dataloaders[dataloader].event,
                                        dataloaders[dataloader].mask)
        events = []
        for event in cartesian:
            events.append([{"px": part[0], "py": part[1], "pz": part[2], "E": part[3]} for part in event if np.abs(part[0])!=0])

        array = ak.Array(events)
        cluster = fastjet.ClusterSequence(array, jetdef)
        jets = cluster.inclusive_jets(min_pt=10)
        
        jets["pt"] = -np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arcsinh(jets["pz"]/jets["pt"])
        jets=fastjet.sorted_by_pt(jets)
        

        def _take_leading_jet(jets):
            jet = np.zeros((dataloaders[dataloader].event.shape[0],4))
            jet[:,0] = -np.array(list(itertools.zip_longest(*jets.pt.to_list(), fillvalue=0))).T[:,0]
            jet[:,1] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T[:,0]
            jet[:,2] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T[:,0]
            jet[:,3] = np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T[:,0]
            return jet
            
        dataloaders[dataloader].jet = _take_leading_jet(jets)
        

        


def gather_data(dataloaders):
    for dataloader in dataloaders:
        dataloaders[dataloader].mask = np.reshape(dataloaders[dataloader].mask,(-1))
        dataloaders[dataloader].part = hvd.allgather(tf.constant(dataloaders[dataloader].part.reshape(
            (-1,dataloaders[dataloader].part.shape[-1]))[dataloaders[dataloader].mask])).numpy()
        
        dataloaders[dataloader].event = hvd.allgather(tf.constant(dataloaders[dataloader].event)).numpy()
        dataloaders[dataloader].jet = hvd.allgather(tf.constant(dataloaders[dataloader].jet)).numpy()
        dataloaders[dataloader].weight = hvd.allgather(tf.constant(dataloaders[dataloader].weight)).numpy()
        dataloaders[dataloader].mask = hvd.allgather(tf.constant(dataloaders[dataloader].mask)).numpy()
        
        
def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt=utils.LoadJson(flags.config)
    dataloaders = get_dataloaders(flags)

    reference_name, version = get_version(flags,opt)
    if flags.load:
        raise ValueError("ERROR:NOT IMPLEMENTED")
    else:        
        weights  = load_model(flags,opt,version,dataloaders)
        
    if hvd.rank()==0:
        print("Done with network evaluation")
    #Important to only undo the preprocessing after the weights are derived!
    undo_standardizing(flags,dataloaders)
    num_part = dataloaders['Rapgap'].part.shape[1]

    jet = cluster_jets(dataloaders)
    
    gather_data(dataloaders)
    
    dataloaders['Rapgap'].unfolded_weights = weights
    plot_jet(flags,dataloaders,reference_name,version)
    plot_event(flags,dataloaders,reference_name,version)
    plot_particles(flags,dataloaders,reference_name,version,num_part = num_part)

if __name__ == '__main__':
    main()



    
