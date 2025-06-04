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
from train import get_sample_name

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
    
    # parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/H1/h5', help='Folder containing data and MC files')
    parser.add_argument(
        "--dataset", default="ep", help="Choice between ep or em datasets",)
    parser.add_argument('--weights', default='../weights', help='Folder to store trained weights')
    parser.add_argument('--load_pretrain', action='store_true', default=False,help='Load pretrained model instead of starting from scratch')
    parser.add_argument('--finetuned', action='store_true', default=False,help='Load pretrained, but reset classifier head. All weight still trainable')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_argument('--reco', action='store_true', default=False,help='Plot reco level  results')
    parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights (npy files in weights dir)')
    parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
    parser.add_argument('--n_iter', type=int, default=5, help='Omnifold iteration to load (usually 0-4 for 5 iters)')
    parser.add_argument('--n_ens', type=int, default=5, help='which ensemble to load')
    parser.add_argument('--n_jobs', type=int, default=10, help='number of jobs (super-ensembels)')
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


def get_version(flags, jobID, opt):
    '''OmniFold_H1_Feb_Seed1234_emaGetSetWeighs_archL61_closure_job9_pretrained_iter4_step2_ensemble3.pkl'''

    version = opt['NAME']

    version += f'_{flags.dataset}'

    if flags.closure:
        version += '_closure'
        reference_name = 'Djangoh'
    else:
        reference_name = 'Rapgap_unfolded'

    version += f'_job{jobID}'

    if flags.load_pretrain:
        version += '_pretrained'

    elif flags.finetuned:
        version += '_finetuned'

    else:
        version += '_fromscratch'

    if flags.reco:
        reference_name = 'data'

    return reference_name, version


def load_model(flags, opt, version, dataloaders, jobID):
    # Load the trained model
    ''' {}/OmniFold_{version}_iter4_step2_ensemble4.pkl'''
    # OmniFold_H1_Feb_Seed1234_emaGetSetWeighs_archL61_closure_job9_pretrained_iter4_step2_ensemble3.pkl
    # pretrain or from scratch is appended to the version string. Either pass the flag if it exists, or
    # or that flag AND the jobID need to be inside version


    mfold = Multifold(version = version, verbose = hvd.rank()==0)
    mfold.PrepareModel()
    for ens in range(flags.n_ens):

        if flags.reco:
            if flags.n_iter > 0:
                warnings.warn('Reco level weights are only reasonable if flags.n_iter == 0')
            model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(flags.weights,version,flags.n_iter-1)

        else:
            # model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(flags.weights,version,flags.n_iter)
            model_name = '{}/OmniFold_{}_iter{}_step2_ensemble{}/checkpoint'.format(
                            flags.weights,version,flags.n_iter-1, ens)
            #n_iter - 1 for last iter (count from 0)

            #OmniFold_H1_Feb_Seed1234_emaGetSetWeighs_archL61_closure_job9_pretrained_iter4_step2_ensemble3.pkl
            #means the version must incorporate the jobArray number.
            #I want the ensemble avg for each job. Means I need to pass list of ensembles into reweight
            #see plot_superEnsembles.py


        if hvd.rank()==0:
            print(f"Loading Model {model_name}")

        ens_model = tf.keras.models.clone_model(mfold.model2)  # clones original model layers and architecture
        ens_model.load_weights(model_name).expect_partial() 

        # mfold.model2.load_weights(model_name).expect_partial() 
        # ens_model.set_weights(self.model1.get_weights())  #actually clones weights. 

        mfold.step2_models.append(ens_model)
    unfolded_weights = mfold.reweight(dataloaders['Rapgap'].evts,mfold.model2,batch_size=1000)
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

def plot_event(flags,dataloaders,reference_name,version, axes, ens):
    #Event level observables

    for feature in range(dataloaders['Rapgap'].event.shape[-1]):
        feed_dict = {
            f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].event[:,feature],
            'Rapgap': dataloaders['Rapgap'].event[:,feature],
            'Djangoh': dataloaders['Djangoh'].event[:,feature],
        }
        weights = {
            f'Rapgap_unfolded{ens}':dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights,
            'Rapgap': dataloaders['Rapgap'].weight,
            'Djangoh': dataloaders['Djangoh'].weight,
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].event[:,feature]
            weights['data'] = dataloaders['data'].weight
            
        save_str = f"_{version}_event{feature}_ens{ens}"
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.event_names[str(feature)],
                                   weights = weights,
                                   reference_name = reference_name,
                                   label_loc='upper left', axes=axes[feature], save_str=save_str)

        fig.savefig('../plots/{}_event_{}_ens_{}.pdf'.format(version,feature,ens))

def plot_jet(flags,dataloaders,reference_name,version, axes, ens):
    #Jet level observables

    #at least 1 jet with pT above minimum cut

    weights = {
        f'Rapgap_unfolded{ens}':(dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet[:,0] > 0],
    }


    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,0][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,0][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_pT_ens{ens}"
    fig, ax = utils.HistRoutine(feed_dict,
                               xlabel=r"Jet $p_{T}$ [GeV]",
                               weights = weights,
                               binning = np.geomspace(10,100,6),
                               logy=True,
                               logx=True,
                               reference_name = reference_name,
                               label_loc='upper left',
                               axes=axes[0], save_str=save_str)

    fig.savefig('../plots/{}_jet_pt_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,1][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,1][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,1][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,1][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_eta_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"Jet $\eta$ [GeV]",
                               weights = weights,
                               binning = np.linspace(-1,2.,7),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[1], save_str=save_str
                               )
    fig.savefig('../plots/{}_jet_eta_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,4][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,4][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,4][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,4][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_tau11_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"ln $\lambda_{1}^{1}$",
                               weights = weights,
                               binning = np.linspace(-4,0.,7),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[1], save_str=save_str
                               )
    fig.savefig('../plots/{}_jet_tau11_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,5][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,5][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,5][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,5][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_tau11p5_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"ln $\lambda_{1.5}^{1}$",
                               weights = weights,
                               binning = np.linspace(-4,0.,7),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[1], save_str=save_str
                               )
    fig.savefig('../plots/{}_jet_tau11p5_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,6][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:,6][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,6][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,6][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_tau12_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"ln $\lambda_{2}^{1}$",
                               weights = weights,
                               binning = np.linspace(-4,0.,7),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[1], save_str=save_str
                               )
    fig.savefig('../plots/{}_jet_tau12_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    feed_dict = {
        f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].jet[:,8][dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap':dataloaders['Rapgap'].jet[:,8][dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:,8][dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:,8][dataloaders['data'].jet[:,0] > 0]
        weights['data'] = dataloaders['data'].weight[dataloaders['data'].jet[:,0] > 0]

                    
    save_str = f"_{version}_jet_ptD_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel = r"$p_{T}D \, (\sqrt{\lambda_{0}^{2}})$",
                               weights = weights,
                               binning = np.linspace(0,1,7),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[1], save_str=save_str
                               )
    fig.savefig('../plots/{}_jet_ptD_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    def _get_deltaphi(jet,elec):
        delta_phi = np.abs(np.pi + jet[:,2] - elec[:,4])
        delta_phi[delta_phi>2*np.pi] -=  2*np.pi
        return delta_phi
    
    feed_dict = {
        f'Rapgap_unfolded{ens}': _get_deltaphi(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': _get_deltaphi(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': _get_deltaphi(dataloaders['Djangoh'].jet,dataloaders['Djangoh'].event)[dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = _get_deltaphi(dataloaders['data'].jet,dataloaders['data'].event)[dataloaders['data'].jet[:,0] > 0]
                    
    save_str = f"_{version}_jet_phi_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"$\Delta\phi^{jet}$ [rad]",
                               weights = weights,
                               logy=True,
                               logx=True,
                               binning = np.linspace(0,1,8),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[2], save_str=save_str
                               )

    ax.set_ylim(1e-2,50)
    fig.savefig('../plots/{}_jet_deltaphi_ens_{}.pdf'.format(version, ens))
    plt.close(fig)


    def _get_qtQ(jet,elec):
        qt = np.sqrt((jet[:,0]*np.cos(jet[:,2]) + elec[:,2]*np.exp(elec[:,0]/2.)*np.cos(elec[:,4]))**2 + (jet[:,0]*np.sin(jet[:,2]) + elec[:,2]*np.exp(elec[:,0]/2.)*np.sin(elec[:,4]))**2  )
        return qt/np.exp(elec[:,0]/2.)
    
    feed_dict = {
        f'Rapgap_unfolded{ens}': _get_qtQ(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Rapgap': _get_qtQ(dataloaders['Rapgap'].jet,dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:,0] > 0],
        'Djangoh': _get_qtQ(dataloaders['Djangoh'].jet,dataloaders['Djangoh'].event)[dataloaders['Djangoh'].jet[:,0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = _get_qtQ(dataloaders['data'].jet,dataloaders['data'].event)[dataloaders['data'].jet[:,0] > 0]
                    
    save_str = f"_{version}_jet_qT_ens{ens}"
    fig,ax = utils.HistRoutine(feed_dict,
                               xlabel=r"$q_{T}/Q$",
                               weights = weights,
                               logy=True,
                               logx=True,
                               binning = np.geomspace(1e-2,1,8),
                               reference_name = reference_name,
                               label_loc='upper left', axes=axes[3], save_str=save_str
                               )

    ax.set_ylim(1e-2,20)
    fig.savefig('../plots/{}_jet_qtQ_ens_{}.pdf'.format(version, ens))
    plt.close(fig)

    

                                                                
def plot_particles(flags,dataloaders,reference_name,version,num_part, axes, ens):    
    #Particle level observables
    for feature in range(dataloaders['Rapgap'].part.shape[-1]):
        feed_dict = {
            f'Rapgap_unfolded{ens}': dataloaders['Rapgap'].part[:,feature],
            'Rapgap': dataloaders['Rapgap'].part[:,feature],
            'Djangoh': dataloaders['Djangoh'].part[:,feature],
        }
        
        weights = {
            f'Rapgap_unfolded{ens}':(dataloaders['Rapgap'].weight*dataloaders['Rapgap'].unfolded_weights).reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Rapgap': dataloaders['Rapgap'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Djangoh': dataloaders['Djangoh'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['Djangoh'].mask],
        }
        
        if flags.reco:
            feed_dict['data'] = dataloaders['data'].part[:,feature]
            weights['data'] = dataloaders['data'].weight.reshape(-1,1,1).repeat(num_part,1).reshape(-1)[dataloaders['data'].mask]
                

        save_str = f"_{version}_particle{feature}_ens{ens}"
        fig,ax = utils.HistRoutine(feed_dict,
                                   xlabel=utils.particle_names[str(feature)],
                                   weights = weights,
                                   reference_name = reference_name,
                                   label_loc='upper left', axes=axes[feature], save_str=save_str
                                   )
        fig.savefig('../plots/{}_part_{}_ens_{}.pdf'.format(version,feature,ens))

def cluster_jets(dataloaders):
    """ Updating dataloaders
        1. With the jet info.
    """
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
        print(f"----------------- Started working with {dataloader} ------------------- ")
        cartesian = _convert_kinematics(dataloaders[dataloader].part,
                                        dataloaders[dataloader].event,
                                        dataloaders[dataloader].mask)
        max_len_of_jet_per_dataloader = -1
        list_of_jet =[]
        for event in cartesian:
            list_of_particles = []
            for part in event:
                if np.abs(part[0])!=0:
                    list_of_particles.append(fastjet.PseudoJet(part[0], part[1],  part[2],part[3]))
            
            cluster = fastjet.ClusterSequence(list_of_particles, jetdef)
            jets = cluster.inclusive_jets(ptmin=10)

            # Sort by pt.
            sorted_by_pt_jets=fastjet.sorted_by_pt(jets)

            for i, jet in enumerate(sorted_by_pt_jets):
                tau_11, tau_11p5, tau_12, tau_20, sumpt = 0, 0, 0, 0, 0
                constituents = jet.constituents()  # Get the constituents of the jet                                                                                                                                            
                for j, constituent in enumerate(constituents):

                    tau_11 += constituent.pt()*pow(jet.delta_R(constituent),1)
                    tau_11p5 += constituent.pt()*pow(jet.delta_R(constituent),1.5)
                    tau_12 += constituent.pt()*pow(jet.delta_R(constituent),2)
                    tau_20 += pow(constituent.pt(),2)


                    sumpt += constituent.pt()    

                # Assigning cumulative values of tau10, tau15, tau20
                jet.tau_11 = np.log(tau_11/jet.pt())
                jet.tau_11p5 = np.log(tau_11p5/jet.pt())
                jet.tau_12 = np.log(tau_12/jet.pt())
                jet.tau_20 = tau_20/pow(jet.pt(),2)
                # jet.ncharged = ncharged
                # jet.jetcharge = jet_charge/jet.pt()
                jet.ptD = np.sqrt(tau_20)/sumpt

                # print(f" jet ptD: {np.sqrt(tau_20)}")

        

            def _take_leading_jet(temp_jets):

                if (len(temp_jets) == 0): return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

                # Extract individual jet properties as lists
                pt_list = [jet.pt() for jet in temp_jets]
                eta_list = [jet.eta() for jet in temp_jets]
                phi_list = [jet.phi() for jet in temp_jets]
                E_list = [jet.E() for jet in temp_jets]
                tau_11_list = [jet.tau_11 for jet in temp_jets]
                tau_11p5_list = [jet.tau_11p5 for jet in temp_jets]
                tau_12_list = [jet.tau_12 for jet in temp_jets]
                tau_20_list = [jet.tau_20 for jet in temp_jets]
                # ncharged_list = [jet.ncharged for jet in temp_jets]
                # jetcharge_list = [jet.jetcharge for jet in temp_jets]
                ptD_list = [jet.ptD for jet in temp_jets]
                

                jet = np.zeros(9)
                
                # Just take leading values
                jet[0] = pt_list[0]
                jet[1] = eta_list[0]
                jet[2] = phi_list[0]
                jet[3] = E_list[0]
                jet[4] = tau_11_list[0]
                jet[5] = tau_11p5_list[0]
                jet[6] = tau_12_list[0]
                jet[7] = tau_20_list[0]
                jet[8] = ptD_list[0]



                return jet
            current_list_of_jets = _take_leading_jet(sorted_by_pt_jets)
            list_of_jet.append(current_list_of_jets)

        output = np.array(list_of_jet, dtype=object)
        dataloaders[dataloader].jet = output.astype(np.float32)
        print(f"----------------- Done working with {dataloader} ------------------- ")
        

        


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

    #Define figure and axis for overlay
    jet_axes = []
    for i in range(4): #plotting four
        fig, gs = utils.SetGrid(ratio=True) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex = ax0)
        jet_axes.append([ax0,ax1,fig])

    event_axes = []
    for i in range(5):
        fig, gs = utils.SetGrid(ratio=True) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex = ax0)
        event_axes.append([ax0,ax1,fig])

    part_axes = []
    for i in range(8):
        fig, gs = utils.SetGrid(ratio=True) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex = ax0)
        part_axes.append([ax0,ax1,fig])

    # for feature in range(dataloaders['Rapgap'].event.shape[-1]):


    # dataloaders = get_dataloaders(flags)
    ensemble_avg_weights = 0

    # for e in range(flags.n_ens):  # previous
    for jobID in range(flags.n_jobs):
        '''1. loop through jobs (super-ensembles)
           2. for each job, take ensemble avg (done in load_model -> reweight function)
           3. plot observables for each job
           4. plot avg of all jobs, outside of loop
        '''
        e = 'Avg' #take

        print(f"plotting ensemble {e}")

        dataloaders = get_dataloaders(flags)
        reference_name, version = get_version(flags, jobID, opt)

        if flags.load:
            # raise ValueError("ERROR:NOT IMPLEMENTED")
            print("\n--- Loading .npy weights from ../weights/ ---\n")
            weights = np.load(f"../weights/{version}_ensemble{e}_weights.npy")
        else:        
            weights = load_model(flags, opt, version, dataloaders, e)
            np.save(f"../weights/{version}_ensemble{e}_weights.npy",weights) #already re-weighted, L:109
        if hvd.rank()==0:
            print(weights[:5], weights[-5:])
            print("Mean Weights = ",np.mean(weights))

        if e ==0:
            ensemble_avg_weights = weights/flags.n_jobs
        else:
            ensemble_avg_weights += weights/flags.n_jobs
        
        if hvd.rank()==0:
            print("Done with network evaluation")

        #Important to only undo the preprocessing after the weights are derived!
        undo_standardizing(flags,dataloaders)
        num_part = dataloaders['Rapgap'].part.shape[1]

        jet = cluster_jets(dataloaders)
        
        gather_data(dataloaders)
        
        dataloaders['Rapgap'].unfolded_weights = weights 
        #^see line 157, don't need to spec. ensemble

        plot_jet(flags,dataloaders,reference_name,version, jet_axes, e)
        plot_event(flags,dataloaders,reference_name,
                   version, axes=event_axes, ens=e)
        plot_particles(flags,dataloaders,reference_name,version,
                       num_part = num_part, axes=part_axes, ens=e)


    #Plot averages
    dataloaders['Rapgap'].unfolded_weights = ensemble_avg_weights 
    version = "_".join("_jobAvg_" if p.startswith("job") and p[3:].isdigit() else p for p in version.split("_"))
    print(f"OUTSIDE OF LOOP, VERSION = {version}")
    plot_jet(flags,dataloaders,reference_name,version, jet_axes, 'Avg')
    plot_event(flags,dataloaders,reference_name,
               version, axes=event_axes, ens='Avg')
    plot_particles(flags,dataloaders,reference_name,version,
                   num_part = num_part, axes=part_axes, ens='Avg')

    # plot_stdv(reference_name, 'jets', jet_axes)

if __name__ == '__main__':
    main()
