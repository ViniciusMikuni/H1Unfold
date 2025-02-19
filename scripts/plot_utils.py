import numpy as np
import matplotlib.pyplot as plt
import os,gc, re
from omnifold import  Multifold
import tensorflow as tf
import utils
import horovod.tensorflow as hvd
import warnings
import awkward as ak
import uproot
import subprocess



def get_sample_names(use_sys, sys_list = ['sys0','sys1','sys5','sys7','sys11'],
                     nominal = 'Rapgap',period = 'Eplus0607'):
    mc_file_names = {
        'Rapgap':f'Rapgap_{period}_prep.h5',
        'Djangoh':f'Djangoh_{period}_prep.h5',
    }

    if use_sys:
        for sys in sys_list:
            mc_file_names[f'{sys}'] = f'{nominal}_{period}_{sys}_prep.h5'
    return mc_file_names


def get_version(dataset,flags,opt):
    version = opt['NAME']
    if 'sys' in dataset:
        match = re.search(r'sys(.)', dataset)
        version += f"_sys{match.group(1)}"        
        #version += f'_{dataset}'
    if 'Djangoh' in dataset:
        #Djangoh used for model uncertainty
        version += f'_sys_model'
    if flags.load_pretrain:
        version += '_pretrained'        
    return version

    # if flags.closure:
    #     version +='_closure'
    #     reference_name = 'Djangoh'
    # else:
    #     reference_name = 'Rapgap_unfolded'
        
    # if flags.load_pretrain:
    #     version += '_pretrained'
    # if flags.reco:
    #     reference_name = 'data'
        
    # return reference_name, version



def evaluate_model(flags,opt,dataset,dataloaders,version=None,bootstrap = False,nboot=0):
    if version is None:
        version = get_version(dataset,flags,opt)
    
    model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(flags.weights,version,flags.niter)
    if bootstrap:
        model_name = '{}/OmniFold_{}_iter{}_step2_strap{}/checkpoint'.format(flags.weights,version,flags.niter,nboot)    
    
    if hvd.rank()==0:
        print("Loading model {}".format(model_name))

    mfold = Multifold(version = version,verbose = hvd.rank()==0)
    mfold.PrepareModel()
    mfold.model2.load_weights(model_name).expect_partial() #Doesn't matter which model is loaded since both have the same architecture
    unfolded_weights = mfold.reweight(dataloaders[dataset].gen,mfold.model2_ema,batch_size=1000)
    #return unfolded_weights
    return hvd.allgather(tf.constant(unfolded_weights)).numpy()



def undo_standardizing(flags,dataloaders):
    #Undo preprocessing
    for mc in dataloaders:
        if flags.reco:
            dataloaders[mc].part, dataloaders[mc].event  = dataloaders[mc].revert_standardize(dataloaders[mc].reco[0], dataloaders[mc].reco[1],dataloaders[mc].reco[-1])
            dataloaders[mc].mask = dataloaders[mc].reco[-1]
            del dataloaders[mc].reco
        else:
            dataloaders[mc].part, dataloaders[mc].event  = dataloaders[mc].revert_standardize(dataloaders[mc].gen[0], dataloaders[mc].gen[1],dataloaders[mc].gen[-1])            
            dataloaders[mc].mask = dataloaders[mc].gen[-1]            
            del dataloaders[mc].gen
            
        gc.collect()


def cluster_jets(dataloaders):
    """
    Update dataloaders with clustered jet information.

    Args:
        dataloaders (dict): Dictionary of dataloaders containing particle, event, and mask data.
    """
    import fastjet
    import numpy as np

    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)

    def _convert_kinematics(part, event, mask):
        """Convert particle kinematics to Cartesian coordinates."""
        new_particles = np.zeros((part.shape[0], part.shape[1], 4))
        new_particles[:, :, 0] = np.ma.exp(part[:, :, 2]) * np.cos(np.pi + part[:, :, 1] + event[:, 4, None])
        new_particles[:, :, 1] = np.ma.exp(part[:, :, 2]) * np.sin(np.pi + part[:, :, 1] + event[:, 4, None])
        new_particles[:, :, 2] = np.ma.exp(part[:, :, 2]) * np.sinh(part[:, :, 0] + event[:, 3, None])
        new_particles[:, :, 3] = np.ma.exp(part[:, :, 5])

        return new_particles * mask[:, :, None]
    
    def _convert_electron_kinematics(event_list):
        pt = event_list[:, 2]*np.sqrt(np.exp(event_list[:, 0]))
        phi = event_list[:, 4]
        eta = event_list[:, 3]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2)
        electron_cartesian_dict = {"px":px, "py":py, "pz":pz, "E":E}
        return electron_cartesian_dict
    
    def _calculate_q(final_states, scattered_electron):
        sigma_h = np.array([np.sum(event[:, 3] - event[:, 2]) for event in final_states if any(np.abs(event[:, 0]) != 0)])
        scattered_electron_momentum = np.sqrt(scattered_electron["px"]**2 + scattered_electron["py"]**2 + scattered_electron["pz"]**2)
        scattered_electron_theta = np.arccos(scattered_electron["pz"]/scattered_electron_momentum)
        sigma_eprime = scattered_electron["E"] * (1 - np.cos(scattered_electron_theta))
        sigma_tot = sigma_h + sigma_eprime
        beam_electron_momentum = {"px":np.zeros(len(sigma_tot)), "py":np.zeros(len(sigma_tot)), "pz":-sigma_tot/2., "E":sigma_tot/2.}
        q_x = beam_electron_momentum["px"] - scattered_electron["px"]
        q_y = beam_electron_momentum["py"] - scattered_electron["py"]
        q_z = beam_electron_momentum["pz"] - scattered_electron["pz"]
        q_E = beam_electron_momentum["E"] - scattered_electron["E"]
        q_list = np.stack((q_x, q_y, q_z, q_E), axis=1)
        return q_list
    
    def _calculate_jet_features(jet, q):

        """Calculate jet features such as tau variables and ptD."""
        tau_11, tau_11p5, tau_12, tau_20, sumpt = 0, 0, 0, 0, 0
        for constituent in jet.constituents():
            delta_r = jet.delta_R(constituent)
            pt = constituent.pt()

            tau_11 += pt * delta_r ** 1
            tau_11p5 += pt * delta_r ** 1.5
            tau_12 += pt * delta_r ** 2
            tau_20 += pt ** 2
            sumpt += pt

        jet.tau_11 = np.log(tau_11 / jet.pt()) if jet.pt() > 0 else 0.0
        jet.tau_11p5 = np.log(tau_11p5 / jet.pt()) if jet.pt() > 0 else 0.0
        jet.tau_12 = np.log(tau_12 / jet.pt()) if jet.pt() > 0 else 0.0
        jet.tau_20 = tau_20 / (jet.pt() ** 2) if jet.pt() > 0 else 0.0
        jet.ptD = np.sqrt(tau_20) / sumpt if sumpt > 0 else 0.0

        # Calculating zjet
        P = np.array([0, 0, 920, 920], dtype=np.float32) # 920 GeV is proton beam energy
        P_dot_q = P[3]*q[3] - P[0]*q[0] - P[1]*q[1] - P[2]*q[2]
        z_jet_numerator = P[3]*jet.E() - P[0]*jet.px() - P[1]*jet.py() - P[2]*jet.pz()
        jet.zjet = z_jet_numerator/P_dot_q
        

        
    def _take_leading_jet(jets):
        """Extract features of the leading jet."""
        if not jets:
            return np.zeros(10)

        leading_jet = jets[0]
        return np.array([
            leading_jet.pt(),
            leading_jet.eta(),
            (leading_jet.phi() + np.pi) % (2 * np.pi) - np.pi,
            leading_jet.E(),
            leading_jet.tau_11,
            leading_jet.tau_11p5,
            leading_jet.tau_12,
            leading_jet.tau_20,
            leading_jet.ptD,
            leading_jet.zjet
        ])
    
    def _take_all_jets(jets):
        max_num_jets = 4

        if not jets:
            return np.zeros((max_num_jets, 10))
        jet_array = []
        for i in range(max_num_jets):
            if i < len(jets):
                jet_info = [
                            jet.pt(),
                            jet.eta(),
                            (jet.phi() + np.pi) % (2 * np.pi) - np.pi,
                            jet.E(),
                            jet.tau_11,
                            jet.tau_11p5,
                            jet.tau_12,
                            jet.tau_20,
                            jet.ptD,
                            jet.zjet
                        ]
            else:
                jet_info = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            jet_array.append(jet_info)

        return np.array(jet_array)

    for dataloader_name, data in dataloaders.items():
        #print(f"----------------- Started working with {dataloader_name} -------------------")

        # Convert particles to Cartesian coordinates
        cartesian = _convert_kinematics(data.part, data.event, data.mask)
        electron_momentum = _convert_electron_kinematics(data.event)
        # print(cartesian)
        q = _calculate_q(cartesian, electron_momentum)
        list_of_jets = []
        list_of_all_jets = []
        for i, event in enumerate(cartesian):
            particles = [
                fastjet.PseudoJet(p[0], p[1], p[2], p[3])
                for p in event if np.abs(p[0]) != 0
            ]

            # Cluster jets and sort by pt
            cluster = fastjet.ClusterSequence(particles, jetdef)
            sorted_jets = fastjet.sorted_by_pt(cluster.inclusive_jets(ptmin=10))
            # Calculate jet features
            for jet in sorted_jets:
                _calculate_jet_features(jet, q[i])

            # Take the leading jet's features
            # leading_jet = _take_leading_jet(sorted_jets)
            # list_of_jets.append(leading_jet)
            all_jets = _take_all_jets(sorted_jets)
            list_of_all_jets.append(all_jets)
        # Store the jet features in the dataloader
        #data.jet = np.array(list_of_jets, dtype=np.float32)
        data.all_jets = np.array(list_of_all_jets, dtype=np.float32)
        #print(f"----------------- Done working with {dataloader_name} -------------------")




def plot_particles(flags, dataloaders, data_weights, version, num_part, nbins=10):
    """
    Plot particle-level observables for each feature in the dataset with optional systematic uncertainties.

    Args:
        flags: Object containing configuration flags (e.g., blind, sys, reco).
        dataloaders: Dictionary containing data for different datasets.
        data_weights: Dictionary containing weights for systematic uncertainties.
        version: String to define the output file version.
        num_part: Number of particles to process per event.
        nbins: Number of bins for the histogram (default: 10).

    Returns:
        None. Saves the plots as PDF files.
    """
    import numpy as np
    import utils

    # Determine weight name based on flags
    weight_name = 'closure' if flags.blind else 'Rapgap'

    # Loop over all features in the particle data
    for feature in range(dataloaders['Rapgap'].part.shape[-1]):

        # Compute nominal histogram and systematic uncertainties if enabled
        if flags.sys:
            nominal, binning = np.histogram(
                dataloaders['Rapgap'].part[:, feature],
                bins=nbins,
                density=True,
                weights=(
                    dataloaders['Rapgap'].weight * data_weights['Rapgap']
                ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['Rapgap'].mask]
            )

            nominal_closure, _ = np.histogram(
                dataloaders['Djangoh'].part[:, feature],
                bins=binning,
                density=True,
                weights=(
                    dataloaders['Djangoh'].weight
                ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['Djangoh'].mask]
            )

            total_unc = np.zeros_like(nominal)
            for sys, sys_weights in data_weights.items():
                if sys == 'Rapgap':
                    continue

                sample_name = 'Rapgap' if sys == 'closure' else sys
                sys_var, _ = np.histogram(
                    dataloaders[sample_name].part[:, feature],
                    bins=binning,
                    density=True,
                    weights=(
                        dataloaders[sample_name].weight * sys_weights
                    ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders[sample_name].mask]
                )

                ref_hist = nominal_closure if sys == 'closure' else nominal
                total_unc += (np.ma.divide(sys_var, ref_hist).filled(1) - 1) ** 2

            total_unc = np.sqrt(total_unc)
        else:
            total_unc = None
            binning = None

        # Prepare feed_dict and weights for plotting
        feed_dict = {
            'Rapgap_unfolded': dataloaders['Rapgap'].part[:, feature],
            'Rapgap': dataloaders['Rapgap'].part[:, feature],
            'Djangoh': dataloaders['Djangoh'].part[:, feature],
        }

        weights = {
            'Rapgap_unfolded': (
                dataloaders['Rapgap'].weight * data_weights[weight_name]
            ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Rapgap': (
                dataloaders['Rapgap'].weight
            ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['Rapgap'].mask],
            'Djangoh': (
                dataloaders['Djangoh'].weight
            ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['Djangoh'].mask],
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].part[:, feature]
            weights['data'] = (
                dataloaders['data'].weight
            ).reshape(-1, 1, 1).repeat(num_part, axis=1).reshape(-1)[dataloaders['data'].mask]

        # Plot histogram using utils.HistRoutine
        fig, ax = utils.HistRoutine(
            feed_dict,
            xlabel=utils.particle_names.get(str(feature), f"Feature {feature}"),
            weights=weights,
            reference_name='data' if flags.reco else 'Rapgap_unfolded',
            label_loc='upper left',
            uncertainty=total_unc,
            binning=binning,
        )

        # Save the plot
        fig.savefig(f'../plots/{version}_part_{feature}.pdf')

        
def plot_qtQ(flags,dataloaders,data_weights,version):
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

def plot_deltaphi(flags, dataloaders, data_weights, version):
    import numpy as np
    import utils

    def get_deltaphi(jet, elec):
        delta_phi = np.abs(np.pi + jet[:, 2] - elec[:, 4])
        delta_phi[delta_phi > 2 * np.pi] -= 2 * np.pi
        return delta_phi

    def compute_histogram(dataset_name, weights=None):
        valid_indices = dataloaders[dataset_name].jet[:, 0] > 0
        data = get_deltaphi(dataloaders[dataset_name].jet, dataloaders[dataset_name].event)[valid_indices]
        if weights is not None:
            weights = weights[valid_indices]
        return np.histogram(data, bins=binning, density=True, weights=weights)

    # Determine weight name
    weight_name = 'closure' if flags.blind else 'Rapgap'
    data_name = 'Rapgap_closure' if flags.blind else 'Rapgap_unfolded'

    # Set binning
    binning = np.linspace(0, 1, 8)

    # Compute nominal and closure histograms if systematic uncertainties are enabled
    total_unc = None
    if flags.sys:
        nominal, _ = compute_histogram('Rapgap', dataloaders['Rapgap'].weight * data_weights['Rapgap'])
        nominal_closure, _ = compute_histogram('Djangoh', dataloaders['Djangoh'].weight)

        total_unc = np.zeros_like(nominal)
        for sys, sys_weights in data_weights.items():
            if sys == 'Rapgap':
                continue

            sample_name = 'Rapgap' if sys == 'closure' else sys
            sys_hist, _ = compute_histogram(sample_name, dataloaders[sample_name].weight * sys_weights)

            ref_hist = nominal_closure if sys == 'closure' else nominal
            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            total_unc += unc

            print(f"{sys}: max uncertainty = {np.max(np.sqrt(unc))}")

        total_unc = np.sqrt(total_unc)

    # Prepare weights and data for plotting
    weights = {
        data_name: (dataloaders['Rapgap'].weight * data_weights[weight_name])[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet[:, 0] > 0],
    }

    feed_dict = {
        data_name: get_deltaphi(dataloaders['Rapgap'].jet, dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Rapgap': get_deltaphi(dataloaders['Rapgap'].jet, dataloaders['Rapgap'].event)[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Djangoh': get_deltaphi(dataloaders['Djangoh'].jet, dataloaders['Djangoh'].event)[dataloaders['Djangoh'].jet[:, 0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = get_deltaphi(dataloaders['data'].jet, dataloaders['data'].event)[dataloaders['data'].jet[:, 0] > 0]

    # Generate histogram plot
    fig, ax = utils.HistRoutine(
        feed_dict,
        xlabel=r"$\Delta\phi^{jet}$ [rad]",
        weights=weights,
        logy=True,
        logx=True,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
    )

    # Set plot limits and save
    ax.set_ylim(1e-2, 50)
    fig.savefig(f'../plots/{version}_jet_deltaphi.pdf')
    

def plot_jet_pt(flags, dataloaders, data_weights, version,lab_frame=True):
    import numpy as np
    import utils

    
    def compute_histogram(dataset_name, weights=None,lab_frame=True):
        if lab_frame:
            valid_indices = dataloaders[dataset_name].jet[:, 0] > 0
            data = dataloaders[dataset_name].jet[:,0][valid_indices]
        else:
            valid_indices = dataloaders[dataset_name].jet_breit[:, 0] > 0
            data = dataloaders[dataset_name].jet_breit[:,0][valid_indices]
        if weights is not None:
            weights = weights[valid_indices]
        return np.histogram(data, bins=binning, density=True, weights=weights)

    # Determine weight name
    weight_name = 'closure' if flags.blind else 'Rapgap'
    data_name = 'Rapgap_closure' if flags.blind else 'Rapgap_unfolded'

    # Set binning
    if lab_frame:
        binning = np.logspace(np.log10(10),np.log10(100),7)
    else:
        binning = np.logspace(np.log10(10),np.log10(50),7)

    # Compute nominal and closure histograms if systematic uncertainties are enabled
    total_unc = None
    if flags.sys:
        nominal, _ = compute_histogram('Rapgap', dataloaders['Rapgap'].weight * data_weights['Rapgap'],lab_frame)
        nominal_closure, _ = compute_histogram('Djangoh', dataloaders['Djangoh'].weight,lab_frame)

        total_unc = np.zeros_like(nominal)
        for sys, sys_weights in data_weights.items():
            if sys == 'Rapgap':
                continue

            sample_name = 'Rapgap' if sys == 'closure' else sys
            sys_hist, _ = compute_histogram(sample_name, dataloaders[sample_name].weight * sys_weights,lab_frame)

            ref_hist = nominal_closure if sys == 'closure' else nominal
            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            total_unc += unc

            print(f"{sys}: max uncertainty = {np.max(np.sqrt(unc))}")

        total_unc = np.sqrt(total_unc)

    # Prepare weights and data for plotting

    if lab_frame:
        weights = {
            data_name: (dataloaders['Rapgap'].weight * data_weights[weight_name])[dataloaders['Rapgap'].jet[:, 0] > 0],
            'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet[:, 0] > 0],
            'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet[:, 0] > 0],
        }

        feed_dict = {
            data_name: dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:, 0] > 0],
            'Rapgap': dataloaders['Rapgap'].jet[:,0][dataloaders['Rapgap'].jet[:, 0] > 0],
            'Djangoh': dataloaders['Djangoh'].jet[:,0][dataloaders['Djangoh'].jet[:, 0] > 0],
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].jet[:,0][dataloaders['data'].jet[:, 0] > 0]

    else:
        weights = {
            data_name: (dataloaders['Rapgap'].weight * data_weights[weight_name])[dataloaders['Rapgap'].jet_breit[:, 0] > 0],
            'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet_breit[:, 0] > 0],
            'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet_breit[:, 0] > 0],
        }

        feed_dict = {
            data_name: dataloaders['Rapgap'].jet_breit[:,0][dataloaders['Rapgap'].jet_breit[:, 0] > 0],
            'Rapgap': dataloaders['Rapgap'].jet_breit[:,0][dataloaders['Rapgap'].jet_breit[:, 0] > 0],
            'Djangoh': dataloaders['Djangoh'].jet_breit[:,0][dataloaders['Djangoh'].jet_breit[:, 0] > 0],
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].jet_breit[:,0][dataloaders['data'].jet_breit[:, 0] > 0]


    # Generate histogram plot
    fig, ax = utils.HistRoutine(
        feed_dict,
        xlabel=r'Jet $p_\mathrm{T}$ [GeV]' if lab_frame else r'Breit frame Jet $p_\mathrm{T}$ [GeV]',
        weights=weights,
        logy=True,
        logx=True,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
    )

    # Set plot limits and save
    #ax.set_ylim(1e-2, 50)
    tag = 'lab' if lab_frame else 'breit'
    
    fig.savefig(f'../plots/{version}_jet_pt_{tag}.pdf')

    


def plot_tau(flags, dataloaders, data_weights, version):
    import numpy as np
    import utils


    def compute_histogram(dataset_name, weights=None):
        valid_indices = dataloaders[dataset_name].jet[:, 0] > 0
        data = dataloaders[dataset_name].jet[:, 4][valid_indices]
        if weights is not None:
            weights = weights[valid_indices]
        return np.histogram(data, bins=binning, density=True, weights=weights)

    # Determine weight name
    weight_name = 'closure' if flags.blind else 'Rapgap'
    data_name = 'Rapgap_closure' if flags.blind else 'Rapgap_unfolded'

    # Set binning
    binning = np.array([-4.00,-3.15,-2.59,-2.18,-1.86,-1.58,-1.29,-1.05,-0.81,-0.61,0.00])

    # Compute nominal and closure histograms if systematic uncertainties are enabled
    total_unc = None
    if flags.sys:
        nominal, _ = compute_histogram('Rapgap', dataloaders['Rapgap'].weight * data_weights['Rapgap'])
        nominal_closure, _ = compute_histogram('Djangoh', dataloaders['Djangoh'].weight)

        total_unc = np.zeros_like(nominal)
        for sys, sys_weights in data_weights.items():
            if sys == 'Rapgap':
                continue

            sample_name = 'Rapgap' if sys == 'closure' else sys
            sys_hist, _ = compute_histogram(sample_name, dataloaders[sample_name].weight * sys_weights)

            ref_hist = nominal_closure if sys == 'closure' else nominal
            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            total_unc += unc

            print(f"{sys}: max uncertainty = {np.max(np.sqrt(unc))}")

        total_unc = np.sqrt(total_unc)

    # Prepare weights and data for plotting
    weights = {
        data_name: (dataloaders['Rapgap'].weight * data_weights[weight_name])[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Rapgap': dataloaders['Rapgap'].weight[dataloaders['Rapgap'].jet[:, 0] > 0],
        'Djangoh': dataloaders['Djangoh'].weight[dataloaders['Djangoh'].jet[:, 0] > 0],
    }

    feed_dict = {
        data_name: dataloaders['Rapgap'].jet[:, 4][dataloaders['Rapgap'].jet[:, 0] > 0],
        'Rapgap': dataloaders['Rapgap'].jet[:, 4][dataloaders['Rapgap'].jet[:, 0] > 0],
        'Djangoh': dataloaders['Djangoh'].jet[:, 4][dataloaders['Djangoh'].jet[:, 0] > 0],
    }

    if flags.reco:
        feed_dict['data'] = dataloaders['data'].jet[:, 4][dataloaders['data'].jet[:, 0] > 0]

    # Generate histogram plot
    fig, ax = utils.HistRoutine(
        feed_dict,
        xlabel=r'$\mathrm{ln}(\lambda_1^1)$',
        weights=weights,
        logy=False,
        logx=False,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
    )

    # Set plot limits and save
    ax.set_ylim(0, 1.2)
    fig.savefig(f'../plots/{version}_jet_tau10.pdf')

def plot_zjet(flags, dataloaders, data_weights, version, frame = "lab"):
    import numpy as np
    import utils

    def compute_histogram(dataset_name, weights=None, frame="lab"):
        if frame == "breit":
            mask = dataloaders[dataset_name].all_jets[:, :, 9]>0
            data = ak.drop_none(ak.mask(dataloaders[dataset_name].all_jets[:, :, 9], mask))
            num_jets_per_event = ak.count(data, axis=1)
            data = ak.flatten(data)
        else:
            mask = dataloaders[dataset_name].all_jets_breit[:, :, 7]>0
            data = ak.drop_none(ak.mask(dataloaders[dataset_name].all_jets_breit[:, :, 7], mask))
            num_jets_per_event = ak.count(data, axis=1)
            data = ak.flatten(data)
        if weights is not None:
            weights = np.repeat(weights, num_jets_per_event, axis=0)
        counts, bins = np.histogram(data, bins=binning, density=True, weights=weights)
        return ak.to_numpy(counts), bins

    # Determine weight name
    weight_name = 'closure' if flags.blind else 'Rapgap'
    data_name = 'Rapgap_closure' if flags.blind else 'Rapgap_unfolded'

    # Set binning
    binning = np.linspace(0.2, 1, 10)
    # Compute nominal and closure histograms if systematic uncertainties are enabled
    total_unc = None
    if flags.sys:
        nominal, _ = compute_histogram('Rapgap', dataloaders['Rapgap'].weight * data_weights['Rapgap'], frame)
        nominal_closure, _ = compute_histogram('Djangoh', dataloaders['Djangoh'].weight, frame)

        total_unc = np.zeros_like(nominal)
        for sys, sys_weights in data_weights.items():
            if sys == 'Rapgap':
                continue

            sample_name = 'Rapgap' if sys == 'closure' else sys
            sys_hist, _ = compute_histogram(sample_name, dataloaders[sample_name].weight * sys_weights, frame)

            ref_hist = nominal_closure if sys == 'closure' else nominal
            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            print(unc, total_unc)
            total_unc += unc

            print(f"{sys}: max uncertainty = {np.max(np.sqrt(unc))}")

        total_unc = np.sqrt(total_unc)

    # Prepare weights and data for plotting

    if frame == "breit":
        Rapgap_mask = dataloaders["Rapgap"].all_jets_breit[:, :, 7]>0
        Rapgap_data = ak.drop_none(ak.mask(dataloaders["Rapgap"].all_jets_breit[:, :, 7], Rapgap_mask))
        num_Rapgap_jets_per_event = ak.count(Rapgap_data, axis=1)
        Rapgap_data = ak.flatten(Rapgap_data)

        Djangoh_mask = dataloaders["Djangoh"].all_jets_breit[:, :, 7]>0
        Djangoh_data = ak.drop_none(ak.mask(dataloaders["Djangoh"].all_jets_breit[:, :, 7], Djangoh_mask))
        num_Djangoh_jets_per_event = ak.count(Djangoh_data, axis=1)
        Djangoh_data = ak.flatten(Djangoh_data)

        weights = {
            data_name: np.repeat((dataloaders['Rapgap'].weight * data_weights[weight_name]), num_Rapgap_jets_per_event, axis=0),
            'Rapgap': np.repeat(dataloaders['Rapgap'].weight, num_Rapgap_jets_per_event, axis=0),
            'Djangoh': np.repeat(dataloaders['Djangoh'].weight, num_Djangoh_jets_per_event, axis=0),
        }

        feed_dict = {
            data_name: Rapgap_data,
            'Rapgap': Rapgap_data,
            'Djangoh': Djangoh_data,
        }

        if flags.reco:
            data_mask = dataloaders["data"].all_jets_breit[:, :, 7]>0
            data = ak.drop_none(ak.mask(dataloaders["data"].all_jets_breit[:, :, 7], data_mask))
            data = ak.flatten(data)
            feed_dict['data'] = data
    else:
        Rapgap_mask = dataloaders["Rapgap"].all_jets[:, :, 9]>0
        Rapgap_data = ak.drop_none(ak.mask(dataloaders["Rapgap"].all_jets[:, :, 9], Rapgap_mask))
        num_Rapgap_jets_per_event = ak.count(Rapgap_data, axis=1)
        Rapgap_data = ak.flatten(Rapgap_data)

        Djangoh_mask = dataloaders["Djangoh"].all_jets[:, :, 9]>0
        Djangoh_data = ak.drop_none(ak.mask(dataloaders["Djangoh"].all_jets[:, :, 9], Djangoh_mask))
        num_Djangoh_jets_per_event = ak.count(Djangoh_data, axis=1)
        Djangoh_data = ak.flatten(Djangoh_data)

        weights = {
            data_name: np.repeat((dataloaders['Rapgap'].weight * data_weights[weight_name]), num_Rapgap_jets_per_event, axis=0),
            'Rapgap': np.repeat(dataloaders['Rapgap'].weight, num_Rapgap_jets_per_event, axis=0),
            'Djangoh': np.repeat(dataloaders['Djangoh'].weight, num_Djangoh_jets_per_event, axis=0),
        }

        feed_dict = {
            data_name: Rapgap_data,
            'Rapgap': Rapgap_data,
            'Djangoh': Djangoh_data,
        }

        if flags.reco:
            data_mask = dataloaders["data"].all_jets[:, :, 7]>0
            data = ak.drop_none(ak.mask(dataloaders["data"].all_jets[:, :, 7], data_mask))
            data = ak.flatten(data)
            feed_dict['data'] = data

    # Generate histogram plot
    fig, ax = utils.HistRoutine(
        feed_dict,
        xlabel=f'$z_{{jet}}$ ({frame.capitalize()} kt)',
        weights=weights,
        logy=False,
        logx=False,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
    )
    # Set plot limits and save
    ax.set_ylim(0, 5)
    fig.savefig(f'../plots/{version}_zjet_{frame}.pdf')

def cluster_breit(dataloaders):
    import fastjet
    import awkward as ak
    import vector
    import itertools
    
    def _convert_kinematics(part, event, mask):
        """Convert particle kinematics to Cartesian coordinates."""
        new_particles = np.zeros((part.shape[0], part.shape[1], 4))
        new_particles[:, :, 0] = np.ma.exp(part[:, :, 2]) * np.cos(np.pi + part[:, :, 1] + event[:, 4, None])
        new_particles[:, :, 1] = np.ma.exp(part[:, :, 2]) * np.sin(np.pi + part[:, :, 1] + event[:, 4, None])
        new_particles[:, :, 2] = np.ma.exp(part[:, :, 2]) * np.sinh(part[:, :, 0] + event[:, 3, None])
        new_particles[:, :, 3] = np.ma.exp(part[:, :, 5])

        return new_particles * mask[:, :, None]

    def _convert_electron_kinematics(event_list):
        pt = event_list[:, 2]*np.sqrt(np.exp(event_list[:, 0]))
        phi = event_list[:, 4]
        eta = event_list[:, 3]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2)
        electron_cartesian_dict = {"px":px, "py":py, "pz":pz, "E":E}
        return electron_cartesian_dict

    def boost_particles(final_states, scattered_electron):
        particle_vectors = []
        # Putting all particles into vectors and calculating sigma_tot
        for event in final_states:
            nonzero_particles = np.array([part for part in event if np.abs(part[0])!=0])
            particle_vectors.append([vector.obj(px=part[0], py=part[1], pz=part[2], energy=part[3]) for part in nonzero_particles])
    
        sigma_h = np.array([np.sum(event[:, 3] - event[:, 2]) for event in final_states if any(np.abs(event[:, 0]) != 0)])

        scattered_electron_momentum = np.sqrt(scattered_electron["px"]**2 + scattered_electron["py"]**2 + scattered_electron["pz"]**2)
        scattered_electron_theta = np.arccos(scattered_electron["pz"]/scattered_electron_momentum)
        sigma_eprime = scattered_electron["E"] * (1 - np.cos(scattered_electron_theta))
        sigma_tot = sigma_h + sigma_eprime

        beam_electron_momentum = {"px":np.zeros(len(sigma_tot)), "py":np.zeros(len(sigma_tot)), "pz":-sigma_tot/2., "E":sigma_tot/2.}
        q_x = beam_electron_momentum["px"] - scattered_electron["px"]
        q_y = beam_electron_momentum["py"] - scattered_electron["py"]
        q_z = beam_electron_momentum["pz"] - scattered_electron["pz"]
        q_E = beam_electron_momentum["E"] - scattered_electron["E"]
        q_list = np.stack((q_x, q_y, q_z, q_E), axis=1)
        # Boosting the particles to the Breit frame
        boosted_vectors = []
        z_hat = vector.obj(px=0, py=0, pz=1, energy=1)

        for q, event_particles in zip(q_list, particle_vectors):
            # Calculating boost vector
            # Calculation taken from https://doi.org/10.1140/epjc/s10052-024-13003-1
            q = vector.obj(px=q[0], py=q[1], pz=q[2], energy=q[3])
            Q = np.sqrt(-1*q.dot(q))
            Sigma = q.energy - q.pz
            boostvector = q - Q*Q/Sigma*(-1*z_hat)
            xBoost = vector.obj(px = q.px/q.pt, py = q.py/q.pt, pz = q.pt/Sigma, energy = q.pt/Sigma)
            yBoost = vector.obj(px = -q.py/q.pt, py = q.px/q.pt, pz = 0, energy = 0)
            boosted_event_vectors = [
                vector.obj(
                    px = xBoost.dot(vec),
                    py = yBoost.dot(vec),
                    pz = q.dot(vec)/Q,
                    energy = boostvector.dot(vec)/Q
                )
                for vec in event_particles
            ]

            boosted_vectors.append(boosted_event_vectors)
        return boosted_vectors

    
    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, 1.0)
    for dataloader_name, data in dataloaders.items():
        
        electron_momentum = _convert_electron_kinematics(data.event)
        cartesian = _convert_kinematics(data.part, data.event, data.mask)        
        boosted_vectors = boost_particles(cartesian, electron_momentum)
        events = []

        for event in boosted_vectors:
            events.append([{"px": part_vec.px, "py": part_vec.py, "pz": part_vec.pz, "E": part_vec.E} for part_vec in event])
        
        array = ak.Array(events)
        cluster = fastjet.ClusterSequence(array, jetdef)
        jets = cluster.inclusive_jets(min_pt=5)
        
        jets["pt"] = -np.sqrt(jets["px"]**2 + jets["py"]**2)
        jets["phi"] = np.arctan2(jets["py"],jets["px"])
        jets["eta"] = np.arcsinh(jets["pz"]/jets["pt"])
        jets = fastjet.sorted_by_pt(jets)


        def calculate_eec(jets, event):

            # inelasticity 
            y = event[:, 1]

            scattered_electron_momentum = np.sqrt(scattered_electron["px"]**2 + scattered_electron["py"]**2 + scattered_electron["pz"]**2)
            scattered_electron_theta = np.arccos(scattered_electron["pz"]/scattered_electron_momentum)

            # calculate Bjorken x (invariant wrt frames) using the ISigma method from table 1 in https://arxiv.org/pdf/2110.05505
            x_B = scattered_electron["E"] * np.divide( 1 + np.cos(scattered_electron_theta), 2*y*920 ) # also need to divide by E_proton

            # proton four-momentum in Breit frame
            Q = np.sqrt(np.exp(event[:,0]))
            P = np.divide(Q, 2*x_B) * np.array([1, 0, 0, 1], dtype=np.float32)

            # Proton polar angle
            theta_P = np.arccos(P[3] / np.linalg.norm(P))

            # normalization for EEC in DIS
            px_sum = np.sum(jet.constituents().px())
            py_sum = np.sum(jet.constituents().py())
            pz_sum = np.sum(jet.constituents().pz())
            E_sum = np.sum(jet.constituents().E())
            # z, delta_theta = [], []
            entry = []

            P_dot_psum = P[3]*E_sum - P[0]*px_sum - P[1]*py_sum - P[2]*pz_sum

            for constituent in jet.constituents():

                P_dot_pc = P[3]*constituent.E() - P[0]*constituent.px() - P[1]*constituent.py() - P[2]*constituent.pz()
                z = P_dot_pc / P_dot_psum

                pt = constituent.pt()
                eta = constituent.eta()
                phi = constituent.phi()
                theta_c = 2 * np.arctan(np.exp(-eta))
                
                entry.append( (theta_P - theta_c)/z )

        # dataloaders[dataloader_name].all_jets_breit = calculate_eec(jets, dataloaders[dataloader_name].event)
        

        def _take_leading_jet(jets):
            jet = np.zeros((data.event.shape[0],4))
            jet[:,0] = -np.array(list(itertools.zip_longest(*jets.pt.to_list(), fillvalue=0))).T[:,0]
            jet[:,1] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T[:,0]
            jet[:,2] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T[:,0]
            jet[:,3] = np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T[:,0]
            return jet


        def _take_all_jets(jets,maxjets = 5):
            
            jet = np.zeros((data.event.shape[0],maxjets,7))
            jet[:,:,0] = -np.array(list(itertools.zip_longest(*jets.pt.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,1] = np.array(list(itertools.zip_longest(*jets.eta.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,2] = np.array(list(itertools.zip_longest(*jets.phi.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,3] = np.array(list(itertools.zip_longest(*jets.E.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,4] = np.array(list(itertools.zip_longest(*jets.px.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,5] = np.array(list(itertools.zip_longest(*jets.py.to_list(), fillvalue=0))).T[:,:maxjets]
            jet[:,:,6] = np.array(list(itertools.zip_longest(*jets.pz.to_list(), fillvalue=0))).T[:,:maxjets]
            return jet
        
        # def _take_all_jets(jets, max_num_jets):
        #     all_jets = []
        #     for event in jets:
        #         event_jets = []
        #         for i in range(max_num_jets):
        #             if i < len(event):
        #                 jet_info = [-event[i].pt,
        #                             event[i].eta,
        #                             event[i].phi,
        #                             event[i].E,
        #                             event[i].px,
        #                             event[i].py,
        #                             event[i].pz
        #                         ]
        #             else:
        #                 jet_info = [0, 0, 0, 0, 0, 0, 0]
        #             event_jets.append(jet_info)
        #         all_jets.append(event_jets)
        #     return np.array(all_jets)
            
        #dataloaders[dataloader_name].jet_breit = _take_leading_jet(jets)

        max_num_jets = 4
        dataloaders[dataloader_name].all_jets_breit = _take_all_jets(jets, max_num_jets)

        def calculate_zjet(jet_data, event):
            Q_array = np.sqrt(np.exp(event[:,0]))

            n = np.array([0, 0, 1, 1], dtype=np.float32)
            z_jet = []

            jet_px = jet_data[:, :, 4]
            jet_py = jet_data[:, :, 5]
            jet_pz = jet_data[:, :, 6]
            jet_E = jet_data[:, :, 3]
            mask = (jet_px!=0) & (jet_py!=0) & (jet_pz!=0) & (jet_E!=0)
            jet_px = ak.mask(jet_px, mask)
            jet_py = ak.mask(jet_py, mask)
            jet_pz = ak.mask(jet_pz, mask)
            jet_E = ak.mask(jet_E, mask)
            numerator = n[3]*jet_E- n[0]*jet_px - n[1]*jet_py - n[2]*jet_pz
            counts = ak.num(numerator)
            Q_array = np.repeat(Q_array, counts)
            z_jet = ak.flatten(numerator)/Q_array
            z_jet = np.array(ak.fill_none(ak.unflatten(z_jet, counts), 0))
            z_jet = z_jet.reshape(z_jet.shape[0], z_jet.shape[1], 1)
            jet_data = np.concatenate((jet_data, z_jet), axis=2)
            return jet_data
        dataloaders[dataloader_name].all_jets_breit = calculate_zjet(dataloaders[dataloader_name].all_jets_breit, dataloaders[dataloader_name].event)
    
    
def plot_event(flags, dataloaders, data_weights, version, nbins=10):
    """
    Plot event-level observables for each feature in the dataset with optional systematic uncertainties.

    Args:
        flags: Object containing configuration flags (e.g., blind, sys, reco).
        dataloaders: Dictionary containing data for different datasets.
        data_weights: Dictionary containing weights for systematic uncertainties.
        version: String to define the output file version.
        nbins: Number of bins for the histogram (default: 10).

    Returns:
        None. Saves the plots as PDF files.
    """
    import numpy as np
    import utils

    # Determine weight name based on flags
    weight_name = 'closure' if flags.blind else 'Rapgap'

    # Loop over all features in the event data
    for feature in range(dataloaders['Rapgap'].event.shape[-1]):

        # Compute nominal histogram and systematic uncertainties if enabled
        if flags.sys:
            nominal, binning = np.histogram(
                dataloaders['Rapgap'].event[:, feature],
                bins=nbins,
                density=True,
                weights=data_weights['Rapgap']
            )

            nominal_closure, _ = np.histogram(
                dataloaders['Djangoh'].event[:, feature],
                bins=binning,
                density=True,
                weights=data_weights['Djangoh']
            )

            total_unc = np.zeros_like(nominal)
            for sys, sys_weights in data_weights.items():
                if sys == 'Rapgap':
                    continue

                sample_name = 'Rapgap' if sys == 'closure' else sys
                sys_var, _ = np.histogram(
                    dataloaders[sample_name].event[:, feature],
                    bins=binning,
                    density=True,
                    weights=sys_weights
                )

                ref_hist = nominal_closure if sys == 'closure' else nominal
                total_unc += (np.ma.divide(sys_var, ref_hist).filled(1) - 1) ** 2

            total_unc = np.sqrt(total_unc)
        else:
            total_unc = None
            binning = None

        # Prepare feed_dict and weights for plotting
        feed_dict = {
            'Rapgap_unfolded': dataloaders['Rapgap'].event[:, feature],
            'Rapgap': dataloaders['Rapgap'].event[:, feature],
            'Djangoh': dataloaders['Djangoh'].event[:, feature],
        }
        weights = {
            'Rapgap_unfolded': dataloaders['Rapgap'].weight * data_weights[weight_name],
            'Rapgap': dataloaders['Rapgap'].weight,
            'Djangoh': dataloaders['Djangoh'].weight,
        }

        if flags.reco:
            feed_dict['data'] = dataloaders['data'].event[:, feature]
            weights['data'] = dataloaders['data'].weight

        # Plot histogram using utils.HistRoutine
        fig, ax = utils.HistRoutine(
            feed_dict,
            xlabel=utils.event_names.get(str(feature), f"Feature {feature}"),
            weights=weights,
            reference_name='data' if flags.reco else 'Rapgap_unfolded',
            label_loc='upper left',
            uncertainty=total_unc,
            binning=binning,
        )

        # Save the plot
        fig.savefig(f'../plots/{version}_event_{feature}.pdf')        

def plot_observable(flags, var, dataloaders, version):
    info = utils.ObservableInfo(var)

    def compute_histogram(dataset_name, weights=None,density=True):
        if len(dataloaders[dataset_name][var].shape) > 1:
            multiple_jets_per_event = True
            valid_indices = dataloaders[dataset_name]['jet_pt']>0
            data = ak.mask(dataloaders[dataset_name][var], valid_indices)
            data = ak.drop_none(data)
            num_jets_per_event = ak.count(data, axis = 1)
            data = ak.flatten(data)
        else:
            multiple_jets_per_event = False
            valid_indices = dataloaders[dataset_name]['jet_pt'] > 0
            data = dataloaders[dataset_name][var][valid_indices]
        if weights is not None:
            if multiple_jets_per_event:
                weights = np.repeat(weights, num_jets_per_event, axis=0)
            else:
                weights = weights[valid_indices]
        counts, bins = np.histogram(data, bins=binning, density=density, weights=weights)
        return ak.to_numpy(counts), bins

    # Determine weight name
    weight_name = 'closure_weights' if flags.blind else 'weights'
    data_name = 'Rapgap_closure' if flags.blind else 'Rapgap_unfolded'

    # Set binning
    binning = info.binning

    # Compute nominal and closure histograms if systematic uncertainties are enabled
    total_unc = None
    if flags.sys:
        nominal_weights = dataloaders['Rapgap']['weights'] if not flags.reco else np.ones_like(dataloaders['Rapgap']['weights'])

        nominal, _ = compute_histogram('Rapgap', nominal_weights * dataloaders['Rapgap']['mc_weights'])
        nominal_closure, _ = compute_histogram('Djangoh', dataloaders['Djangoh']['mc_weights'])

        total_unc = np.zeros_like(nominal)
        for sys in dataloaders:
            if 'boot' in sys: continue
            print(sys)
            if flags.reco:
                #Skip model and closure uncertainties at reco level
                if sys == 'Rapgap': continue
                if sys == 'Djangoh': continue
                if sys == 'data': continue

                
            sys_weights = dataloaders[sys]['closure_weights'] if sys == 'Rapgap' else dataloaders[sys]['weights']
            if flags.reco: sys_weights = np.ones_like(sys_weights)
            sys_hist, _ = compute_histogram(sys, dataloaders[sys]['mc_weights'] * sys_weights)

            ref_hist = nominal_closure if sys == 'Rapgap' else nominal
            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            total_unc += unc
            
            print(f"{sys}: max uncertainty = {np.max(np.sqrt(unc))}")
            
        #Statistical Uncertainties
        if flags.reco:
            counts,_ = compute_histogram('data',density=False)
            unc = 1.0/(1e-9+counts)
            total_unc += unc
            print(f"stat: max uncertainty = {np.max(np.sqrt(unc))}")
        else:
            if flags.bootstrap:
                print("Running over boostrap entries")
                stat_unc = []
                for boot in range(1,flags.nboot):
                    if boot ==10: continue
                    if boot == 23: continue
                    sys_weights = dataloaders['bootstrap'][f'weights{boot}']
                    sys_hist, _ = compute_histogram('bootstrap', dataloaders['bootstrap']['mc_weights'] * sys_weights)
                    stat_unc.append(sys_hist)
                stat_unc = np.ma.divide(np.std(stat_unc,0), np.mean(stat_unc,0)).filled(0)
                
                total_unc += stat_unc**2
                print(f"{sys}: max uncertainty = {np.max(stat_unc)}")
                    
        total_unc = np.sqrt(total_unc)

    # Prepare weights and data for plotting
    weights = {}
    feed_dict = {}

    if len(dataloaders['Rapgap'][var].shape) > 1:
        Rapgap_mask = dataloaders["Rapgap"]["jet_pt"]>0
        Rapgap_data = ak.drop_none(ak.mask(dataloaders["Rapgap"][var], Rapgap_mask))
        num_Rapgap_jets_per_event = ak.count(Rapgap_data, axis=1)
        Rapgap_data = ak.flatten(Rapgap_data)

        weights[data_name] = np.repeat(dataloaders['Rapgap']['mc_weights'] * dataloaders['Rapgap'][weight_name], num_Rapgap_jets_per_event, axis=0)
        weights['Rapgap'] = np.repeat(dataloaders['Rapgap']['mc_weights'], num_Rapgap_jets_per_event, axis=0)
        feed_dict[data_name] = Rapgap_data
        feed_dict['Rapgap'] = Rapgap_data
    else:
        weights[data_name] = (dataloaders['Rapgap']['mc_weights'] * dataloaders['Rapgap'][weight_name])[dataloaders['Rapgap']['jet_pt'] > 0]
        weights['Rapgap'] = dataloaders['Rapgap']['mc_weights'][dataloaders['Rapgap']['jet_pt'] > 0]
        feed_dict[data_name] = dataloaders['Rapgap'][var][dataloaders['Rapgap']['jet_pt'] > 0]
        feed_dict['Rapgap'] = dataloaders['Rapgap'][var][dataloaders['Rapgap']['jet_pt'] > 0]
    
    if len(dataloaders['Djangoh'][var].shape) > 1:
        Djangoh_mask = dataloaders["Djangoh"]["jet_pt"]>0
        Djangoh_data = ak.drop_none(ak.mask(dataloaders["Djangoh"][var], Djangoh_mask))
        num_Djangoh_jets_per_event = ak.count(Djangoh_data, axis=1)
        Djangoh_data = ak.flatten(Djangoh_data)

        weights['Djangoh'] = np.repeat(dataloaders['Djangoh']['mc_weights'], num_Djangoh_jets_per_event, axis=0)
        feed_dict['Djangoh'] = Djangoh_data
    else:
        weights['Djangoh'] = dataloaders['Djangoh']['mc_weights'][dataloaders['Djangoh']['jet_pt'] > 0]
        feed_dict['Djangoh'] = dataloaders['Djangoh'][var][dataloaders['Djangoh']['jet_pt'] > 0]

    if flags.reco:
        if len(dataloaders['data'][var].shape) > 1:
            data_mask = dataloaders["data"]["jet_pt"]>0
            data = ak.drop_none(ak.mask(dataloaders["data"][var], data_mask))
            data = ak.flatten(data)
            weights['data'] = np.ones_like(data)

            feed_dict['data'] = data
        else:

            weights['data'] = np.ones_like(dataloaders['data'][var][dataloaders['data']['jet_pt'] > 0])
            feed_dict['data'] = dataloaders['data'][var][dataloaders['data']['jet_pt'] > 0]

    if flags.reco:
        ylabel = r'1/N $\mathrm{dN}/\mathrm{d}$%s'%info.name
    else:
        ylabel = r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s'%info.name
            
    # Generate histogram plot
    fig, ax = utils.HistRoutine(
        feed_dict,
        xlabel=info.name,
        ylabel = ylabel,
        weights=weights,
        logy=info.logy,
        logx=info.logx,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
    )

    # Set plot limits and save
    ax.set_ylim(info.ylow, info.yhigh)
    fig.savefig(f'../plots/{version}_{var}.pdf')


        

def gather_data(dataloaders):

    for dataloader in dataloaders:
        #dataloaders[dataloader].mask = np.reshape(dataloaders[dataloader].mask,(-1))
        # dataloaders[dataloader].part = hvd.allgather(tf.constant(dataloaders[dataloader].part.reshape(
        #     (-1,dataloaders[dataloader].part.shape[-1]))[dataloaders[dataloader].mask])).numpy()
        
        dataloaders[dataloader].event = hvd.allgather(tf.constant(dataloaders[dataloader].event)).numpy()
        # dataloaders[dataloader].jet = hvd.allgather(tf.constant(dataloaders[dataloader].jet)).numpy()
        # dataloaders[dataloader].jet_breit = hvd.allgather(tf.constant(dataloaders[dataloader].jet_breit)).numpy()
        dataloaders[dataloader].all_jets = hvd.allgather(tf.constant(dataloaders[dataloader].all_jets)).numpy()
        dataloaders[dataloader].all_jets_breit = hvd.allgather(tf.constant(dataloaders[dataloader].all_jets_breit)).numpy()
        dataloaders[dataloader].weight = hvd.allgather(tf.constant(dataloaders[dataloader].weight)).numpy()
        #dataloaders[dataloader].mask = hvd.allgather(tf.constant(dataloaders[dataloader].mask)).numpy()

        
        
