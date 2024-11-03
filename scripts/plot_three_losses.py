import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import sys, gc
import utils
import matplotlib.pyplot as plt
import pickle
import utils

utils.SetStyle()


def load_pickle(f):
    with open(f, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict


parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option('--weights', default='../weights', help='Folder to store trained weights')
parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
parser.add_option('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_option('--iter', default=4, help='Iteration to load')
parser.add_option('--step', default=2, help='Step to load')
(flags, args) = parser.parse_args()

opt=utils.LoadJson(flags.config)
version = opt['NAME']

N_ensemble = 5
N_iteration = 5

for iter in range(N_iteration):

    axes = []
    fig, gs = utils.SetGrid(ratio=False) 
    ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1 = None
    axes.append(ax0)
    axes.append(ax1)
    axes.append(fig)

    for e in range(N_ensemble):
        print(f"Plotting Ensemble {e+1}/{N_ensemble}")

        #BASELINE  = From Scratch. No pretrained loading, no finetune
        version = 'H1_Nov_ModelLists_1lrscale_fromscratch_closure'
        fromscratch_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
            flags.weights,version,iter,flags.step,e)
        
        #Fine Tuned file
        version = 'H1_Nov_ModelLists_1lrscale_closure_finetuned'
        finetuned_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
            flags.weights,version,iter,flags.step,e)

        #Pre-Trained file
        version = 'H1_Nov_ModelLists_1lrscale_closure_pretrained'
        pretrained_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
            flags.weights,version,iter,flags.step,e)


        history_baseline = load_pickle(fromscratch_file)
        history_finetuned = load_pickle(finetuned_file)
        history_pretrained = load_pickle(pretrained_file)

        # print("From Scratch: ", history_baseline)
        # print("Pretrained: ", history_finetuned)

        plot_dict = {
            f'FromScratch_Ens{e}':history_baseline['val_loss'],
            f'FineTuned_Ens{e}':history_finetuned['val_loss'],
            f'PreTrained_Ens{e}':history_pretrained['val_loss'],
        }

        _,ax0 = utils.PlotRoutine(
            plot_dict,xlabel='Epochs',ylabel='Validation Loss',axes=axes)
        ax0.set_yscale('log')
        ax0.set_ylim(0.0, 0.1)
        ax0.set_xlim(0.0, 100)

    #Mess with Legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate 'Baseline' and 'Pre-trained' entries
    fromscratch_entries = [(h, l) for h, l in zip(handles, labels) if 'FromScratch' in l]
    finetuned_entries = [(h, l) for h, l in zip(handles, labels) if 'FineTuned' in l]
    pretrained_entries = [(h, l) for h, l in zip(handles, labels) if 'PreTrained' in l]

    # Sort entries to ensure correct order (optional)
    fromscratch_entries.sort(key=lambda x: x[1])
    finetuned_entries.sort(key=lambda x: x[1])
    pretrained_entries.sort(key=lambda x: x[1])

    # Combine entries: Baseline first, then Pre-trained
    combined_entries = fromscratch_entries + finetuned_entries + pretrained_entries
    combined_handles, combined_labels = zip(*combined_entries)

    # Verify the ordering
    print("Combined Labels:")
    for label in combined_labels:
        print(label)

    # Create the legend with two columns
    plt.legend(combined_handles, combined_labels, ncol=2)
    plt.title(f"Iteration {iter} Step {flags.step}")
    plt.show()

    fig.savefig("{}/loss_all_ensembles_iter{}_step{}.pdf".format(flags.plot_folder,iter,flags.step),bbox_inches='tight')
