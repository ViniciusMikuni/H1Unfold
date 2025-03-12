import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import sys, gc
import utils
import matplotlib.pyplot as plt
import pickle
import utils
# from plot_superEnsembles import get_version

utils.SetStyle()


def load_pickle(f):
    with open(f, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict


parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option('--weights', default='../weights', help='Folder to store trained weights')
parser.add_option('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_option('--iter', default=4, help='Iteration to load')
parser.add_option('--n_jobs', type=int, default=0, help='number of jobs (super-ensembels)')
(flags, args) = parser.parse_args()

opt=utils.LoadJson(flags.config)
version = opt['NAME']

N_ensemble = 5
N_iteration = 5

out_path = f'../plots/{version}_LossPlots/'
os.makedirs(out_path, exist_ok=True)

for jobID in range(flags.n_jobs):
    ''' loop through jobs, then iters,  then ensmbles
    decided against averaging loss plots, and will instead show
    a single representative example. For reference, see H1OmniFold paper
    for the first pre-trained vs baseline val plot. shows a lower loss,
    but only marginally faster training length (~25%)
    '''
    for iter in range(N_iteration):

        for step in range(1,3):

            axes = []
            fig, gs = utils.SetGrid(ratio=False) 
            ax0 = plt.subplot(gs[0])
            # ax1 = plt.subplot(gs[1], sharex = ax0)
            ax1 = None
            axes.append(ax0)
            axes.append(ax1)
            axes.append(fig)


            minBase, maxBase = 1e9,-1
            minPT, maxPT = 1e9,-1

            for e in range(N_ensemble):
                print(f"Plotting Ensemble {e+1}/{N_ensemble}")

                #BASELINE  = From Scratch. No pretrained loading, no finetune
                # model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(
                #     flags.weights,version,flags.n_iter-1)
                #OmniFold_H1_Feb_Seed1234_emaGetSetWeighs_archL61_closure_job4_pretrained_iter4_step2_ensemble3.pkl
                fromscratch_file = f'{flags.weights}/OmniFold_{version}_closure_job{jobID}_fromscratch_iter{iter}_step{step}_ensemble{e}.pkl'
                # fromscratch_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
                    # flags.weights, version + '_closure_fromscratch',iter,flags.step,e)
                print(fromscratch_file)
                print('OmniFold_H1_Feb_Seed1234_emaGetSetWeighs_archL61_closure_job4_pretrained_iter4_step2_ensemble3.pkl')

                #Fine Tuned file
                finetuned_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
                    flags.weights,version + '_closure_finetuned',iter,step,e)

                #Pre-Trained file
                pretrained_file = f'{step}/OmniFold_{version}_closure_job{jobID}_pretrained_iter{iter}_step{step}_ensemble{e}.pkl'
                pretrained_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
                    flags.weights,version + '_closure_pretrained',iter,step,e)

                history_baseline = load_pickle(fromscratch_file)
                history_finetuned = load_pickle(finetuned_file)
                history_pretrained = load_pickle(pretrained_file)

                # print("From Scratch: ", history_baseline)
                # print("Pretrained: ", history_finetuned)

                plot_dict = {
                    f'Baseline_Ens{e}':history_baseline['val_loss'],
                    # f'Finetuned_Ens{e}':history_finetuned['val_loss'],
                    f'Pre-trained_Ens{e}':history_pretrained['val_loss'],
                }

                _,ax0 = utils.PlotRoutine(
                    plot_dict,xlabel='Epochs',ylabel='Validation Loss',axes=axes)
                ax0.set_yscale('log')

                minBase = min(min(history_baseline['val_loss']), minBase)
                maxBase = max(max(history_baseline['val_loss']), maxBase)
                minPT = min(min(history_pretrained['val_loss']), minPT)
                maxPT = max(max(history_pretrained['val_loss']), maxPT)


            #Mess with Legend

            # ax0.set_ylim(min(minBase,minPT)/2, max(maxBase,maxPT)*2)
            # ax0.set_ylim(0.070, 0.082)
            # ax0.set_xlim(0.0, 100)

            handles, labels = plt.gca().get_legend_handles_labels()

            # Separate 'Baseline' and 'Pre-trained' entries
            fromscratch_entries = [(h, l) for h, l in zip(handles, labels) if 'Baseline' in l]
            finetuned_entries = [(h, l) for h, l in zip(handles, labels) if 'Finetuned' in l]
            pretrained_entries = [(h, l) for h, l in zip(handles, labels) if 'Pre-trained' in l]

            # Sort entries to ensure correct order (optional)
            fromscratch_entries.sort(key=lambda x: x[1])
            finetuned_entries.sort(key=lambda x: x[1])
            pretrained_entries.sort(key=lambda x: x[1])

            # Combine entries: Baseline first, then Pre-trained
            # combined_entries = fromscratch_entries + finetuned_entries + pretrained_entries
            combined_entries = fromscratch_entries + pretrained_entries
            combined_handles, combined_labels = zip(*combined_entries)

            # Verify the ordering
            print("Combined Labels:")
            for label in combined_labels:
                print(label)

            # Create the legend with two columns
            plt.legend(combined_handles, combined_labels, ncol=2,fontsize=10)
            plt.title(f"Iteration {iter} Step {step}")
            plt.show()

            version = opt['NAME']
            print("{}/{}_job{}_LossAllEnsembles_iter{}_step{}.pdf".format(
                out_path,version, jobID, iter, step))
            fig.savefig("{}/{}_job{}_LossAllEnsembles_iter{}_step{}.pdf".format(
                out_path,version, jobID, iter, step), bbox_inches='tight')
            # fromscratch_file = f'{flags.weights}/OmniFold_{version}_closure_job{jobID}_fromscratch_iter{iter}_step{flags.step}_ensemble{e}.pkl'
