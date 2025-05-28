from optparse import OptionParser
import utils
import pickle

utils.SetStyle()


def load_pickle(f):
    with open(f, "rb") as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict


parser = OptionParser(usage="%prog [opt]  inputFiles")
<<<<<<< HEAD
parser.add_option('--weights', default='../weights', help='Folder to store trained weights')
parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
parser.add_option('--config', default='config_general.json', help='Basic config file containing general options')
parser.add_option('--iter', default=4, help='Iteration to load')
parser.add_option('--step', default=2, help='Step to load')
=======
parser.add_option(
    "--weights", default="../weights", help="Folder to store trained weights"
)
parser.add_option(
    "--plot_folder",
    type="string",
    default="../plots",
    help="Folder to save the outputs",
)
parser.add_option(
    "--config",
    default="config_general.json",
    help="Basic config file containing general options",
)
parser.add_option("--iter", default=0, help="Iteration to load")
parser.add_option("--step", default=1, help="Step to load")
>>>>>>> origin/main
(flags, args) = parser.parse_args()

opt = utils.LoadJson(flags.config)
version = opt["NAME"]

<<<<<<< HEAD
N_ensemble = 5
N_iteration = 5
=======
baseline_file = "{}/OmniFold_{}_iter{}_step{}.pkl".format(
    flags.weights, version, flags.iter, flags.step
)
ft_file = "{}/OmniFold_{}_pretrained_iter{}_step{}.pkl".format(
    flags.weights, version, flags.iter, flags.step
)
>>>>>>> origin/main

for iter in range(N_iteration):

<<<<<<< HEAD
    axes = []
    fig, gs = utils.SetGrid(ratio=False) 
    ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1 = None
    axes.append(ax0)
    axes.append(ax1)
    axes.append(fig)

    N_ensemble = 3
    for e in range(N_ensemble):
        print(f"Plotting Ensemble {e+1}/{N_ensemble}")

        #BASELINE  = From Scratch. No pretrained loading, no finetune
        version = 'H1_Nov_ModelLists_1lrscale_fromscratch_closure'
        baseline_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
            flags.weights,version,iter,flags.step,e)
        
=======
history_baseline = load_pickle(baseline_file)
history_ft = load_pickle(ft_file)
>>>>>>> origin/main

        #PreTrained
        version = 'H1_Nov_ModelLists_1lrscale_closure_pretrained'
        ft_file = '{}/OmniFold_{}_iter{}_step{}_ensemble{}.pkl'.format(
            flags.weights,version,iter,flags.step,e)

<<<<<<< HEAD

        history_baseline = load_pickle(baseline_file)
        history_ft = load_pickle(ft_file)

        # print("From Scratch: ", history_baseline)
        # print("Pretrained: ", history_ft)

        plot_dict = {
            f'FromScratch_Ens{e}':history_baseline['val_loss'],
            f'PreTrained_Ens{e}':history_ft['val_loss'],
        }


        _,_ = utils.PlotRoutine(plot_dict,xlabel='Epochs',ylabel='Validation Loss',axes=axes)
        # ax0.set_yscale('log')
        # ax0.set_ylim(0.0, 0.1)

    #Mess with Legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate 'Baseline' and 'Pre-trained' entries
    baseline_entries = [(h, l) for h, l in zip(handles, labels) if 'FromScratch' in l]
    pretrained_entries = [(h, l) for h, l in zip(handles, labels) if 'PreTrained' in l]

    # Sort entries to ensure correct order (optional)
    baseline_entries.sort(key=lambda x: x[1])
    pretrained_entries.sort(key=lambda x: x[1])

    # Combine entries: Baseline first, then Pre-trained
    combined_entries = baseline_entries + pretrained_entries
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
=======
plot_dict = {
    "Baseline": history_baseline["val_loss"],
    "Pre-trained": history_ft["val_loss"],
}

fig, ax = utils.PlotRoutine(plot_dict, xlabel="Epochs", ylabel="Validation Loss")
fig.savefig(
    "{}/loss_{}_{}.pdf".format(flags.plot_folder, flags.iter, flags.step),
    bbox_inches="tight",
)
>>>>>>> origin/main
