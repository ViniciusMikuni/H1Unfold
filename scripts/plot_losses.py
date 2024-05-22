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
parser.add_option('--iter', default=0, help='Iteration to load')
parser.add_option('--step', default=1, help='Step to load')
(flags, args) = parser.parse_args()

opt=utils.LoadJson(flags.config)
version = opt['NAME']

baseline_file = '{}/OmniFold_{}_iter{}_step{}.pkl'.format(flags.weights,version,flags.iter,flags.step)
ft_file = '{}/OmniFold_{}_pretrained_iter{}_step{}.pkl'.format(flags.weights,version,flags.iter,flags.step)


        
history_baseline = load_pickle(baseline_file)
history_ft = load_pickle(ft_file)

print(history_baseline)

plot_dict = {
    'Baseline':history_baseline['val_loss'],
    'Pre-trained':history_ft['val_loss'],
}

fig,ax = utils.PlotRoutine(plot_dict,xlabel='Epochs',ylabel='Validation Loss')
fig.savefig("{}/loss_{}_{}.pdf".format(flags.plot_folder,flags.iter,flags.step),bbox_inches='tight')
