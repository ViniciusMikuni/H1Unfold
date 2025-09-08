import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import json
from omnifold_QEDcorrections import  Multifold
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow.keras.backend as K
import utils
from dataloader_QEDcorrections import Dataset
# tf.random.set_seed(1234)
# np.random.seed(1234)

if __name__ == "__main__":
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/H1/h5/', help='Folder containing data and MC files')
    parser.add_argument('--config', default='config_general.json', help='Basic config file containing general options')
    parser.add_argument('--config_omnifold', default='config_omnifold.json', help='Basic config file containing general options')
    parser.add_argument('--closure', action='store_true', default=False,help='Train omnifold for a closure test using simulation')
    parser.add_argument('--verbose', action='store_true', default=False,help='Display additional information during training')
    parser.add_argument('--nmax', type=int, default=20_000_000, help='Maximum number of events to use in training')
    flags = parser.parse_args()

    if flags.verbose:
        print(80*'#')
        print("Total hvd size {}, rank: {}, local size: {}, local rank{}".format(hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank()))
        print(80*'#')

    opt=utils.LoadJson(flags.config)
    version = opt['NAME']

    norad_mc_file_name = ["Rapgap_Eplus0607_NoRad_prep.h5"]
    mc_file_names = ['Rapgap_Eplus0607_prep.h5']
    version += '_radiation'

    norad_mc = Dataset(norad_mc_file_name,flags.data_folder,is_mc=True,
                   rank=hvd.rank(),size=hvd.size(),nmax=flags.nmax) #match the normalization from MC files

    mc = Dataset(mc_file_names,flags.data_folder,is_mc=True,
                 rank=hvd.rank(),size=hvd.size(),nmax=flags.nmax)
        
    K.clear_session()
    mfold = Multifold(
        version = version,
        config_file = flags.config_omnifold,
        verbose = flags.verbose,        
        pretrain=True
    )

    mfold.mc = mc
    mfold.data = norad_mc
    mfold.Preprocessing()
    mfold.Unfold()
    print(mfold.weights_pull)