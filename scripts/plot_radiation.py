import argparse
import gc
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from plot_utils import plot_particles_radiation, plot_event_radiation, plot_event_radiation_2D, plot_event_radiation_2D_fractions, plot_Empz,plot_radiative_event_topologies, calculate_Empz
hvd.init()
utils.SetStyle()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/global/cfs/cdirs/m3246/H1/h5/",
        help="Folder containing data and MC files",
    )
    # parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument(
        "--weights", default="../weights", help="Folder to store trained weights"
    )
    parser.add_argument(
        "--load_pretrain",
        action="store_true",
        default=False,
        help="Load pretrained model instead of starting from scratch",
    )
    parser.add_argument(
        "--config",
        default="config_general.json",
        help="Basic config file containing general options",
    )
    parser.add_argument(
        "--plot_folder", default="../plots", help="Folder to store plots"
    )
    parser.add_argument(
        "--reco", action="store_true", default=False, help="Plot reco level  results"
    )
    # parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
    parser.add_argument(
        "--sys", action="store_true", default=False, help="Load systematic variations"
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        default=False,
        help="Show the results based on closure instead of data",
    )
    # parser.add_argument('--closure', action='store_true', default=False,help='Plot closure results')
    parser.add_argument(
        "--niter", type=int, default=0, help="Omnifold iteration to load"
    )
    parser.add_argument(
        "--nmax", type=int, default=1000000, help="Maximum number of events to load"
    )
    parser.add_argument("--img_fmt", default="pdf", help="Format of the output figures")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Increase print level"
    )
    parser.add_argument(
        "--dataset",
        default="ep",
        help="Choice between ep or em datasets",
    )

    flags = parser.parse_args()

    if flags.blind and flags.reco:
        raise ValueError("Unable to run blinded and reco modes at the same time")
    return flags
def gather_data(dataloaders):
    for dataloader in dataloaders:
        num_valid_particles = np.sum(dataloaders[dataloader].mask, axis=1)
        dataloaders[dataloader].mask = np.reshape(dataloaders[dataloader].mask,(-1))
        dataloaders[dataloader].part = hvd.allgather(tf.constant(dataloaders[dataloader].part.reshape(
            (-1,dataloaders[dataloader].part.shape[-1]))[dataloaders[dataloader].mask])).numpy()
        
        dataloaders[dataloader].event = hvd.allgather(tf.constant(dataloaders[dataloader].event)).numpy()
        dataloaders[dataloader].weight = hvd.allgather(tf.constant(dataloaders[dataloader].weight)).numpy()
        dataloaders[dataloader].mask = hvd.allgather(tf.constant(dataloaders[dataloader].mask)).numpy()
        dataloaders[dataloader].num_valid_particles = hvd.allgather(tf.constant(num_valid_particles)).numpy()
        dataloaders[dataloader].theta_per_particle = hvd.allgather(tf.constant(dataloaders[dataloader].theta_per_particle)).numpy()
        dataloaders[dataloader].Empz_per_particle = hvd.allgather(tf.constant(dataloaders[dataloader].Empz_per_particle)).numpy()
        dataloaders[dataloader].delta_phi = hvd.allgather(tf.constant(dataloaders[dataloader].delta_phi)).numpy()


def get_dataloaders(flags, mc_file_names):
    # Load the data from the simulations
    dataloaders = {}
    for name, file_path in mc_file_names.items():
        if name=="Rapgap":
            data_folder = "/pscratch/sd/r/rmilton"
            # data_folder = flags.data_folder
            dataloaders[name] = Dataset([file_path],data_folder,is_mc=True,
                                    rank=hvd.rank(),size=hvd.size(),
                                    nmax=flags.nmax,pass_fiducial= True, use_reco=False, pass_gen_Empz = True, rescale_eptQ=True)
        else:
            data_folder = flags.data_folder
            dataloaders[name] = Dataset([file_path],data_folder,is_mc=True,
                                    rank=hvd.rank(),size=hvd.size(),
                                    nmax=flags.nmax,pass_fiducial= True, use_reco=False, pass_gen_Empz = True)
        # print(name)
        

    return dataloaders




def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)
    mc_files = {"Rapgap":"Rapgap_Eplus0607_prep.h5", "Rapgap_no_rad":"Rapgap_Eplus0607_NoRad_prep.h5"}
    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files : {mc_files.keys()}")
    dataloaders = get_dataloaders(flags, mc_files)
    
    for dataset in dataloaders:
        if hvd.rank() == 0:
            print(np.sum(dataloaders[dataset].weight))
    QED_corrections = {}
    for dataset in dataloaders:
        if flags.verbose and hvd.rank() == 0:
            print(f"Evaluating weights for dataset {dataset}")
        QED_corrections[dataset] = utils.evaluate_model(flags, opt, dataset, dataloaders, QED_corrections=True)
        if hvd.rank() == 0:
            print(f"Num events after cuts in {dataset}: ", len(QED_corrections[dataset]))
    if hvd.rank() == 0:
        print("Done with network evaluation")
    # Important to only undo the preprocessing after the weights are derived!
    utils.undo_standardizing(flags, dataloaders)

    num_part = dataloaders["Rapgap"].part.shape[1]
    
    calculate_Empz(dataloaders)

    gather_data(dataloaders)
    plot_particles_radiation(flags, dataloaders, QED_corrections, opt["NAME"], num_part=num_part)
    plot_event_radiation(flags, dataloaders, QED_corrections, opt["NAME"])
    plot_event_radiation_2D(flags, dataloaders, QED_corrections, opt["NAME"])
    plot_event_radiation_2D_fractions(flags, dataloaders, QED_corrections, opt["NAME"])
    plot_Empz(flags, dataloaders, QED_corrections, opt["NAME"])
    plot_radiative_event_topologies(flags, dataloaders, QED_corrections, opt["NAME"])



if __name__ == "__main__":
    main()
