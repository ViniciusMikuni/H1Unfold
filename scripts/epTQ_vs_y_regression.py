import argparse
import gc
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
hvd.init()
utils.SetStyle()
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import matplotlib.pyplot as plt

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


def get_dataloaders(flags, mc_file_names):
    # Load the data from the simulations
    dataloaders = {}
    for name, file_path in mc_file_names.items():
        if name=="Rapgap":
            data_folder = "/pscratch/sd/r/rmilton"
            # data_folder = flags.data_folder
            dataloaders[name] = Dataset([file_path],data_folder,is_mc=True,
                                    rank=hvd.rank(),size=hvd.size(),
                                    nmax=flags.nmax,pass_fiducial= False, use_reco=False, pass_gen_Empz = True)
        else:
            data_folder = flags.data_folder
            dataloaders[name] = Dataset([file_path],data_folder,is_mc=True,
                                    rank=hvd.rank(),size=hvd.size(),
                                    nmax=flags.nmax,pass_fiducial= False, use_reco=False, pass_gen_Empz = True)
        # print(name)
        

    return dataloaders


def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)
    mc_files = {"Rapgap_no_rad":"Rapgap_Eplus0607_NoRad_prep.h5"}
    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files : {mc_files.keys()}")
    dataloaders = get_dataloaders(flags, mc_files)
    utils.undo_standardizing(flags, dataloaders)
    y = dataloaders["Rapgap_no_rad"].event[:,1,None]
    epTQ = dataloaders["Rapgap_no_rad"].event[:,2]
    print("Training model")
    regression_model = GradientBoostingRegressor(
        n_estimators=2000,
    )
    regression_model.fit(y, epTQ)
    epTQ_pred = regression_model.predict(y)

    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(14,8))


    # No rad
    h0 = axes[0].hist2d(
        dataloaders["Rapgap_no_rad"].event[:, 1],
        dataloaders["Rapgap_no_rad"].event[:, 2],
        bins=(100, 100),
        range=((0, 1), (.4, 1)),
        norm=LogNorm()
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h0[3], ax=axes[0])

    # Prediction
    h1 = axes[1].hist2d(
        dataloaders["Rapgap_no_rad"].event[:, 1],
        epTQ_pred,
        bins=(100, 100),
        range=((0, 1), (.4, 1)),
        norm=LogNorm()
    )
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h1[3], ax=axes[1])

    plt.tight_layout()
    plt.savefig("y_vs_epTQ_truth_vs_pred_nonfiducial_norad.pdf")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14,8))


    # No rad
    h0 = axes[0].hist2d(
        dataloaders["Rapgap_no_rad"].event[:, 0],
        dataloaders["Rapgap_no_rad"].event[:, 2],
        bins=(100, 100),
        range=((4, 10), (.4, 1)),
        norm=LogNorm()
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("log($Q^2$)")
    axes[0].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h0[3], ax=axes[0])

    # Prediction
    h1 = axes[1].hist2d(
        dataloaders["Rapgap_no_rad"].event[:, 0],
        epTQ_pred,
        bins=(100, 100),
        range=((4, 10), (.4, 1)),
        norm=LogNorm()
    )
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("log($Q^2$)")
    axes[1].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h1[3], ax=axes[1])

    plt.tight_layout()
    plt.savefig("logQ2_vs_epTQ_truth_vs_pred_nonfiducial_norad.pdf")
    plt.close()


    with open('eptQ_y_regressor_nonfiducial_1million_maxdepth10_Empzcut.pkl', 'wb') as file:
        pickle.dump(regression_model, file)
    

    mc_files = {"Rapgap":"Rapgap_Eplus0607_prep.h5"}
    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files : {mc_files.keys()}")
    dataloaders = get_dataloaders(flags, mc_files)
    utils.undo_standardizing(flags, dataloaders)
    y = dataloaders["Rapgap"].event[:,1,None]
    epTQ_pred = regression_model.predict(y)
    


    fig, axes = plt.subplots(1, 2, figsize=(14,8))


    # Truth
    h0 = axes[0].hist2d(
    dataloaders["Rapgap"].event[:, 1],
    dataloaders["Rapgap"].event[:, 2],
    bins=(100, 100),
    range=((0, 1), (.4, 1)),
    norm=LogNorm()
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h0[3], ax=axes[0])

    # Prediction
    h1 = axes[1].hist2d(
    dataloaders["Rapgap"].event[:, 1],
    epTQ_pred,
    bins=(100, 100),
    range=((0, 1), (.4, 1)),
    norm=LogNorm()
    )
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h1[3], ax=axes[1])

    plt.tight_layout()
    plt.savefig("y_vs_epTQ_truth_vs_pred_nonfiducial.pdf")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14,8))


    # Truth
    h0 = axes[0].hist2d(
        dataloaders["Rapgap"].event[:, 0],
        dataloaders["Rapgap"].event[:, 2],
        bins=(100, 100),
        range=((4, 10), (.4, 1)),
        norm=LogNorm()
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("log($Q^2$)")
    axes[0].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h0[3], ax=axes[0])

    # Prediction
    h1 = axes[1].hist2d(
        dataloaders["Rapgap"].event[:, 0],
        epTQ_pred,
        bins=(100, 100),
        range=((4, 10), (.4, 1)),
        norm=LogNorm()
    )
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("log($Q^2$)")
    axes[1].set_ylabel(r"$e p_T / Q$")
    plt.colorbar(h1[3], ax=axes[1])

    plt.tight_layout()
    plt.savefig("logQ2_vs_epTQ_truth_vs_pred_nonfiducial.pdf")
    plt.close()

if __name__ == "__main__":
    main()
