import numpy as np
import argparse
import os
import gc
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
from plot_utils import *
import h5py as h5

hvd.init()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/pscratch/sd/v/vmikuni/H1v2/h5",
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
        "--reco", action="store_true", default=False, help="Plot reco level  results"
    )
    # parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
    parser.add_argument(
        "--file", default="Rapgap_Eplus0607_prep.h5", help="File to load"
    )
    parser.add_argument(
        "--niter", type=int, default=4, help="Omnifold iteration to load"
    )
    parser.add_argument(
        "--nmax", type=int, default=10_000_000, help="Maximum number of events to load"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Load models for bootstrapping",
    )
    parser.add_argument(
        "--nboot", type=int, default=50, help="Number of bootstrap models to load"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Increase print level"
    )
    parser.add_argument("--eec", action="store_true", default=False, help="Get EEC")
    flags = parser.parse_args()

    return flags


def get_dataloaders(flags, file_names):
    # Load the data from the simulations
    dataloaders = {}

    for name in file_names:
        if "data" in name:
            dataloaders[name] = Dataset(
                [name],
                flags.data_folder,
                is_mc=False,
                rank=hvd.rank(),
                size=hvd.size(),
                nmax=None,
                pass_reco=True,
            )

        else:
            dataloaders[name] = Dataset(
                [name],
                flags.data_folder,
                is_mc=True,
                rank=hvd.rank(),
                size=hvd.size(),
                nmax=flags.nmax,
                pass_fiducial=not flags.reco,
                pass_reco=flags.reco,
            )

    return dataloaders


def get_deltaphi(jet, elec):
    delta_phi = np.abs(np.pi + jet[:, :, 2] - elec[:, None, 4])
    delta_phi[delta_phi > 2 * np.pi] -= 2 * np.pi
    return delta_phi


def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)
    mc_files = [flags.file]

    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files : {mc_files.keys()}")

    dataloaders = get_dataloaders(flags, mc_files)

    if "data" not in flags.file:
        weights = {}
        for dataset in dataloaders:
            if flags.verbose and hvd.rank() == 0:
                print(f"Evaluating weights for dataset {dataset}")

            if flags.bootstrap:
                for i in range(1, flags.nboot):
                    weights[str(i)] = evaluate_model(
                        flags, opt, dataset, dataloaders, bootstrap=True, nboot=i
                    )
            else:
                weights[dataset] = evaluate_model(flags, opt, dataset, dataloaders)
                if "Rapgap" in flags.file and "sys" not in flags.file:
                    weights["closure"] = evaluate_model(
                        flags,
                        opt,
                        dataset,
                        dataloaders,
                        version=(
                            opt["NAME"] + "_closure" + "_pretrained"
                            if flags.load_pretrain
                            else ""
                        ),
                    )

    if hvd.rank() == 0:
        print("Done with network evaluation")
    # Important to only undo the preprocessing after the weights are derived!
    undo_standardizing(flags, dataloaders)

    cluster_jets(dataloaders)
    cluster_breit(flags, dataloaders)
    del dataloaders[flags.file].part, dataloaders[flags.file].mask
    gc.collect()
    gather_data(dataloaders)

    replace_string = f"unfolded_{flags.niter}"
    if flags.reco:
        replace_string += "_reco"
    if flags.bootstrap:
        replace_string += "_boot"

    output_file_name = flags.file.replace("prep", replace_string)

    if hvd.rank() == 0:
        with h5.File(os.path.join(flags.data_folder, output_file_name), "w") as fh5:
            if "data" not in flags.file:
                if flags.bootstrap:
                    for i in range(1, flags.nboot):
                        dset = fh5.create_dataset(f"weights{i}", data=weights[str(i)])
                else:
                    dset = fh5.create_dataset("weights", data=weights[flags.file])
                dset = fh5.create_dataset(
                    "mc_weights", data=dataloaders[flags.file].weight
                )
                if "closure" in weights:
                    dset = fh5.create_dataset(
                        "closure_weights", data=weights["closure"]
                    )

                
            dset = fh5.create_dataset(
                "jet_pt", data=dataloaders[flags.file].all_jets[:, :, 0]
            )
            dset = fh5.create_dataset(
                "jet_breit_pt", data=dataloaders[flags.file].all_jets_breit[:, :, 0]
            )
            dset = fh5.create_dataset(
                "deltaphi",
                data=get_deltaphi(
                    dataloaders[flags.file].all_jets, dataloaders[flags.file].event
                ),
            )
            dset = fh5.create_dataset(
                "jet_tau10", data=dataloaders[flags.file].all_jets[:, :, 4]
            )
            dset = fh5.create_dataset(
                "zjet", data=dataloaders[flags.file].all_jets[:, :, 9]
            )
            dset = fh5.create_dataset(
                "zjet_breit", data=dataloaders[flags.file].all_jets_breit[:, :, 7]
            )
            dset = fh5.create_dataset("eec", data=dataloaders[flags.file].eec[:, :, 0])
            dset = fh5.create_dataset(
                "E_wgt", data=dataloaders[flags.file].eec[:, :, 1]
            )  # per particle energy weighting
            dset = fh5.create_dataset(
                "theta", data=dataloaders[flags.file].eec[:, :, 2]
            )
            dset = fh5.create_dataset('zh', data=dataloaders[flags.file].zh)
            dset = fh5.create_dataset('jt', data=dataloaders[flags.file].jt)
            dset = fh5.create_dataset('jt_photon', data=dataloaders[flags.file].jt_photon)
            dset = fh5.create_dataset('jet_qt', data=dataloaders[flags.file].all_jets[:, :, 10])
            dset = fh5.create_dataset("EEC_energyweight", data=dataloaders[flags.file].EEC_energyweight)
            dset = fh5.create_dataset("R_L", data=dataloaders[flags.file].R_L)
    
if __name__ == "__main__":
    main()