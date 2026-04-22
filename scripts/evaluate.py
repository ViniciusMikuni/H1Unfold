import argparse
import os
from dataloader import Dataset
import utils
import horovod.tensorflow as hvd
import h5py as h5

hvd.init()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/global/cfs/cdirs/m3246/H1/h5/",
        help="Folder containing data and MC files",
    )
    # parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
    parser.add_argument(
        "--weights",
        default="/global/cfs/cdirs/m3246/H1/weights/",
        help="Folder to store trained weights",
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
    parser.add_argument("--file", default="Rapgap", help="File to load")
    parser.add_argument(
        "--dataset",
        default="ep",
        help="Choice between ep or em datasets",
    )

    parser.add_argument(
        "--niter", type=int, default=4, help="Omnifold iteration to load"
    )

    parser.add_argument(
        "--nmax", type=int, default=-1, help="Maximum number of events to load"
    )

    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Load models for bootstrapping",
    )
    parser.add_argument(
        "--nboot", type=int, default=100, help="Number of bootstrap models to load"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Increase print level"
    )
    flags = parser.parse_args()

    return flags


def get_dataloaders(flags, file_names):
    # Load the data from the simulations
    dataloaders = {}

    for name in file_names:
        dataloaders[name] = Dataset(
            [name],
            flags.data_folder,
            is_mc=True,
            rank=hvd.rank(),
            size=hvd.size(),
            nmax=None if flags.nmax < 0 else flags.nmax,
            pass_fiducial=False,
            pass_reco=False,
        )

    return dataloaders


def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)

    mc_files = [utils.get_sample_name(flags.file, flags.dataset)]

    if flags.verbose and hvd.rank() == 0:
        print(f"Will load the following files : {mc_files[0]}")

    dataloaders = get_dataloaders(flags, mc_files)

    assert "data" not in flags.file, "ERROR: Data does not need to be evaluated!"

    weights = {}
    for dataset in dataloaders:
        if flags.verbose and hvd.rank() == 0:
            print(f"Evaluating weights for dataset {dataset}")

        if flags.bootstrap:
            for i in range(1, flags.nboot):
                if hvd.rank() == 0:
                    print(f"Evaluating boot {i}")
                weights[f"unfolded_weights_boots{str(i)}"] = utils.evaluate_model(
                    flags, opt, dataset, dataloaders, bootstrap=True, nboot=i
                )

        weights["unfolded_weights"] = utils.evaluate_model(
            flags, opt, dataset, dataloaders
        )
        if "Rapgap" in flags.file and "sys" not in flags.file:
            weights["unfolded_weights_closure"] = utils.evaluate_model(
                flags,
                opt,
                dataset,
                dataloaders,
                version=(
                    opt["NAME"] + f"_{flags.dataset}_closure" + "_pretrained"
                    if flags.load_pretrain
                    else ""
                ),
            )

    if hvd.rank() == 0:
        print("Done with network evaluation, saving new files...")
        replace_string = f"unfolded_niter_{flags.niter}"
        input_file_name = os.path.join(
            flags.data_folder, utils.get_sample_name(flags.file, flags.dataset)
        )
        output_file_name = input_file_name.replace("prep", replace_string)
        output_file_name = os.path.join(flags.data_folder, output_file_name)

        with (
            h5.File(input_file_name, "r") as src_file,
            h5.File(output_file_name, "w") as dst_file,
        ):
            # Copy all previous content
            def visitor(name, node):
                if isinstance(node, h5.Dataset):
                    src_file.copy(name, dst_file, name)

            src_file.visititems(visitor)
            # save weights
            for w_name in weights:
                dst_file.create_dataset(w_name, data=weights[w_name])


if __name__ == "__main__":
    main()
