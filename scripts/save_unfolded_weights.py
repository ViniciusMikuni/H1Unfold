import numpy as np
import argparse
import os
import gc
from dataloader import Dataset
from utils import *
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
    parser.add_argument(
        "--output_folder",
        default="/pscratch/sd/t/twamorka/h1/batchfiles/",
        help="",
    )
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
        "--reco", action="store_true", default=False, help="Plot reco level results"
    )
    # parser.add_argument(
    #     "--file", default="Rapgap_Eplus0607_prep.h5", help="File to load"
    # )
    parser.add_argument(
        "--niter", type=int, default=4, help="Omnifold iteration to load"
    )
    parser.add_argument(
        "--nmax", type=int, default=10_000_000, help="Maximum number of events to load"
    )
    parser.add_argument(
        "--batch_size", type=int, default=30_000, help="Batch size for processing"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Load models for bootstrapping",
    )
    # parser.add_argument(
    #     "--pre_weights_file",
    #     default=None,
    #     help="Path to HDF5 file containing pre-evaluated weights",
    # )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to HDF5 file containing pre-evaluated weights",
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


def get_dataloaders_batch(flags, file_names, batch_start, batch_end):
    """Load a batch window [batch_start, batch_end) entirely on this rank"""
    dataloaders = {}

    for name in file_names:
        if "data" in name:
            dataloaders[name] = Dataset(
                [name],
                flags.data_folder,
                is_mc=False,
                rank=0,
                size=1,
                nmax=flags.nmax,
                global_start=batch_start,
                global_end = batch_end,
                pass_reco=True,
            )
        else:
            dataloaders[name] = Dataset(
                [name],
                flags.data_folder,
                is_mc=True,
                rank=0,
                size=1,
                nmax=flags.nmax,
                global_start=batch_start,
                global_end = batch_end,
                pass_fiducial=not flags.reco,
                pass_reco=flags.reco,
            )

    return dataloaders


def get_deltaphi(jet, elec):
    delta_phi = np.abs(np.pi + jet[:, :, 2] - elec[:, None, 4])
    delta_phi[delta_phi > 2 * np.pi] -= 2 * np.pi
    return delta_phi


def load_weights_slice(flags, global_start, global_end):
    """Load a slice of weights from the HDF5 file"""
    weights = {}

    if "data" not in flags.file:
        print(f"[rank {hvd.rank()}] load_weights_slice: opening weights file", flush=True)
        with h5.File(os.path.join(flags.data_folder, flags.file), "r") as fh5:
            total_size = fh5["reco_event_features"].shape[0]
            start = global_start
            end = min(global_end, total_size)

            print(f"[rank {hvd.rank()}] load_weights_slice: reading [{start}, {end})", flush=True)

            pass_gen = fh5["gen_event_features"][start:end, -1] == 1
            pass_reco = fh5["reco_event_features"][start:end, -1] == 1

            if not flags.reco:
                combined_mask = pass_gen
            else:
                combined_mask = pass_reco

            if "unfolded_weights" in fh5:
                weights[flags.file] = fh5["unfolded_weights"][start:end][combined_mask]
            else:
                raise KeyError("File has no dataset 'unfolded_weights'")

            if flags.bootstrap and "Rapgap" in flags.file:
                for i in range(1, flags.nboot + 1):
                    key = f"unfolded_weights_boots{i}"
                    if key in fh5:
                        weights[str(i)] = fh5[key][start:end][combined_mask]
                    else:
                        print(f"[rank {hvd.rank()}] Warning: bootstrap key {key} not found, skipping")

            if "unfolded_weights_closure" in fh5:
                weights["closure"] = fh5["unfolded_weights_closure"][start:end][combined_mask]

        print(f"[rank {hvd.rank()}] load_weights_slice: done", flush=True)

    return weights


def process_batch(flags, batch_start, batch_end, weights_dict, opt):
    """Load and process events in [batch_start, batch_end) split across ranks"""

    print(f"[rank {hvd.rank()}] process_batch: loading data [{batch_start}, {batch_end})", flush=True)

    mc_files = [flags.file]
    dataloaders = get_dataloaders_batch(flags, mc_files, batch_start, batch_end)

    print(f"[rank {hvd.rank()}] process_batch: undo_standardizing", flush=True)
    undo_standardizing(flags, dataloaders)
    print(f"[rank {hvd.rank()}] process_batch: cluster_jets", flush=True)
    cluster_jets(dataloaders, n_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    print(f"[rank {hvd.rank()}] process_batch: cluster_breit", flush=True)
    cluster_breit(flags, dataloaders)

    del dataloaders[flags.file].part, dataloaders[flags.file].mask
    gc.collect()

    dataset = dataloaders[flags.file]

    print(f"[rank {hvd.rank()}] process_batch: extracting results ({dataset.all_jets.shape[0]} events)", flush=True)
    results = {
        'jet_pt': dataset.all_jets[:, :, 0],
        'jet_breit_pt': dataset.all_jets_breit[:, :, 0],
        'deltaphi': get_deltaphi(dataset.all_jets, dataset.event),
        'jet_tau10': dataset.all_jets[:, :, 4],
        'zjet': dataset.all_jets[:, :, 9],
        'zjet_breit': dataset.all_jets_breit[:, :, 7],
    }

    if "data" not in flags.file:
        results['mc_weights'] = dataset.weight
        results['weights'] = weights_dict

    del dataloaders
    gc.collect()

    print(f"[rank {hvd.rank()}] process_batch: done", flush=True)
    return results




def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)

    total_events = flags.nmax
    batch_size = flags.batch_size
    num_batches = (total_events + batch_size - 1) // batch_size

    print(f"[rank {hvd.rank()}] total_events={total_events}, num_batches={num_batches}, hvd.size()={hvd.size()}", flush=True)

    replace_string = f"unfolded_niter_{flags.niter}"
    if flags.reco:
        replace_string += "_reco"
    if flags.bootstrap:
        replace_string += "_boot"

    # Each rank independently processes the batch assigned to it.
    # Ranks with index >= num_batches have nothing to do.
    batch_idx = hvd.rank()
    if batch_idx >= num_batches:
        print(f"[rank {hvd.rank()}] no batch to process, exiting", flush=True)
        return

    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, total_events)

    print(f"[rank {hvd.rank()}] processing batch {batch_idx} events [{batch_start}, {batch_end})", flush=True)

    if "data" not in flags.file:
        weights_dict = load_weights_slice(flags, batch_start, batch_end)
    else:
        weights_dict = {}

    # Load the full batch on this rank (size=1, rank=0 within the batch)
    results = process_batch(flags, batch_start, batch_end, weights_dict, opt)

    stem = os.path.splitext(flags.file)[0]
    base_name = stem.split("_unfolded")[0].removesuffix("_prep") + "_"
    output_file_name = f"{base_name}{replace_string}_batch{batch_idx:04d}.h5"
    output_path = os.path.join(flags.output_folder, output_file_name)

    if os.path.exists(output_path):
        os.remove(output_path)

    with h5.File(output_path, 'w') as fh5:
        if "data" not in flags.file and 'weights' in results and len(results['weights']) > 0:
            if flags.file in results['weights']:
                fh5.create_dataset("weights_nominal", data=results['weights'][flags.file])
            if flags.bootstrap and "Rapgap" in flags.file:
                for i in range(1, flags.nboot):
                    if str(i) in results['weights']:
                        fh5.create_dataset(f"weights{i}", data=results['weights'][str(i)])
            else:
                if flags.file in results['weights']:
                    fh5.create_dataset("weights", data=results['weights'][flags.file])

            if 'mc_weights' in results:
                fh5.create_dataset("mc_weights", data=results['mc_weights'])

            if "closure" in results['weights']:
                fh5.create_dataset("closure_weights", data=results['weights']["closure"])

        fh5.create_dataset("jet_pt", data=results['jet_pt'])
        fh5.create_dataset("jet_breit_pt", data=results['jet_breit_pt'])
        fh5.create_dataset("deltaphi", data=results['deltaphi'])
        fh5.create_dataset("jet_tau10", data=results['jet_tau10'])
        fh5.create_dataset("zjet", data=results['zjet'])
        fh5.create_dataset("zjet_breit", data=results['zjet_breit'])

    n_entries = results['jet_pt'].shape[0]
    print(f"[rank {hvd.rank()}] saved batch {batch_idx} to {output_file_name} ({n_entries:,} entries)", flush=True)
    del results, weights_dict
    gc.collect()
    

if __name__ == "__main__":
    main()