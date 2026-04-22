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
        "--batch_size", type=int, default=30_000, help="Batch size for processing"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Load models for bootstrapping",
    )
    parser.add_argument(
        "--pre_weights_file",
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


def get_dataloaders_limited(flags, file_names, nmax_override):
    """Load data with a specific nmax limit"""
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
                nmax=nmax_override,
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
        with h5.File(flags.data_folder + flags.pre_weights_file, "r") as fh5:
            total_size = fh5["reco_event_features"].shape[0]
            
            per_rank = (global_end - global_start + hvd.size() - 1) // hvd.size()
            rank_start = global_start + hvd.rank() * per_rank
            rank_end = min(rank_start + per_rank, global_end, total_size)
            
            if rank_start >= rank_end:
                return {}
            
            pass_gen = fh5["gen_event_features"][rank_start:rank_end, -1] == 1
            pass_reco = fh5["reco_event_features"][rank_start:rank_end, -1] == 1
            
            if not flags.reco:
                combined_mask = pass_gen 
            else:
                combined_mask = pass_reco
            
            if "unfolded_weights" in fh5:
                weights[flags.file] = fh5["unfolded_weights"][rank_start:rank_end][combined_mask]
            else:
                raise KeyError("File has no dataset 'unfolded_weights'")
            
            if flags.bootstrap:
                for i in range(1, flags.nboot + 1):
                    key = f"unfolded_weights_boots{i}"
                    if key in fh5:
                        weights[str(i)] = fh5[key][rank_start:rank_end][combined_mask]
                    else:
                        if hvd.rank() == 0:
                            print(f"Warning: bootstrap key {key} not found, skipping")
            
            if "unfolded_weights_closure" in fh5:
                weights["closure"] = fh5["unfolded_weights_closure"][rank_start:rank_end][combined_mask]
    
    return weights


def process_batch(flags, batch_nmax_end, weights_dict, opt, global_batch_idx, events_processed_so_far):
    """Process a single batch by loading data with nmax limit"""
    
    if hvd.rank() == 0 and flags.verbose:
        print(f"Loading batch data with nmax={batch_nmax_end}...")
    
    mc_files = [flags.file]
    dataloaders = get_dataloaders_limited(flags, mc_files, batch_nmax_end)
    
    if hvd.rank() == 0 and flags.verbose:
        print(f"Processing batch (standardization, clustering)...")
    
    undo_standardizing(flags, dataloaders)
    cluster_jets(dataloaders)
    cluster_breit(flags, dataloaders)
    
    del dataloaders[flags.file].part, dataloaders[flags.file].mask
    gc.collect()
    
    dataset = dataloaders[flags.file]
    
    # Calculate how many events this rank has in the current dataset
    num_events_this_rank = dataset.all_jets.shape[0]
    
    # Skip events that were already processed in previous batches
    if events_processed_so_far < num_events_this_rank:
        start_idx = events_processed_so_far
        
        results = {
            'jet_pt': dataset.all_jets[start_idx:, :, 0],
            'jet_breit_pt': dataset.all_jets_breit[start_idx:, :, 0],
            'deltaphi': get_deltaphi(dataset.all_jets[start_idx:], dataset.event[start_idx:]),
            'jet_tau10': dataset.all_jets[start_idx:, :, 4],
            'zjet': dataset.all_jets[start_idx:, :, 9],
            'zjet_breit': dataset.all_jets_breit[start_idx:, :, 7],
        }
        
        if "data" not in flags.file:
            results['mc_weights'] = dataset.weight[start_idx:]
            results['weights'] = weights_dict
    else:
        # This rank has no new events for this batch
        results = {
            'jet_pt': np.array([]).reshape(0, dataset.all_jets.shape[1], 0),
            'jet_breit_pt': np.array([]).reshape(0, dataset.all_jets_breit.shape[1], 0),
            'deltaphi': np.array([]).reshape(0, dataset.all_jets.shape[1]),
            'jet_tau10': np.array([]).reshape(0, dataset.all_jets.shape[1], 0),
            'zjet': np.array([]).reshape(0, dataset.all_jets.shape[1], 0),
            'zjet_breit': np.array([]).reshape(0, dataset.all_jets_breit.shape[1], 0),
        }
        
        if "data" not in flags.file:
            results['mc_weights'] = np.array([])
            results['weights'] = weights_dict
    
    # Clean up dataloaders
    del dataloaders
    gc.collect()
    
    return results, num_events_this_rank


def gather_results_across_ranks(results):
    """Gather results from all ranks to rank 0"""
    gathered = {}
    
    # Gather each array separately
    for key in results.keys():
        if key == 'weights':
            # Handle nested weights dictionary
            gathered[key] = {}
            for weight_key in results[key].keys():
                gathered[key][weight_key] = hvd.allgather_object(results[key][weight_key])
        else:
            gathered[key] = hvd.allgather_object(results[key])
    
    # Only rank 0 concatenates the results
    if hvd.rank() == 0:
        concatenated = {}
        for key in gathered.keys():
            if key == 'weights':
                concatenated[key] = {}
                for weight_key in gathered[key].keys():
                    # Filter out None/empty arrays and concatenate
                    valid_arrays = [arr for arr in gathered[key][weight_key] if arr is not None and len(arr) > 0]
                    if valid_arrays:
                        concatenated[key][weight_key] = np.concatenate(valid_arrays, axis=0)
            else:
                # Filter out None/empty arrays and concatenate
                valid_arrays = [arr for arr in gathered[key] if arr is not None and len(arr) > 0]
                if valid_arrays:
                    concatenated[key] = np.concatenate(valid_arrays, axis=0)
        return concatenated
    else:
        return None


def main():
    utils.setup_gpus(hvd.local_rank())
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)
    
    # Calculate batch parameters
    total_events = flags.nmax
    batch_size = flags.batch_size
    num_batches = (total_events + batch_size - 1) // batch_size
    
    if hvd.rank() == 0:
        print(f"Total events to process: {total_events}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"Processing with {hvd.size()} MPI ranks")
    
    # Initialize output file base name
    replace_string = f"unfolded_{flags.niter}_centauro"
    if flags.reco:
        replace_string += "_reco"
    if flags.bootstrap:
        replace_string += "_boot"
    
    # Process data in batches
    output_files = []
    events_processed_so_far = 0  # Track events already processed by this rank
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_events)
        
        if hvd.rank() == 0:
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_idx + 1}/{num_batches}")
            print(f"Global events: {batch_start} to {batch_end}")
            print(f"{'='*60}")
        
        # Load weights for this batch slice
        if "data" not in flags.file:
            weights_dict = load_weights_slice(flags, batch_start, batch_end)
        else:
            weights_dict = {}
        
        batch_results, num_events_loaded = process_batch(
            flags, batch_end, weights_dict, opt, batch_idx, events_processed_so_far
        )
        
        events_processed_so_far = num_events_loaded
        
        if hvd.rank() == 0 and flags.verbose:
            print(f"Gathering results from all ranks...")
        
        aggregated_results = gather_results_across_ranks(batch_results)
        
        del batch_results, weights_dict
        gc.collect()
        
        if hvd.rank() == 0:
            base_name = flags.file.replace("prep.h5", "")
            output_file_name = f"{base_name}{replace_string}_batch{batch_idx:04d}.h5"
            output_path = os.path.join(flags.data_folder, output_file_name)
            output_files.append(output_file_name)
            
            with h5.File(output_path, 'w') as fh5:
                
                if "data" not in flags.file and 'weights' in aggregated_results and len(aggregated_results['weights']) > 0:
                    if flags.bootstrap:
                        for i in range(1, flags.nboot):
                            if str(i) in aggregated_results['weights']:
                                fh5.create_dataset(
                                    f"weights{i}", 
                                    data=aggregated_results['weights'][str(i)]
                                )
                    else:
                        if flags.file in aggregated_results['weights']:
                            fh5.create_dataset(
                                "weights", 
                                data=aggregated_results['weights'][flags.file]
                            )
                    
                    if 'mc_weights' in aggregated_results:
                        fh5.create_dataset(
                            "mc_weights", 
                            data=aggregated_results['mc_weights']
                        )
                    
                    if "closure" in aggregated_results['weights']:
                        fh5.create_dataset(
                            "closure_weights", 
                            data=aggregated_results['weights']["closure"]
                        )
                
                # Write jet features
                fh5.create_dataset("jet_pt", data=aggregated_results['jet_pt'])
                fh5.create_dataset("jet_breit_pt", data=aggregated_results['jet_breit_pt'])
                fh5.create_dataset("deltaphi", data=aggregated_results['deltaphi'])
                fh5.create_dataset("jet_tau10", data=aggregated_results['jet_tau10'])
                fh5.create_dataset("zjet", data=aggregated_results['zjet'])
                fh5.create_dataset("zjet_breit", data=aggregated_results['zjet_breit'])
            
            print(f"Saved aggregated batch {batch_idx + 1} to: {output_file_name}")
            if flags.verbose:
                print(f"  File contains data from {hvd.size()} ranks")
        
        # Clean up aggregated results
        if hvd.rank() == 0:
            del aggregated_results
            gc.collect()
    

if __name__ == "__main__":
    main()