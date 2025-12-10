# import numpy as np
# import argparse
# import os
# import gc
# from dataloader import Dataset
# # import utils
# from utils import *
# import horovod.tensorflow as hvd
# from plot_utils import *
# import h5py as h5

# hvd.init()


# def parse_arguments():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--data_folder",
#         default="/pscratch/sd/v/vmikuni/H1v2/h5",
#         help="Folder containing data and MC files",
#     )
#     # parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3246/vmikuni/H1v2/h5/', help='Folder containing data and MC files')
#     parser.add_argument(
#         "--weights", default="../weights", help="Folder to store trained weights"
#     )
#     parser.add_argument(
#         "--load_pretrain",
#         action="store_true",
#         default=False,
#         help="Load pretrained model instead of starting from scratch",
#     )
#     parser.add_argument(
#         "--config",
#         default="config_general.json",
#         help="Basic config file containing general options",
#     )
#     parser.add_argument(
#         "--reco", action="store_true", default=False, help="Plot reco level  results"
#     )
#     # parser.add_argument('--load', action='store_true', default=False,help='Load unfolded weights')
#     parser.add_argument(
#         "--file", default="Rapgap_Eplus0607_prep.h5", help="File to load"
#     )
#     parser.add_argument(
#         "--niter", type=int, default=4, help="Omnifold iteration to load"
#     )
#     parser.add_argument(
#         "--nmax", type=int, default=10_000_000, help="Maximum number of events to load"
#     )
#     parser.add_argument(
#         "--bootstrap",
#         action="store_true",
#         default=False,
#         help="Load models for bootstrapping",
#     )
#     parser.add_argument(
#         "--pre_weights_file",
#         default=None,
#         help="Path to HDF5 file containing pre-evaluated weights",
#     )
#     parser.add_argument(
#         "--nboot", type=int, default=50, help="Number of bootstrap models to load"
#     )
#     parser.add_argument(
#         "--verbose", action="store_true", default=False, help="Increase print level"
#     )
#     parser.add_argument("--eec", action="store_true", default=False, help="Get EEC")
#     flags = parser.parse_args()

#     return flags


# def get_dataloaders(flags, file_names):
#     # Load the data from the simulations
#     dataloaders = {}

#     for name in file_names:
#         if "data" in name:
#             dataloaders[name] = Dataset(
#                 [name],
#                 flags.data_folder,
#                 is_mc=False,
#                 rank=hvd.rank(),
#                 size=hvd.size(),
#                 nmax=None,
#                 pass_reco=True,
#             )

#         else:
#             # print("here 1")
#             # print(file_names)
#             dataloaders[name] = Dataset(
#                 [name],
#                 flags.data_folder,
#                 is_mc=True,
#                 rank=hvd.rank(),
#                 size=hvd.size(),
#                 nmax=flags.nmax,
#                 pass_fiducial=not flags.reco,
#                 pass_reco=flags.reco,
#             )

#     return dataloaders


# def get_deltaphi(jet, elec):
#     delta_phi = np.abs(np.pi + jet[:, :, 2] - elec[:, None, 4])
#     delta_phi[delta_phi > 2 * np.pi] -= 2 * np.pi
#     return delta_phi


# def main():
#     utils.setup_gpus(hvd.local_rank())
#     flags = parse_arguments()
#     opt = utils.LoadJson(flags.config)
#     mc_files = [flags.file]
    
#     dataloaders = get_dataloaders(flags, mc_files)
    
#     # Load weights file as a Dataset
#     weights_dataloader = get_dataloaders(flags, [flags.pre_weights_file])


#     for name, dataset in weights_dataloader.items():
#         print(f"{name}: {len(dataset.weight)} events")

#     weights = {}
    
#     if "data" not in flags.file:
#         # Get the dataset name (first key)
#         weights_dataset_name = list(weights_dataloader.keys())[0]
#         weights_dataset = weights_dataloader[weights_dataset_name]
        
#         # Now load the actual unfolded_weights from the HDF5 file

#         with h5.File(flags.data_folder+flags.pre_weights_file, "r") as fh5:
#             # Apply the same rank splitting that Dataset does
#             nmax = flags.nmax if flags.nmax > 0 else fh5["reco_event_features"].shape[0]
#             per_rank = (nmax + hvd.size() - 1) // hvd.size()
#             start = hvd.rank() * per_rank
#             end = min(start + per_rank, nmax)
            
#             pass_gen = fh5["gen_event_features"][start:end, -1] == 1 ## pass fiducial
#             pass_reco = fh5["reco_event_features"][start:end, -1] == 1 ## pass reco
            
#             if not flags.reco:  # pass_fiducial=True
#                 combined_mask = pass_gen 
#             else:  # pass_reco only
#                 combined_mask = pass_reco
            
#             if "unfolded_weights" in fh5:
#                 weights[flags.file] = fh5["unfolded_weights"][start:end][combined_mask]
#             else:
#                 raise KeyError("File has no dataset 'unfolded_weights'")
            
#             # Load bootstrap weights
#             if flags.bootstrap:
#                 for i in range(1, flags.nboot + 1):
#                     key = f"unfolded_weights_boots{i}"
#                     if key in fh5:
#                         weights[str(i)] = fh5[key][start:end][combined_mask]
#                     else:
#                         if hvd.rank() == 0:
#                             print(f"Warning: bootstrap key {key} not found, skipping")
            
#             # Load closure weights
#             if "unfolded_weights_closure" in fh5:
#                 weights["closure"] = fh5["unfolded_weights_closure"][start:end][combined_mask]
    
    
#     undo_standardizing(flags, dataloaders)
    
#     cluster_jets(dataloaders)
#     cluster_breit(flags, dataloaders)
#     del dataloaders[flags.file].part, dataloaders[flags.file].mask
#     gc.collect()
    
#     if hvd.rank() == 0:
#         print("Gathering data from all ranks...")

#     replace_string = f"unfolded_{flags.niter}_centauro"
#     if flags.reco:
#         replace_string += "_reco"
#     if flags.bootstrap:
#         replace_string += "_boot"

#     output_file_name = flags.file.replace("prep", replace_string)

#     if hvd.rank() == 0:
#             with h5.File(os.path.join(flags.data_folder, output_file_name), "w") as fh5:
#                 if "data" not in flags.file:
#                     if flags.bootstrap:
#                         for i in range(1, flags.nboot):
#                             dset = fh5.create_dataset(f"weights{i}", data=weights[str(i)])
#                     else:
#                         dset = fh5.create_dataset("weights", data=weights[flags.file])
#                     dset = fh5.create_dataset(
#                         "mc_weights", data=dataloaders[flags.file].weight
#                     )
#                     if "closure" in weights:
#                         dset = fh5.create_dataset(
#                             "closure_weights", data=weights["closure"]
#                         )

#                 dset = fh5.create_dataset(
#                     "jet_pt", data=dataloaders[flags.file].all_jets[:, :, 0]
#                 )
#                 dset = fh5.create_dataset(
#                     "jet_breit_pt", data=dataloaders[flags.file].all_jets_breit[:, :, 0]
#                 )
#                 dset = fh5.create_dataset(
#                     "deltaphi",
#                     data=get_deltaphi(
#                         dataloaders[flags.file].all_jets, dataloaders[flags.file].event
#                     ),
#                 )
#                 dset = fh5.create_dataset(
#                     "jet_tau10", data=dataloaders[flags.file].all_jets[:, :, 4]
#                 )
#                 dset = fh5.create_dataset(
#                     "zjet", data=dataloaders[flags.file].all_jets[:, :, 9]
#                 )
#                 dset = fh5.create_dataset(
#                     "zjet_breit", data=dataloaders[flags.file].all_jets_breit[:, :, 7]
#                 )
#                 # dset = fh5.create_dataset("eec", data=dataloaders[flags.file].eec[:, :, 0])
#                 # dset = fh5.create_dataset(
#                 #     "E_wgt", data=dataloaders[flags.file].eec[:, :, 1]
#                 # )  # per particle energy weighting
#                 # dset = fh5.create_dataset(
#                 #     "theta", data=dataloaders[flags.file].eec[:, :, 2]
#                 # )


# if __name__ == "__main__":
#     main()


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
            # Get total size
            total_size = fh5["reco_event_features"].shape[0]
            
            # Calculate this rank's portion of the global slice
            per_rank = (global_end - global_start + hvd.size() - 1) // hvd.size()
            rank_start = global_start + hvd.rank() * per_rank
            rank_end = min(rank_start + per_rank, global_end, total_size)
            
            if rank_start >= rank_end:
                # This rank has no data for this batch
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
            
            # Load bootstrap weights
            if flags.bootstrap:
                for i in range(1, flags.nboot + 1):
                    key = f"unfolded_weights_boots{i}"
                    if key in fh5:
                        weights[str(i)] = fh5[key][rank_start:rank_end][combined_mask]
                    else:
                        if hvd.rank() == 0:
                            print(f"Warning: bootstrap key {key} not found, skipping")
            
            # Load closure weights
            if "unfolded_weights_closure" in fh5:
                weights["closure"] = fh5["unfolded_weights_closure"][rank_start:rank_end][combined_mask]
    
    return weights


def process_batch(flags, batch_nmax_end, weights_dict, opt, global_batch_idx):
    """Process a single batch by loading data with nmax limit"""
    
    if hvd.rank() == 0 and flags.verbose:
        print(f"Loading batch data with nmax={batch_nmax_end}...")
    
    # Load batch data using nmax to limit how much is loaded
    mc_files = [flags.file]
    dataloaders = get_dataloaders_limited(flags, mc_files, batch_nmax_end)
    
    if hvd.rank() == 0 and flags.verbose:
        print(f"Processing batch (standardization, clustering)...")
    
    # Process the batch
    undo_standardizing(flags, dataloaders)
    cluster_jets(dataloaders)
    cluster_breit(flags, dataloaders)
    
    # Clean up memory
    del dataloaders[flags.file].part, dataloaders[flags.file].mask
    gc.collect()
    
    # Extract results
    dataset = dataloaders[flags.file]
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
    
    # Clean up dataloaders
    del dataloaders
    gc.collect()
    
    return results


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
        
        # Process batch - Dataset will load up to batch_end events total,
        # but hvd will split them across ranks
        batch_results = process_batch(flags, batch_end, weights_dict, opt, batch_idx)
        
        # Each rank writes its portion
        # Create unique filename for this batch and rank
        base_name = flags.file.replace("prep.h5", "")
        output_file_name = f"{base_name}{replace_string}_batch{batch_idx:04d}_rank{hvd.rank():02d}.h5"
        output_path = os.path.join(flags.data_folder, output_file_name)
        
        if hvd.rank() == 0 or True:  # All ranks write their own files
            output_files.append(output_file_name)
            
            with h5.File(output_path, 'w') as fh5:
                # Write metadata
                fh5.attrs['batch_idx'] = batch_idx
                fh5.attrs['global_batch_start'] = batch_start
                fh5.attrs['global_batch_end'] = batch_end
                fh5.attrs['hvd_rank'] = hvd.rank()
                fh5.attrs['hvd_size'] = hvd.size()
                
                # Write weights
                if "data" not in flags.file and len(weights_dict) > 0:
                    if flags.bootstrap:
                        for i in range(1, flags.nboot):
                            if str(i) in batch_results['weights']:
                                dset = fh5.create_dataset(
                                    f"weights{i}", 
                                    data=batch_results['weights'][str(i)]
                                )
                    else:
                        if flags.file in batch_results['weights']:
                            dset = fh5.create_dataset(
                                "weights", 
                                data=batch_results['weights'][flags.file]
                            )
                    
                    if 'mc_weights' in batch_results:
                        dset = fh5.create_dataset(
                            "mc_weights", 
                            data=batch_results['mc_weights']
                        )
                    
                    if "closure" in batch_results['weights']:
                        dset = fh5.create_dataset(
                            "closure_weights", 
                            data=batch_results['weights']["closure"]
                        )
                
                # Write jet features
                dset = fh5.create_dataset("jet_pt", data=batch_results['jet_pt'])
                dset = fh5.create_dataset("jet_breit_pt", data=batch_results['jet_breit_pt'])
                dset = fh5.create_dataset("deltaphi", data=batch_results['deltaphi'])
                dset = fh5.create_dataset("jet_tau10", data=batch_results['jet_tau10'])
                dset = fh5.create_dataset("zjet", data=batch_results['zjet'])
                dset = fh5.create_dataset("zjet_breit", data=batch_results['zjet_breit'])
            
            if flags.verbose or hvd.rank() == 0:
                print(f"Rank {hvd.rank()} saved batch {batch_idx + 1} to: {output_file_name}")
        
        # Clean up batch results
        del batch_results, weights_dict
        gc.collect()
    
    if hvd.rank() == 0:
        print(f"\n{'='*60}")
        print(f"All batches processed successfully!")
        print(f"Each rank created {num_batches} output files")
        print(f"Total files created: {num_batches * hvd.size()}")
        print(f"Example files from rank 0:")
        for f in output_files[:5]:
            print(f"  - {f}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()