import h5py as h5
import os
import re
import glob
import argparse


def merge_h5_files(input_files, output_file, is_mc=True, delete_inputs=False):
    # delete_inputs: remove each input file once it has been copied, so the inputs and the
    # full merged output never take up disk space at the same time
    if len(input_files) == 0:
        raise RuntimeError("No input files found")

    if is_mc:
        dataset_names = [
            "reco_particle_features",
            "reco_event_features",
            "gen_particle_features",
            "gen_event_features",
        ]
    else:
        dataset_names = [
            "reco_particle_features",
            "reco_event_features",
        ]

    # Check that all files are compatible and count the total number of events
    print("Inspecting input files")
    file_info = []
    total_events = 0
    for filename in input_files:
        with h5.File(filename, "r") as fin:
            n_events = fin["reco_event_features"].shape[0]
            file_info.append((filename, n_events))
            total_events += n_events

            if len(file_info) == 1:
                shapes = {name: fin[name].shape[1:] for name in dataset_names}
                dtypes = {name: fin[name].dtype for name in dataset_names}
            else:
                for name in dataset_names:
                    if fin[name].shape[1:] != shapes[name]:
                        raise ValueError(
                            f"Shape mismatch for {name} in {filename}: "
                            f"{fin[name].shape[1:]} vs expected {shapes[name]}"
                        )
                    if fin[name].dtype != dtypes[name]:
                        raise ValueError(
                            f"dtype mismatch for {name} in {filename}: "
                            f"{fin[name].dtype} vs expected {dtypes[name]}"
                        )
                    if fin[name].shape[0] != n_events:
                        raise ValueError(
                            f"{name} in {filename} has {fin[name].shape[0]} events, "
                            f"expected {n_events}"
                        )

    print(f"Total events: {total_events}")

    # Preallocate the output and copy one input file at a time, so only one file is in memory
    with h5.File(output_file, "w") as fout:
        for name in dataset_names:
            fout.create_dataset(
                name,
                shape=(total_events,) + shapes[name],
                dtype=dtypes[name],
            )

        output_start = 0
        for filename, n_events in file_info:
            output_stop = output_start + n_events
            print(f"Adding {filename} -> rows {output_start}:{output_stop}")

            with h5.File(filename, "r") as fin:
                for name in dataset_names:
                    fout[name][output_start:output_stop] = fin[name][:]

            if delete_inputs:
                fout.flush()
                os.remove(filename)
                print(f"Deleted {filename}")

            output_start = output_stop

    print(f"Done. Wrote {total_events} events to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Folder containing the per-file h5 outputs",
    )
    parser.add_argument(
        "--data_output",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Output folder",
    )
    parser.add_argument(
        "--sample",
        default="Rapgap_Eplus0607_prep",
        help="Prefix of the per-file outputs to merge, e.g. Rapgap_Eplus0607 or Rapgap_Eplus0607_prep",
    )
    flags = parser.parse_args()

    output_file = os.path.join(flags.data_output, flags.sample + ".h5")

    # Only take <sample>_<N>.h5, so other samples sharing the prefix (e.g. _prep, _sys0) are not merged in
    pattern = re.compile(rf"^{re.escape(flags.sample)}_(\d+)\.h5$")
    file_list = sorted(
        [
            f
            for f in glob.glob(os.path.join(flags.data_input, flags.sample) + "_*.h5")
            if pattern.match(os.path.basename(f))
            and os.path.abspath(f) != os.path.abspath(output_file)
        ],
        key=lambda f: int(pattern.match(os.path.basename(f)).group(1)),
    )
    print(file_list)

    is_mc = False if "data" in flags.sample else True
    merge_h5_files(input_files=file_list, output_file=output_file, is_mc=is_mc)
