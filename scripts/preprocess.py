import numpy as np
import argparse
import h5py as h5
import os
import re
import glob
import random
from concurrent.futures import ProcessPoolExecutor
from merge_h5 import merge_h5_files


def preprocess(data):
    p, e = data
    # return (p,e)
    mask = p[:, :, 0] != 0

    # use log(pt/Q), delta_eta, delta_phi
    log_pt_rel = np.ma.log(
        np.ma.divide(p[:, :, 0], np.sqrt(e[:, None, 0])).filled(0)
    ).filled(0)
    log_pt = np.ma.log(p[:, :, 0]).filled(0)

    log_e_rel = np.ma.log(
        np.ma.divide(p[:, :, 0] * np.cosh(p[:, :, 1]), np.sqrt(e[:, None, 0])).filled(0)
    ).filled(0)
    log_e = np.ma.log(p[:, :, 0] * np.cosh(p[:, :, 1])).filled(0)

    delta_eta = p[:, :, 1] - np.ma.arctanh(
        e[:, None, 4]
        / np.sqrt(e[:, None, 2] ** 2 + e[:, None, 3] ** 2 + e[:, None, 4] ** 2)
    ).filled(0)
    delta_phi = p[:, :, 2] - np.pi - np.arctan2(e[:, None, 3], e[:, None, 2])
    delta_phi[delta_phi > np.pi] -= 2 * np.pi
    delta_phi[delta_phi < -np.pi] += 2 * np.pi
    delta_r = np.hypot(delta_eta, delta_phi + np.pi)
    new_p = (
        np.stack(
            [delta_eta, delta_phi, log_pt, log_pt_rel, log_e_rel, log_e, delta_r], -1
        )
        * mask[:, :, None]
    )

    log_Q2 = np.ma.log(e[:, 0]).filled(0)
    new_e = np.stack(
        [
            log_Q2,
            e[:, 1],
            np.ma.divide(np.sqrt(e[:, 2] ** 2 + e[:, 3] ** 2), np.sqrt(e[:, 0])).filled(
                0
            ),
            np.ma.arctanh(
                e[:, 4] / np.sqrt(e[:, 2] ** 2 + e[:, 3] ** 2 + e[:, 4] ** 2)
            ).filled(0),
            np.arctan2(e[:, 3], e[:, 2]),
        ],
        -1,
    )

    return new_e, new_p * mask[:, :, None]


def preprocess_one_file(input_file, output_file, is_mc):
    with h5.File(input_file, "r") as fh5:
        reco_p = fh5["reco_particle_features"][:].astype(np.float32)
        reco_e = fh5["reco_event_features"][:].astype(np.float32)
        if is_mc:
            gen_p = fh5["gen_particle_features"][:].astype(np.float32)
            gen_e = fh5["gen_event_features"][:].astype(np.float32)

    weight = reco_e[:, -2]
    pass_reco = reco_e[:, -1]
    print(f"Running {weight.shape[0]} Reco events from {input_file}")
    reco_e, reco_p = preprocess((reco_p, reco_e[:, :-2]))
    reco_e = np.concatenate([reco_e, weight[:, None], pass_reco[:, None]], -1)

    if is_mc:
        print(f"Running Gen events from {input_file}")
        pass_gen = gen_e[:, -1]
        gen_e, gen_p = preprocess((gen_p, gen_e[:, :-1]))
        gen_e = np.concatenate([gen_e, pass_gen[:, None]], -1)

    with h5.File(output_file, "w") as fh5:
        fh5.create_dataset("reco_particle_features", data=reco_p)
        fh5.create_dataset("reco_event_features", data=reco_e)
        if is_mc:
            fh5.create_dataset("gen_particle_features", data=gen_p)
            fh5.create_dataset("gen_event_features", data=gen_e)

    print(f"Finished {input_file} -> {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Folder containing data and MC files",
    )
    parser.add_argument(
        "--file_name",
        default="data_Eplus0607",
        help="Prefix of the prepare_data.py outputs, e.g. Rapgap_Eplus0607. "
        "Processes every <file_name>_<N>.h5 in data_folder",
    )
    parser.add_argument(
        "--no-combine",
        dest="combine",
        action="store_false",
        help="Do not merge the per-file outputs into a single <file_name>_prep.h5 at the end",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the per-file <file_name>_<N>_prep.h5 outputs. Without this they are deleted as they are merged",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of files to process in parallel",
    )
    parser.add_argument(
        "--random_order",
        action="store_true",
        help="Merge the per-file outputs in a random order instead of sorted by file number",
    )
    flags = parser.parse_args()
    if not flags.combine and not flags.keep_intermediate:
        raise Exception(
            "ERROR: --no-combine without --keep-intermediate would not save any output"
        )

    is_mc = False if "data" in flags.file_name else True
    prefix_path = os.path.join(flags.data_folder, flags.file_name)

    # Only take <file_name>_<N>.h5, so merged, already preprocessed, or other samples
    # sharing the prefix (e.g. _sys0, _<N>_prep) are not picked up
    # If only the merged file exists in the directory, it will be picked up in the next if statement
    pattern = re.compile(rf"^{re.escape(flags.file_name)}_(\d+)\.h5$")
    file_numbers = {}
    for f in glob.glob(prefix_path + "_*.h5"):
        m = pattern.match(os.path.basename(f))
        if m:
            file_numbers[f] = m.group(1)
    file_list = sorted(file_numbers, key=lambda f: int(file_numbers[f]))
    if len(file_list) == 0:
        # prepare_data.py deletes its per-file outputs after merging unless run with
        # --keep-intermediate, so fall back to the merged <file_name>.h5
        if os.path.exists(prefix_path + ".h5"):
            print(
                f"No {prefix_path}_<N>.h5 files found, processing the merged {prefix_path}.h5"
            )
            preprocess_one_file(prefix_path + ".h5", prefix_path + "_prep.h5", is_mc)
        else:
            raise RuntimeError(
                f"No files found matching {prefix_path}_<N>.h5 or {prefix_path}.h5"
            )
    else:
        print(f"Found {len(file_list)} files")
        # Preprocess num_processes files at a time. The results come back in the same order as file_list
        with ProcessPoolExecutor(max_workers=flags.num_processes) as executor:
            futures = [
                executor.submit(
                    preprocess_one_file,
                    f,
                    f"{prefix_path}_{file_numbers[f]}_prep.h5",
                    is_mc,
                )
                for f in file_list
            ]
            output_files = [future.result() for future in futures]

        if flags.combine:
            if flags.random_order:
                random.shuffle(output_files)
            merge_h5_files(
                input_files=output_files,
                output_file=prefix_path + "_prep.h5",
                is_mc=is_mc,
                delete_inputs=not flags.keep_intermediate,
            )
