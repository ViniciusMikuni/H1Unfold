import numpy as np

import os
import h5py as h5
import fnmatch


def convert_to_np(
    file_list,
    base_path,
    name,
    is_data=False,
    max_part=191,  # Overall maximum
    max_nonzero=132,  # Maximum number after applying the per-particle selection
    nevts=30000000,
    gen_only=False,
):
    reco_dict = {
        "event_features": [],
        "particle_features": [],
    }
    gen_dict = {
        "event_features": [],
        "particle_features": [],
    }
    var_list = ["Q2", "y", "e_px", "e_py", "e_pz", "wgt"]
    mask_list = [
        "y",
        "ptmiss",
        "Empz",
    ]  # Variables used only to determine the selection but not used during unfolding
    particle_list = ["part_pt", "part_eta", "part_phi", "part_charge"]

    for ifile, f in enumerate(file_list):
        try:
            # print("evaluating file {}".format(f))
            tmp_file = uproot.open(os.path.join(base_path, f))[
                "{}/minitree".format(name)
            ]
            # print("loaded file")
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            continue

        reco_dict["event_features"].append(
            np.stack([tmp_file[feat].array()[:nevts] for feat in var_list], -1)
        )
        reco_dict["particle_features"].append(
            np.stack(
                [
                    tmp_file[feat].array()[:nevts].pad(max_part).fillna(0).regular()
                    for feat in particle_list
                ],
                -1,
            )
        )
        mask_reco = np.stack([tmp_file[feat].array()[:nevts] for feat in mask_list], -1)
        if "Data" not in name:
            gen_dict["event_features"].append(
                np.stack(
                    [
                        tmp_file["gen_" + feat].array()[:nevts]
                        for feat in var_list
                        if "wgt" not in feat
                    ],
                    -1,
                )
            )
            mask_gen = tmp_file["gen_y"].array()[:nevts]
            # Keep only Q2>100 GeV^2

            mask_evt = gen_dict["event_features"][ifile][:, 0] > 100
            reco_dict["particle_features"][ifile] = reco_dict["particle_features"][
                ifile
            ][mask_evt]
            reco_dict["event_features"][ifile] = reco_dict["event_features"][ifile][
                mask_evt
            ]

            gen_dict["particle_features"].append(
                np.stack(
                    [
                        tmp_file["gen_" + feat]
                        .array()[:nevts]
                        .pad(max_part)
                        .fillna(0)
                        .regular()
                        for feat in particle_list
                    ],
                    -1,
                )
            )

            # Set charge to 0 for gen particles outside the tracker acceptance
            gen_dict["particle_features"][-1][:, :, -1] *= (
                np.abs(gen_dict["particle_features"][-1][:, :, 1]) < 2.0
            )

            # print("Removing events")
            # Remove events not passing Q2> 100
            gen_dict["particle_features"][ifile] = gen_dict["particle_features"][ifile][
                mask_evt
            ]
            gen_dict["event_features"][ifile] = gen_dict["event_features"][ifile][
                mask_evt
            ]
            mask_reco = mask_reco[mask_evt]
            mask_gen = mask_gen[mask_evt]

        print("Number of events: {}".format(mask_reco.shape[0]))
        # 0.08 < y < 0.7, ptmiss < 10, 45 < empz < 65 and Q2 > 150
        pass_reco = (
            (mask_reco[:, 0] > 0.08)
            & (mask_reco[:, 0] < 0.7)
            & (mask_reco[:, 1] < 10.0)
            & (mask_reco[:, 2] > 45.0)
            & (mask_reco[:, 2] < 65)
            & (reco_dict["event_features"][ifile][:, 0] > 150)
        )
        reco_dict["event_features"][ifile] = np.concatenate(
            (reco_dict["event_features"][ifile], pass_reco[:, None]), -1
        )
        del mask_reco, pass_reco
        # Particle dataset

        # part pT > 0.1 GeV, -1.5 < part eta < 2.75
        mask_part = (
            (reco_dict["particle_features"][ifile][:, :, 0] > 0.1)
            & (reco_dict["particle_features"][ifile][:, :, 1] > -1.5)
            & (reco_dict["particle_features"][ifile][:, :, 1] < 2.75)
        )
        reco_dict["particle_features"][ifile] = (
            reco_dict["particle_features"][ifile] * mask_part[:, :, None]
        )
        if gen_only:
            mask_evt = np.ones(
                reco_dict["particle_features"][ifile].shape[0], dtype=bool
            )
        else:
            mask_evt = np.sum(reco_dict["particle_features"][ifile][:, :, 0], 1) > 0

        del mask_part
        if "Data" not in name:
            print("Adding Gen info")

            # 0.2 < y < 0.7 and Q2 > 150
            pass_gen = (
                (mask_gen > 0.2)
                & (mask_gen < 0.7)
                & (gen_dict["event_features"][ifile][:, 0] > 150)
            )
            gen_dict["event_features"][ifile] = np.concatenate(
                (gen_dict["event_features"][ifile], pass_gen[:, None]), -1
            )
            # part pT > 0.1 GeV, -1.5 < part eta < 2.75

            mask_part = (
                (gen_dict["particle_features"][ifile][:, :, 0] > 0.1)
                & (gen_dict["particle_features"][ifile][:, :, 1] > -1.5)
                & (gen_dict["particle_features"][ifile][:, :, 1] < 2.75)
            )
            gen_dict["particle_features"][ifile] = (
                gen_dict["particle_features"][ifile] * mask_part[:, :, None]
            )

            # Reject events in case there's no particle left
            mask_evt_gen = np.sum(gen_dict["particle_features"][ifile][:, :, 0], 1) > 0

            mask_evt *= mask_evt_gen
            gen_dict["particle_features"][ifile] = gen_dict["particle_features"][ifile][
                mask_evt
            ]
            gen_dict["event_features"][ifile] = gen_dict["event_features"][ifile][
                mask_evt
            ]

            del mask_gen, pass_gen

        print("Rejecting {}".format(1.0 - 1.0 * np.sum(mask_evt) / mask_evt.shape[0]))
        reco_dict["particle_features"][ifile] = reco_dict["particle_features"][ifile][
            mask_evt
        ]
        reco_dict["event_features"][ifile] = reco_dict["event_features"][ifile][
            mask_evt
        ]

    reco_dict["event_features"] = np.concatenate(reco_dict["event_features"])
    reco_dict["particle_features"] = np.concatenate(reco_dict["particle_features"])

    # Make sure reco particles that do not pass reco cuts are indeed zero padded
    reco_dict["particle_features"] *= reco_dict["event_features"][:, -1, None, None]
    order = np.argsort(-reco_dict["particle_features"][:, :, 0], 1)
    reco_dict["particle_features"] = np.take_along_axis(
        reco_dict["particle_features"], order[:, :, None], 1
    )
    max_nonzero_reco = np.max(np.sum(reco_dict["particle_features"][:, :, 0] > 0, 1))
    reco_dict["particle_features"] = reco_dict["particle_features"][:, :max_nonzero]

    print("Maximum reco particle multiplicity", max_nonzero_reco)
    if "Data" not in name:
        gen_dict["event_features"] = np.concatenate(gen_dict["event_features"])
        gen_dict["particle_features"] = np.concatenate(gen_dict["particle_features"])

        order = np.argsort(-gen_dict["particle_features"][:, :, 0], 1)
        gen_dict["particle_features"] = np.take_along_axis(
            gen_dict["particle_features"], order[:, :, None], 1
        )
        max_nonzero_gen = np.max(np.sum(gen_dict["particle_features"][:, :, 0] > 0, 1))
        gen_dict["particle_features"] = gen_dict["particle_features"][:, :max_nonzero]

        print("Maximum gen particle multiplicity", max_nonzero_gen)
    del tmp_file
    return reco_dict, gen_dict


def find_files_with_string(directory, string):
    matching_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, string):
            matching_files.append(filename)
    return matching_files


def create_toy(data_output):
    nevts = 1000000
    nfeat = 4
    npart = 100

    mean1 = 1.0
    std1 = 1.0
    mean2 = 0.8
    std2 = 1.0

    std_smear = 0.1

    def create_gen(nevts, nfeat, npart, mean, std):
        evt = np.random.normal(
            size=(nevts, nfeat),
            loc=mean * np.ones((nevts, nfeat)),
            scale=std * np.ones((nevts, nfeat)),
        )
        part = np.random.normal(
            size=(nevts, npart, nfeat),
            loc=mean * np.ones((nevts, npart, nfeat)),
            scale=std * np.ones((nevts, npart, nfeat)),
        )
        return evt, part

    gen1_evt, gen1_part = create_gen(nevts, nfeat, npart, mean1, std1)
    gen2_evt, gen2_part = create_gen(nevts, nfeat, npart, mean2, std2)

    def smear(sample, std):
        return std * np.random.normal(size=sample.shape) + sample

    reco1_evt = smear(gen1_evt, std_smear)
    reco1_part = smear(gen1_part, std_smear)

    reco2_evt = smear(gen2_evt, std_smear)
    reco2_part = smear(gen2_part, std_smear)

    # Add mock weights and pass reco flags
    pass_reco1 = np.random.randint(2, size=(nevts, 1))
    reco1_evt *= pass_reco1
    pass_reco2 = np.random.randint(2, size=(nevts, 1))
    reco2_evt *= pass_reco2
    weights = np.ones((nevts, 1))
    reco1_evt = np.concatenate((reco1_evt, weights, pass_reco1), -1)
    reco2_evt = np.concatenate((reco2_evt, weights, pass_reco2), -1)
    gen1_evt = np.concatenate((gen1_evt, weights), -1)
    gen2_evt = np.concatenate((gen2_evt, weights), -1)

    with h5.File(os.path.join(data_output, "toy1.h5"), "w") as fh5:
        fh5.create_dataset("reco_particle_features", data=reco1_part)
        fh5.create_dataset("reco_event_features", data=reco1_evt)
        fh5.create_dataset("gen_particle_features", data=gen1_part)
        fh5.create_dataset("gen_event_features", data=gen1_evt)

    with h5.File(os.path.join(data_output, "toy2.h5"), "w") as fh5:
        fh5.create_dataset("reco_particle_features", data=reco2_part)
        fh5.create_dataset("reco_event_features", data=reco2_evt)
        fh5.create_dataset("gen_particle_features", data=gen2_part)
        fh5.create_dataset("gen_event_features", data=gen2_evt)


if __name__ == "__main__":
    import argparse

    # import uproot
    import uproot3 as uproot

    # Convert root files to h5 inputs that are easier to load when training OmniFold

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-input",
        default="/global/cfs/cdirs/m3246/H1/root/",
        help="Folder containing data and MC files in the root format",
    )
    parser.add_argument(
        "--data-output",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Output folder containing data and MC files",
    )

    parser.add_argument(
        "--sample",
        default="dataEp",
        help="Sample to process. Options are: DjangohEp, DjangohEm,RapgapEp,RapgapEm,data",
    )
    flags = parser.parse_args()

    # create_toy(flags.data_output)

    if flags.sample == "dataEp":
        print("Processing Data")
        file_list = [
            "out_ep0607/Data_Eplus0607.root",
            #'out_em06/Data_Eminus06_0.nominal.root',
            #'out_em06/Data_Eminus06_1.nominal.root'
        ]
        data, _ = convert_to_np(file_list, flags.data_input, name="Data")
        with h5.File(os.path.join(flags.data_output, "data_Eplus0607.h5"), "w") as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=data["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=data["event_features"]
            )

    elif flags.sample == "dataEm":
        print("Processing Data")
        file_list = [  #'out_ep0607/Data_Eplus0607.root',
            "out_em06/Data_Eminus06_0.nominal.root",
            "out_em06/Data_Eminus06_1.nominal.root",
        ]
        data, _ = convert_to_np(file_list, flags.data_input, name="Data")
        with h5.File(os.path.join(flags.data_output, "data_Eminus06.h5"), "w") as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=data["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=data["event_features"]
            )

    elif flags.sample == "RapgapEp":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/out_ep0607", "Rapgap_Eplus0607_"
        )
        # file_list = ["/global/cfs/cdirs/m3246/vmikuni/H1v2/root/out_ep0607/Rapgap_Eplus0607_4.nominal.root"]
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "DjangohEp":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/out_ep0607", "Django_Eplus0607_"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEm":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/out_em06", "Django_Eminus06_"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEm":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/out_em06", "Rapgap_Eminus06_"
        )
        ###file_list = ['Rapgap_Eminus06_122.nominal.root'] #quick testing
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_em06", name="Rapgap"
        )
        with h5.File(os.path.join(flags.data_output, "Rapgap_Eminus06.h5"), "w") as fh5:
            ###with h5.File(os.path.join(flags.data_output,"test_sim.h5"),'w') as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEp_sys0":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Rapgap_Eplus0607_*sys_0.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_sys0.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEp_sys1":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Rapgap_Eplus0607_*sys_1.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_sys1.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEp_sys5":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Rapgap_Eplus0607_*sys_5.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_sys5.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "RapgapEp_sys7":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Rapgap_Eplus0607_*sys_7.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_sys7.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "RapgapEp_sys11":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Rapgap_Eplus0607_*sys_11.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_sys11.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEp_sys0":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Django_Eplus0607_*sys_0.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_sys0.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEp_sys1":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Django_Eplus0607_*sys_1.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_sys1.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEp_sys5":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Django_Eplus0607_*sys_5.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_sys5.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "DjangohEp_sys7":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Django_Eplus0607_*sys_7.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_sys7.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "DjangohEp_sys11":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_ep0607", "Django_Eplus0607_*sys_11.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_ep0607", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_sys11.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEm_sys0":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Rapgap_Eminus06_*sys_0.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eminus06_sys0.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEm_sys1":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Rapgap_Eminus06_*sys_1.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eminus06_sys1.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEm_sys5":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Rapgap_Eminus06_*sys_5.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eminus06_sys5.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "RapgapEm_sys7":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Rapgap_Eminus06_*sys_7.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eminus06_sys7.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "RapgapEm_sys11":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Rapgap_Eminus06_*sys_11.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Rapgap"
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eminus06_sys11.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEm_sys0":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Django_Eminus06_*sys_0.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06_sys0.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEm_sys1":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Django_Eminus06_*sys_1.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06_sys1.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEm_sys5":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Django_Eminus06_*sys_5.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06_sys5.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "DjangohEm_sys7":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Django_Eminus06_*sys_7.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06_sys7.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])
    elif flags.sample == "DjangohEm_sys11":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/sys/out_em06", "Django_Eminus06_*sys_11.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/sys/out_em06", name="Django"
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eminus06_sys11.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "RapgapEp_NoRad":
        print("Processing Rapgap")
        file_list = find_files_with_string(
            flags.data_input + "/out_ep_norad", "RaNorad_Eplus0607*.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_ep_norad", name="RaNorad", gen_only=True
        )
        with h5.File(
            os.path.join(flags.data_output, "Rapgap_Eplus0607_NoRad.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    elif flags.sample == "DjangohEp_NoRad":
        print("Processing Djangoh")
        file_list = find_files_with_string(
            flags.data_input + "/out_ep_norad", "DjNorad_Eplus0607*.root"
        )
        reco, gen = convert_to_np(
            file_list, flags.data_input + "/out_ep_norad", name="DjNorad", gen_only=True
        )
        with h5.File(
            os.path.join(flags.data_output, "Djangoh_Eplus0607_NoRad.h5"), "w"
        ) as fh5:
            fh5.create_dataset(
                "reco_particle_features", data=reco["particle_features"]
            )
            fh5.create_dataset(
                "reco_event_features", data=reco["event_features"]
            )
            fh5.create_dataset(
                "gen_particle_features", data=gen["particle_features"]
            )
            fh5.create_dataset("gen_event_features", data=gen["event_features"])

    else:
        raise Exception("ERROR: Sample not recognized")
