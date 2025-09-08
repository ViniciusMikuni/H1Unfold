import numpy as np

import os
import h5py as h5
import gc


class Dataset:
    def __init__(
        self,
        file_names,
        base_path,
        rank=0,
        size=1,
        is_mc=False,
        use_reco=True,
        nmax=None,
        norm=None,
        pass_fiducial=False,
        pass_reco=False,
        preprocess=True,
    ):
        self.rank = rank
        self.size = size
        self.base_path = base_path
        self.is_mc = is_mc
        self.use_reco = use_reco
        self.nmax = nmax
        self.preprocess = preprocess

        assert is_mc or use_reco, "ERROR: Dataset must have reco, mc, or both"
        # Preprocessing parameters

        self.mean_part = [
            0.0,
            0.0,
            -0.761949,
            -3.663438,
            -2.8690917,
            0.03239748,
            3.9436243,
        ]
        self.std_part = [
            1.0,
            1.0,
            1.0133458,
            1.03931,
            1.0040112,
            0.98908925,
            1.2256976,
        ]

        self.mean_event = [6.4188385, 0.3331013, 0.8914633, -0.8352072, -0.07296985]
        self.std_event = [0.97656405, 0.1895471, 0.14934653, 0.4191545, 1.734126]

        self.prepare_dataset(file_names, pass_fiducial, pass_reco)
        self.normalize_weights(self.nmax if norm is None else norm)

    def normalize_weights(self, norm):
        # print("Total number of reco events {}".format(self.num_pass_reco))
        if self.use_reco:
            self.weight = (norm * self.weight / self.num_pass_reco).astype(np.float32)
        else:
            self.weight = (norm * self.weight / self.num_pass_gen).astype(np.float32)

    def standardize(self, new_p, new_e, mask):
        mask = new_p[:, :, 2] != 0
        p = mask[:, :, None] * (new_p - self.mean_part) / self.std_part
        e = (new_e - self.mean_event) / self.std_event
        return p, e, mask

    def revert_standardize(self, new_p, new_e, mask):
        p = new_p * self.std_part + self.mean_part
        e = new_e * self.std_event + self.mean_event
        return p * mask[:, :, None], e

    def concatenate(self, data_list):
        data_part1 = [item[0] for item in data_list]  # Extracting all (M, P, Q) arrays
        data_part2 = [item[1] for item in data_list]  # Extracting all (M, F) arrays

        # Concatenate along the first axis (N * M)
        concatenated_part1 = np.concatenate(data_part1, axis=0)
        concatenated_part2 = np.concatenate(data_part2, axis=0)
        mask = concatenated_part1[:, :, 2] != 0
        del data_list
        gc.collect()
        return concatenated_part1, concatenated_part2, mask

    def prepare_dataset(self, file_names, pass_fiducial, pass_reco):
        """Load h5 files containing the data. The structure of the h5 file should be
        reco_particle_features: p_pt,p_eta,p_phi,p_charge (B,N,4)
        reco_event_features   : Q2, e_px, e_py, e_pz, wgt, pass_reco (B,6)
        if MC should also contain
        gen_particle_features : p_pt,p_eta,p_phi,p_charge (B,N,4)
        gen_event_features    : Q2, e_px, e_py, e_pz, pass_gen (B,5)

        """
        self.num_pass_reco = 0
        self.num_pass_gen = 0
        self.weight = []
        self.pass_reco = []
        self.pass_gen = []
        reco = []
        gen = []
        for ifile, f in enumerate(file_names):
            if self.rank == 0:
                print("Loading file {}".format(f))
            # Determine the total number of event passing reco for normalization of the weights

            if self.nmax is None:
                self.nmax = h5.File(os.path.join(self.base_path, f), "r")[
                    "reco_event_features"
                ].shape[0]

            # Sum of weighted events for collisions passing the reco cuts
            self.num_pass_reco += np.sum(
                h5.File(os.path.join(self.base_path, f), "r")["reco_event_features"][
                    : self.nmax, -2
                ][
                    h5.File(os.path.join(self.base_path, f), "r")[
                        "reco_event_features"
                    ][: self.nmax, -1]
                    == 1
                ]
            )

            per_rank = (self.nmax + self.size - 1) // self.size  # ceiling division
            start = self.rank * per_rank
            end = min(start + per_rank, self.nmax)

            reco_p = h5.File(os.path.join(self.base_path, f), "r")[
                "reco_particle_features"
            ][start:end].astype(np.float32)
            reco_e = h5.File(os.path.join(self.base_path, f), "r")[
                "reco_event_features"
            ][start:end].astype(np.float32)

            self.weight.append(reco_e[:, -2].astype(np.float32))
            if not self.use_reco:
                reco_e = np.nan_to_num(
                    reco_e
                )  # reco_e has a nan entry (reco_e[:,3]) when there's not proper reco info
                self.pass_reco.append(np.full_like(reco_e[:, -1], False, dtype=bool))
            else:
                self.pass_reco.append(reco_e[:, -1] == 1)

            if pass_reco:
                mask_reco = self.pass_reco[-1]
            else:
                mask_reco = np.ones_like(self.pass_reco[-1])

            if self.is_mc:
                gen_p = h5.File(os.path.join(self.base_path, f), "r")[
                    "gen_particle_features"
                ][start:end].astype(np.float32)
                gen_e = h5.File(os.path.join(self.base_path, f), "r")[
                    "gen_event_features"
                ][start:end].astype(np.float32)
                self.num_pass_gen += np.sum(
                    h5.File(os.path.join(self.base_path, f), "r")[
                        "reco_event_features"
                    ][: self.nmax, -2][
                        h5.File(os.path.join(self.base_path, f), "r")[
                            "gen_event_features"
                        ][: self.nmax, -1]
                        == 1
                    ]
                )

                self.pass_gen.append(gen_e[:, -1] == 1)

                if pass_fiducial:
                    mask_fid = self.pass_gen[-1]
                else:
                    mask_fid = np.ones_like(self.pass_gen[-1])
                gen_p = gen_p[mask_fid * mask_reco]
                gen_e = gen_e[mask_fid * mask_reco]
                self.pass_gen[-1] = self.pass_gen[-1][mask_fid * mask_reco]

                gen.append((gen_p, gen_e[:, :-1]))
            else:
                self.pass_gen = None
                mask_fid = mask_reco

            reco_p = reco_p[mask_fid * mask_reco]
            reco_e = reco_e[mask_fid * mask_reco]
            self.weight[-1] = self.weight[-1][mask_fid * mask_reco]
            self.pass_reco[-1] = self.pass_reco[-1][mask_fid * mask_reco]

            reco.append((reco_p, reco_e[:, :-2]))

        self.weight = np.concatenate(self.weight)
        self.pass_reco = np.concatenate(self.pass_reco)

        if self.preprocess:
            self.reco = self.standardize(*self.concatenate(reco))
        else:
            self.reco = self.concatenate(reco)

        del reco
        gc.collect()
        assert not np.any(np.isnan(self.reco[0])), "ERROR: NAN in particle dataset"
        assert not np.any(np.isnan(self.reco[1])), "ERROR: NAN in event dataset"

        # self.reco =  self.return_dataset(reco)
        if self.is_mc:
            self.pass_gen = np.concatenate(self.pass_gen)
            if self.preprocess:
                self.gen = self.standardize(*self.concatenate(gen))
            else:
                self.gen = self.concatenate(gen)
            del gen
            gc.collect()
            assert not np.any(np.isnan(self.gen[0])), "ERROR: NAN in particle dataset"
            assert not np.any(np.isnan(self.gen[1])), "ERROR: NAN in event dataset"
        else:
            self.gen = None
