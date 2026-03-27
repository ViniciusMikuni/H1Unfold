"""
plot_from_batches.py

Loads Rapgap and Djangoh batch h5 files one at a time and accumulates weighted
histograms to avoid holding all events in memory simultaneously.

Expected h5 keys per batch file:
  jet_pt, jet_breit_pt, deltaphi, jet_tau10, zjet, zjet_breit  -- shape (n_events, n_jets)
  mc_weights                                                     -- shape (n_events,)
  weights_nominal                                                -- shape (n_events,)  [nominal unfolded]
  weights1 .. weightsN                                           -- shape (n_events,)  [bootstrap]
"""

import argparse
import glob
import os
import gc

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

import utils
import options
from utils import ObservableInfo

utils.SetStyle()

VAR_NAMES = [
    "jet_pt",
    "jet_breit_pt",
    "deltaphi",
    "jet_tau10",
    "zjet",
    "zjet_breit",
]

SYS_LIST_DEFAULT = ["sys0", "sys1", "sys5", "sys7", "sys11"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot unfolded observables from batch h5 files without loading all data at once."
    )
    parser.add_argument(
        "--data_folder",
        default="/global/cfs/cdirs/m3246/H1/h5",
        help="Directory containing the batch h5 files",
    )
    parser.add_argument(
        "--plot_folder",
        default="../plots",
        help="Directory to write output PDF plots",
    )
    parser.add_argument(
        "--niter", type=int, default=4, help="OmniFold iteration used in file names"
    )
    parser.add_argument(
        "--period", default="Eplus0607", help="Data-taking period string in file names"
    )
    parser.add_argument(
        "--nboot",
        type=int,
        default=49,
        help="Number of bootstrap weight sets (weights1..weightsN) to use for stat uncertainty",
    )
    parser.add_argument(
        "--nominal_weight",
        default="weights_nominal",
        help="Key to use as the nominal unfolded weight (default: weights_nominal)",
    )
    parser.add_argument(
        "--sys",
        action="store_true",
        default=False,
        help="Load systematic variation batch files and compute systematic uncertainties",
    )
    parser.add_argument(
        "--sys_list",
        nargs="+",
        default=SYS_LIST_DEFAULT,
        help="List of systematic labels to include (default: sys0 sys1 sys5 sys7 sys11)",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        default=False,
        help="Closure mode: plot closure_weights result vs RAPGAP truth instead of data",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Histogram accumulation helpers
# ---------------------------------------------------------------------------

def _valid_mask(jet_pt):
    """Return boolean mask of jets with pt > 0; shape matches jet_pt."""
    return jet_pt > 0


def accumulate_var_histograms(batch_files, var, binning, nominal_wkey, nboot, verbose):
    """
    Iterate over batch files and accumulate histogram counts without keeping
    raw arrays in memory across iterations.

    Parameters
    ----------
    batch_files : list of str
    var : str
        Observable key in the h5 file.
    binning : 1-D array
    nominal_wkey : str
        Dataset key for the nominal unfolded weight (e.g. 'weights_nominal').
    nboot : int
        Number of bootstrap weight sets (weights1..weightsN). Pass 0 to skip.
    verbose : bool

    Returns
    -------
    counts_mc : (n_bins,)      mc_weights only
    counts_unf : (n_bins,)     mc_weights * nominal_wkey
    counts_boot : (nboot, n_bins)  mc_weights * weightsI, I = 1..nboot
    counts_closure : (n_bins,) mc_weights * closure_weights (zeros if key absent)
    """
    n_bins = len(binning) - 1
    counts_mc = np.zeros(n_bins)
    counts_unf = np.zeros(n_bins)
    counts_boot = np.zeros((nboot, n_bins))
    counts_closure = np.zeros(n_bins)

    for fpath in batch_files:
        if verbose:
            print(f"  reading {os.path.basename(fpath)}")
        with h5.File(fpath, "r") as fh5:
            values = fh5[var][:]
            jet_pt = fh5["jet_pt"][:]
            mc_w = fh5["mc_weights"][:]
            nominal_w = fh5[nominal_wkey][:] if nominal_wkey in fh5 else None

            valid = _valid_mask(jet_pt)
            per_jet = values.ndim == 2

            if per_jet:
                flat_vals = values[valid]
            else:
                flat_vals = values

            # --- MC (mc_weights only) ---
            if per_jet:
                w_mc = np.where(valid, mc_w[:, None], 0.0)[valid]
            else:
                w_mc = mc_w
            counts_mc += np.histogram(flat_vals, bins=binning, weights=w_mc)[0]

            # --- Unfolded (mc_weights * nominal_w) ---
            if nominal_w is not None:
                combined = mc_w * nominal_w
                if per_jet:
                    w_unf = np.where(valid, combined[:, None], 0.0)[valid]
                else:
                    w_unf = combined
                counts_unf += np.histogram(flat_vals, bins=binning, weights=w_unf)[0]

            # --- Bootstrap ---
            for i in range(1, nboot + 1):
                key = f"weights{i}"
                if key not in fh5:
                    continue
                boot_w = fh5[key][:]
                combined_b = mc_w * boot_w
                if per_jet:
                    w_b = np.where(valid, combined_b[:, None], 0.0)[valid]
                else:
                    w_b = combined_b
                counts_boot[i - 1] += np.histogram(flat_vals, bins=binning, weights=w_b)[0]

            # --- Closure (mc_weights * closure_weights) ---
            if "closure_weights" in fh5:
                closure_w = fh5["closure_weights"][:]
                combined_c = mc_w * closure_w
                if per_jet:
                    w_c = np.where(valid, combined_c[:, None], 0.0)[valid]
                else:
                    w_c = combined_c
                counts_closure += np.histogram(flat_vals, bins=binning, weights=w_c)[0]

        del values, jet_pt, mc_w, valid
        gc.collect()

    return counts_mc, counts_unf, counts_boot, counts_closure


def accumulate_sys_histogram(batch_files, var, binning, nominal_wkey, verbose):
    """
    Accumulate mc_weights * nominal_wkey histogram for a systematic variation.
    Returns counts_unf of shape (n_bins,).
    """
    n_bins = len(binning) - 1
    counts_unf = np.zeros(n_bins)

    for fpath in batch_files:
        if verbose:
            print(f"  reading {os.path.basename(fpath)}")
        with h5.File(fpath, "r") as fh5:
            if var not in fh5 or nominal_wkey not in fh5:
                continue
            values = fh5[var][:]
            jet_pt = fh5["jet_pt"][:]
            mc_w = fh5["mc_weights"][:]
            nominal_w = fh5[nominal_wkey][:]

            valid = _valid_mask(jet_pt)
            per_jet = values.ndim == 2

            combined = mc_w * nominal_w
            if per_jet:
                flat_vals = values[valid]
                w_unf = np.where(valid, combined[:, None], 0.0)[valid]
            else:
                flat_vals = values
                w_unf = combined
            counts_unf += np.histogram(flat_vals, bins=binning, weights=w_unf)[0]

        del values, jet_pt, mc_w, valid
        gc.collect()

    return counts_unf


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def bootstrap_stat_unc(counts_boot, binning):
    """
    Compute relative statistical uncertainty per bin as std/mean over bootstrap
    replicas. Returns array of shape (n_bins,) with values in [0, inf).
    """
    mean = np.mean(counts_boot, axis=0)
    std = np.std(counts_boot, axis=0)
    return np.where(mean > 0, std / mean, 0.0)


def systematic_unc(sys_hist, ref_hist):
    """
    Fractional systematic uncertainty from one source: |sys/ref - 1|.
    Returns array of shape (n_bins,).
    """
    safe_ref = np.where(ref_hist > 0, ref_hist, np.nan)
    return np.abs(np.nan_to_num(sys_hist / safe_ref, nan=1.0) - 1.0)


# ---------------------------------------------------------------------------
# Normalise to density
# ---------------------------------------------------------------------------

def to_density(counts, binning):
    """Normalise counts to unit-area density; returns copy."""
    bin_widths = np.diff(binning)
    total = np.sum(counts * bin_widths)
    if total <= 0:
        return counts.copy()
    return counts / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_var(var, info, rapgap_mc, rapgap_unf, djangoh_mc,
             stat_unc, plot_folder, version, total_unc=None, blind=False):
    """
    Build a main + ratio panel plot for one observable and save to PDF.

    Parameters
    ----------
    var : str
    info : ObservableInfo
    rapgap_mc   : density-normalised counts (n_bins,) -- RAPGAP MC only
    rapgap_unf  : density-normalised counts (n_bins,) -- RAPGAP unfolded (or closure)
    djangoh_mc  : density-normalised counts (n_bins,) -- DJANGOH MC only
    stat_unc    : relative stat uncertainty per bin (n_bins,), from bootstrap
    plot_folder : str
    version     : str
    total_unc   : relative total uncertainty per bin (n_bins,), stat + sys combined.
                  If None, only stat_unc is shown.
    blind       : if True, label the unfolded points as the closure result
    """
    data_name = "Rapgap_closure" if blind else "Data_unfolded"
    binning = info.binning
    bin_centers = 0.5 * (binning[:-1] + binning[1:])
    unc_band = total_unc if total_unc is not None else stat_unc

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(wspace=0.025, hspace=0.1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax0.xaxis.set_visible(False)

    # ---- main panel ----
    ax0.stairs(
        rapgap_mc, binning,
        color=options.colors["Rapgap"],
        fill=True, alpha=0.2,
        label=options.name_translate["Rapgap"],
    )
    ax0.stairs(
        djangoh_mc, binning,
        color=options.colors["Djangoh"],
        linewidth=2,
        label=options.name_translate["Djangoh"],
    )
    ax0.errorbar(
        bin_centers, rapgap_unf,
        yerr=[rapgap_unf * stat_unc, rapgap_unf * stat_unc],
        fmt="o", color=options.colors[data_name],
        markersize=6,
        label=options.name_translate[data_name],
    )
    # Total uncertainty band on main panel
    ax0.fill_between(
        bin_centers,
        rapgap_unf * (1 - unc_band),
        rapgap_unf * (1 + unc_band),
        alpha=0.2, color=options.colors[data_name],
        step=None,
    )

    ylabel = r"$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s" % info.name
    ax0.set_ylabel(ylabel)
    if info.logy:
        ax0.set_yscale("log")
    if info.logx:
        ax0.set_xscale("log")
    ax0.set_ylim(info.ylow, info.yhigh)
    ax0.legend(loc="upper left")
    utils.FormatFig(info.name, ylabel, ax0)

    # ---- ratio panel: MC / unfolded ----
    ref = rapgap_unf
    safe_ref = np.where(ref > 0, ref, np.nan)

    ratio_rapgap  = rapgap_mc  / safe_ref
    ratio_djangoh = djangoh_mc / safe_ref

    ax1.stairs(ratio_rapgap,  binning, color=options.colors["Rapgap"],  linewidth=2)
    ax1.stairs(ratio_djangoh, binning, color=options.colors["Djangoh"], linewidth=2)

    # Total uncertainty band around 1
    for ibin in range(len(binning) - 1):
        xlow, xup = binning[ibin], binning[ibin + 1]
        unc = unc_band[ibin]
        ax1.fill_between(
            [xlow, xup], 1.0 - unc, 1.0 + unc,
            alpha=0.15, color="black",
        )
        ax1.bar(
            (xlow + xup) / 2, 2 * unc, width=(xup - xlow),
            bottom=1.0 - unc, hatch="//",
            color="none", edgecolor="grey",
        )

    ax1.errorbar(
        bin_centers, np.ones_like(bin_centers),
        yerr=stat_unc,
        fmt="o", color=options.colors[data_name], markersize=6,
    )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_ylim(0.5, 1.5)
    ax1.set_ylabel("MC / Unfolded")
    ax1.set_xlabel(info.name)
    if info.logx:
        ax1.set_xscale("log")

    os.makedirs(plot_folder, exist_ok=True)
    suffix = "closure" if blind else "unfolded"
    out_path = os.path.join(plot_folder, f"{version}_{var}_{suffix}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    flags = parse_arguments()

    replace_string = f"unfolded_{flags.niter}_centauro_boot"
    rapgap_pattern  = os.path.join(
        flags.data_folder, f"Rapgap_{flags.period}_{replace_string}_batch*.h5"
    )
    djangoh_pattern = os.path.join(
        flags.data_folder, f"Djangoh_{flags.period}_{replace_string}_batch*.h5"
    )

    rapgap_files  = sorted(glob.glob(rapgap_pattern))
    djangoh_files = sorted(glob.glob(djangoh_pattern))

    if not rapgap_files:
        raise FileNotFoundError(f"No Rapgap batch files found: {rapgap_pattern}")
    if not djangoh_files:
        raise FileNotFoundError(f"No Djangoh batch files found: {djangoh_pattern}")

    print(f"Found {len(rapgap_files)} Rapgap  batch files")
    print(f"Found {len(djangoh_files)} Djangoh batch files")

    # Collect systematic batch file lists once
    sys_files = {}
    if flags.sys:
        for sys_label in flags.sys_list:
            pattern = os.path.join(
                flags.data_folder,
                f"Rapgap_{flags.period}_{sys_label}_{replace_string}_batch*.h5",
            )
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"WARNING: no batch files found for {sys_label} ({pattern}), skipping")
            else:
                print(f"Found {len(files)} batch files for {sys_label}")
                sys_files[sys_label] = files

    version = f"Rapgap_{flags.period}"

    for var in VAR_NAMES:
        info = ObservableInfo(var)
        binning = info.binning
        print(f"\n--- {var} ---")

        print("  Rapgap ...")
        rapgap_mc_raw, rapgap_unf_raw, rapgap_boot_raw, rapgap_closure_raw = accumulate_var_histograms(
            rapgap_files, var, binning,
            flags.nominal_weight, flags.nboot, flags.verbose,
        )

        print("  Djangoh ...")
        djangoh_mc_raw, _, _, _ = accumulate_var_histograms(
            djangoh_files, var, binning,
            flags.nominal_weight, 0, flags.verbose,
        )

        # Normalise to density
        rapgap_mc      = to_density(rapgap_mc_raw,      binning)
        rapgap_unf     = to_density(rapgap_unf_raw,     binning)
        rapgap_closure = to_density(rapgap_closure_raw, binning)
        djangoh_mc     = to_density(djangoh_mc_raw,     binning)

        # Statistical uncertainty from bootstrap replicas
        stat_unc = bootstrap_stat_unc(rapgap_boot_raw, binning)

        # Systematic uncertainties
        total_unc = None
        if flags.sys:
            total_unc_sq = stat_unc ** 2

            # Model uncertainty: Rapgap closure vs Djangoh MC (matches plot_from_file)
            has_closure = np.any(rapgap_closure > 0)
            if not has_closure:
                print("  WARNING: no closure_weights found in Rapgap batch files; skipping model uncertainty")
            else:
                model_unc = systematic_unc(rapgap_closure, djangoh_mc)
                total_unc_sq += model_unc ** 2
                print(f"  model: max unc = {np.max(model_unc):.4f}")

            # Each systematic variation
            for sys_label, sfiles in sys_files.items():
                print(f"  {sys_label} ...")
                sys_raw = accumulate_sys_histogram(
                    sfiles, var, binning, flags.nominal_weight, flags.verbose
                )
                sys_hist = to_density(sys_raw, binning)
                unc = systematic_unc(sys_hist, rapgap_unf)
                total_unc_sq += unc ** 2
                print(f"  {sys_label}: max unc = {np.max(unc):.4f}")
                del sys_raw, sys_hist
                gc.collect()

            total_unc = np.sqrt(total_unc_sq)
            print(f"  total: max unc = {np.max(total_unc):.4f}")

        plot_ref = rapgap_closure if flags.blind else rapgap_unf
        plot_var(
            var, info,
            rapgap_mc, plot_ref, djangoh_mc,
            stat_unc,
            flags.plot_folder, version,
            total_unc=total_unc,
            blind=flags.blind,
        )

        del rapgap_mc_raw, rapgap_unf_raw, rapgap_boot_raw, rapgap_closure_raw, djangoh_mc_raw
        gc.collect()


if __name__ == "__main__":
    main()
