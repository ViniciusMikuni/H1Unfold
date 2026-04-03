"""
plot_from_batches.py

Memory-efficient version of plot_part_from_file.py for EEC observables.
Instead of loading all events at once, it reads multiple batch h5 files
(e.g. Rapgap_Eplus0607_unfolded_4_centauro_boot_batch0000.h5, batch0001.h5, ...)
and accumulates weighted histograms across them.

Systematics are calculated and applied exactly as in plot_part_from_file.py
(plot_part_observable in plot_utils.py):
  - nominal         = Rapgap   mc_weights * weights        (density)
  - nominal_closure = Djangoh  mc_weights only             (density)
  - For 'Rapgap' sys source : closure_weights, reference = nominal_closure
  - For all other sys sources: weights,        reference = nominal
  - Per-source uncertainty: (sys_hist / ref_hist - 1)^2
  - Bootstrap stat uncertainty: std / mean over bootstrap replicas
  - total_unc = sqrt( sum of squared uncertainties )

Plot style uses utils.HistRoutinePart (same as plot_part_from_file.py).
"""

import argparse
import glob
import os
import gc

import numpy as np
import h5py as h5

import utils
import options
from utils import ObservableInfo

utils.SetStyle()

var_names = ['deltaphi', 'jet_pt', 'jet_tau10', 'zjet', 'zjet_breit']#, 'eec', 'theta']


# ---------------------------------------------------------------------------
# Batch file discovery
# ---------------------------------------------------------------------------

def get_batch_files(data_folder, period, niter, suffix,
                    use_sys, sys_list=None, nominal='Rapgap',
                    reco=False):
    """
    Return a dict mapping dataset label -> sorted list of batch h5 file paths.

    File naming convention (example):
      Rapgap_Eplus0607_unfolded_4_centauro_boot_batch0000.h5
      Rapgap_Eplus0607_sys0_unfolded_4_centauro_boot_batch0000.h5
      Djangoh_Eplus0607_unfolded_4_centauro_boot_batch0000.h5
      data_Eplus0607_unfolded_4_centauro_reco_batch0000.h5  (reco mode, data only)
    """
    if sys_list is None:
        sys_list = ['sys0', 'sys1', 'sys5', 'sys7', 'sys11']

    mc_suffix = suffix.replace('_boot', '_reco_boot') if reco else suffix

    def _glob(base_name):
        pattern = os.path.join(
            data_folder,
            f'{base_name}_unfolded_{niter}_{mc_suffix}_batch*.h5'
        )
        files = sorted(glob.glob(pattern))
        return files

    batch_files = {
        'Rapgap':  _glob(f'Rapgap_{period}'),
        'Djangoh': _glob(f'Djangoh_{period}'),
    }

    if reco:
        reco_data_suffix = suffix.replace('_boot', '') + '_reco'
        data_pattern = os.path.join(
            data_folder,
            f'data_{period}_unfolded_{niter}_{reco_data_suffix}_batch*.h5'
        )
        batch_files['data'] = sorted(glob.glob(data_pattern))

    if use_sys:
        for sys in sys_list:
            batch_files[sys] = _glob(f'{nominal}_{period}_{sys}')

    return batch_files


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Memory-efficient EEC plot from batch h5 files.'
    )
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/H1v2/h5',
                        help='Folder containing the batch h5 files')
    parser.add_argument('--config', default='config_general.json',
                        help='Basic config file containing general options')
    parser.add_argument('--plot_folder', default='../plots',
                        help='Folder to store plots')
    parser.add_argument('--period', default='Eplus0607',
                        help='Data-taking period string in file names')
    parser.add_argument('--suffix', default='centauro_boot',
                        help='Middle suffix in batch file names (e.g. centauro_boot)')
    parser.add_argument('--reco', action='store_true', default=False,
                        help='Plot reco level results')
    parser.add_argument('--sys', action='store_true', default=False,
                        help='Load systematic variations')
    parser.add_argument('--blind', action='store_true', default=False,
                        help='Show results based on closure instead of data')
    parser.add_argument('--niter', type=int, default=4,
                        help='OmniFold iteration to load')
    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Compute stat uncertainty from bootstrap replicas in batch files')
    parser.add_argument('--nboot', type=int, default=50,
                        help='Number of bootstrap replicas (weights1..weightsN) in the files')
    parser.add_argument('--eec', action='store_true', default=False,
                        help='Use eec mask and E_wgt (EEC mode)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Increase print level')
    flags = parser.parse_args()

    if flags.blind and flags.reco:
        raise ValueError('Unable to run blinded and reco modes at the same time')
    return flags


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def to_density(counts, binning):
    """Normalise raw counts to unit-area density histogram."""
    bin_widths = np.diff(binning)
    total = np.sum(counts * bin_widths)
    if total <= 0:
        return counts.copy()
    return counts / total


# ---------------------------------------------------------------------------
# Per-file histogram accumulation helpers
# ---------------------------------------------------------------------------

def _accumulate_eec_file(fh5, binning, weight_keys, include_E_wgt):
    """
    Accumulate EEC histograms from one open h5 file (read fully).

    weight_keys : list of (label, h5_key_or_None)
        If h5_key_or_None is not None: weight = mc_weights * fh5[h5_key]
        If h5_key_or_None is None:     weight = mc_weights only
    include_E_wgt : bool
        Multiply pair weights by E_wgt (for plot feed_dict, not systematics).

    Returns dict label -> raw count array of shape (n_bins,).
    """
    n_bins = len(binning) - 1
    out = {lbl: np.zeros(n_bins) for lbl, _ in weight_keys}

    eec_vals = fh5['eec'][:]      # (N, P)
    E_wgt_vals = fh5['E_wgt'][:]  # (N, P)
    mc_w = fh5['mc_weights'][:]   # (N,)
    valid = eec_vals != -100      # (N, P)

    for lbl, wk in weight_keys:
        # 'weights' is absent in bootstrap-mode batch files; fall back to 'weights_nominal'
        if wk == 'weights' and wk not in fh5 and 'weights_nominal' in fh5:
            wk = 'weights_nominal'
        if wk is not None and wk in fh5:
            event_w = mc_w * fh5[wk][:]
        else:
            event_w = mc_w.copy()

        if include_E_wgt:
            pair_w = event_w[:, None] * E_wgt_vals
        else:
            pair_w = np.broadcast_to(event_w[:, None], eec_vals.shape).copy()

        out[lbl] += np.histogram(eec_vals[valid], bins=binning,
                                 weights=pair_w[valid])[0]

    return out


def _accumulate_jet_file(fh5, var, binning, weight_keys):
    """
    Accumulate jet-observable histograms from one open h5 file (read fully).
    Same interface as _accumulate_eec_file (minus include_E_wgt).
    """
    n_bins = len(binning) - 1
    out = {lbl: np.zeros(n_bins) for lbl, _ in weight_keys}

    values = fh5[var][:]        # (N, J) or (N,)
    jet_pt = fh5['jet_pt'][:]   # (N, J)
    mc_w = fh5['mc_weights'][:] # (N,)

    per_jet = values.ndim == 2
    if per_jet:
        valid = jet_pt > 0
        flat_vals = values[valid]
    else:
        flat_vals = values

    for lbl, wk in weight_keys:
        # 'weights' is absent in bootstrap-mode batch files; fall back to 'weights_nominal'
        if wk == 'weights' and wk not in fh5 and 'weights_nominal' in fh5:
            wk = 'weights_nominal'
        if wk is not None and wk in fh5:
            event_w = mc_w * fh5[wk][:]
        else:
            event_w = mc_w.copy()

        if per_jet:
            flat_w = np.broadcast_to(event_w[:, None], values.shape).copy()[valid]
        else:
            flat_w = event_w

        out[lbl] += np.histogram(flat_vals, bins=binning, weights=flat_w)[0]

    return out


# ---------------------------------------------------------------------------
# Multi-file accumulation (iterates over batch file list)
# ---------------------------------------------------------------------------

def accumulate_histograms(file_list, var, binning, weight_keys,
                          eec_mode=False, include_E_wgt=False, verbose=False):
    """
    Iterate over a list of batch h5 files and accumulate weighted histograms.

    Returns dict: label -> raw count array of shape (n_bins,)
    """
    n_bins = len(binning) - 1
    totals = {lbl: np.zeros(n_bins) for lbl, _ in weight_keys}

    for fpath in file_list:
        with h5.File(fpath, 'r') as fh5:
            n_events = fh5['mc_weights'].shape[0]
            if verbose:
                print(f'  {os.path.basename(fpath)}  ({n_events} events)')
            if eec_mode:
                batch = _accumulate_eec_file(fh5, binning, weight_keys, include_E_wgt)
            else:
                batch = _accumulate_jet_file(fh5, var, binning, weight_keys)
        for lbl in totals:
            totals[lbl] += batch[lbl]
        gc.collect()

    return totals


def accumulate_bootstrap_histograms(file_list, var, binning, nboot,
                                    eec_mode=False, include_E_wgt=False,
                                    verbose=False):
    """
    Accumulate one histogram per bootstrap replica (weights1 .. weightsN)
    across all batch files.

    Returns array of shape (nboot, n_bins) with raw counts.
    """
    n_bins = len(binning) - 1
    boot_counts = np.zeros((nboot, n_bins))

    for fpath in file_list:
        with h5.File(fpath, 'r') as fh5:
            n_events = fh5['mc_weights'].shape[0]
            if verbose:
                print(f'  [bootstrap] {os.path.basename(fpath)}  ({n_events} events)')
            weight_keys = [
                (str(i), f'weights{i}')
                for i in range(1, nboot)
                if f'weights{i}' in fh5
            ]
            if eec_mode:
                batch = _accumulate_eec_file(fh5, binning, weight_keys, include_E_wgt)
            else:
                batch = _accumulate_jet_file(fh5, var, binning, weight_keys)

        for i in range(1, nboot):
            if str(i) in batch:
                boot_counts[i - 1] += batch[str(i)]
        gc.collect()

    return boot_counts


# ---------------------------------------------------------------------------
# Per-variable plot function
# ---------------------------------------------------------------------------

def plot_observable(flags, var, batch_files, version):
    info = ObservableInfo(var)
    binning = info.binning
    bin_widths = np.diff(binning)
    bin_centers = 0.5 * (binning[:-1] + binning[1:])

    eec_mode = flags.eec and var in ('eec', 'theta')

    weight_name = 'closure_weights' if flags.blind else 'weights'
    if flags.blind:
        data_name = 'Rapgap_closure'
    elif flags.reco:
        data_name = 'Rapgap_unfolded'
    else:
        data_name = 'Data_unfolded'

    rapgap_files  = batch_files['Rapgap']
    djangoh_files = batch_files['Djangoh']

    # ------------------------------------------------------------------
    # Debug: summarise input files and event counts
    # ------------------------------------------------------------------
    if flags.verbose:
        print(f'\n=== {var} ===')
        for label, flist in batch_files.items():
            if not flist:
                print(f'  {label}: no files found')
                continue
            total_events = 0
            for fpath in flist:
                with h5.File(fpath, 'r') as fh5:
                    count_key = 'mc_weights' if label != 'data' else 'jet_pt'
                    total_events += fh5[count_key].shape[0]
            print(f'  {label}: {len(flist)} file(s), {total_events} events total')

    # ------------------------------------------------------------------
    # Accumulate raw histograms
    # ------------------------------------------------------------------

    # Rapgap: always collect mc, weights, and closure_weights separately
    # so we can replicate plot_part_observable's systematics exactly.
    if flags.verbose:
        print(f'Accumulating Rapgap ({var}) ...')
    rapgap_sys_keys = [
        ('mc',      None),
        ('weights', 'weights'),
        ('closure', 'closure_weights'),
    ]
    rapgap_sys = accumulate_histograms(
        rapgap_files, var, binning, rapgap_sys_keys,
        eec_mode=eec_mode, include_E_wgt=False, verbose=flags.verbose,
    )

    # For the plot feed_dict, include E_wgt in EEC mode
    if eec_mode:
        rapgap_plot_keys = [
            ('mc',       None),
            ('unfolded', weight_name),
        ]
        rapgap_plot = accumulate_histograms(
            rapgap_files, var, binning, rapgap_plot_keys,
            eec_mode=True, include_E_wgt=True, verbose=flags.verbose,
        )
    else:
        rapgap_plot = {
            'mc':       rapgap_sys['mc'],
            'unfolded': rapgap_sys['closure'] if flags.blind else rapgap_sys['weights'],
        }

    # Djangoh: mc only (for nominal_closure) + weights (for model uncertainty sys)
    if flags.verbose:
        print(f'Accumulating Djangoh ({var}) ...')
    djangoh_sys = accumulate_histograms(
        djangoh_files, var, binning, [('mc', None), ('weights', 'weights')],
        eec_mode=eec_mode, include_E_wgt=False, verbose=flags.verbose,
    )
    if eec_mode:
        djangoh_plot = accumulate_histograms(
            djangoh_files, var, binning, [('mc', None)],
            eec_mode=True, include_E_wgt=True, verbose=flags.verbose,
        )
    else:
        djangoh_plot = djangoh_sys

    # Systematic variation files
    sys_raw = {}
    if flags.sys:
        for sys_label, sfiles in batch_files.items():
            if sys_label in ('Rapgap', 'Djangoh', 'data'):
                continue
            if flags.verbose:
                print(f'Accumulating {sys_label} ({var}) ...')
            # In reco mode use mc_weights only (unit unfolded weights),
            # matching plot_part_observable's `if flags.reco: sys_weights = np.ones_like(...)`
            sys_weight_key = None if flags.reco else 'weights'
            res = accumulate_histograms(
                sfiles, var, binning, [('weights', sys_weight_key)],
                eec_mode=eec_mode, include_E_wgt=False, verbose=flags.verbose,
            )
            sys_raw[sys_label] = res['weights']

    # Reco data raw counts (for stat uncertainty)
    data_raw_counts = None
    if flags.reco and 'data' in batch_files:
        if flags.verbose:
            print(f'Accumulating data counts ({var}) ...')
        n_bins = len(binning) - 1
        data_raw_counts = np.zeros(n_bins)
        for fpath in batch_files['data']:
            with h5.File(fpath, 'r') as fh5:
                n_events = fh5['jet_pt'].shape[0]
                if flags.verbose:
                    print(f'  {os.path.basename(fpath)}  ({n_events} events)')
                if eec_mode:
                    eec_b = fh5['eec'][:]
                    valid = eec_b != -100
                    data_raw_counts += np.histogram(eec_b[valid], bins=binning)[0]
                else:
                    vals = fh5[var][:]
                    jpt = fh5['jet_pt'][:]
                    valid = jpt > 0
                    flat = vals[valid] if vals.ndim == 2 else vals
                    data_raw_counts += np.histogram(flat, bins=binning)[0]
            gc.collect()

    # Bootstrap (weights1..weightsN stored inside the same Rapgap batch files)
    boot_raw = None
    if flags.bootstrap:
        if flags.verbose:
            print(f'Accumulating bootstrap ({var}) ...')
        boot_raw = accumulate_bootstrap_histograms(
            rapgap_files, var, binning, flags.nboot,
            eec_mode=eec_mode, include_E_wgt=False, verbose=flags.verbose,
        )

    # ------------------------------------------------------------------
    # Density-normalise for systematics
    # ------------------------------------------------------------------
    # In reco mode, nominal uses mc_weights only (no unfolded reweighting)
    nominal         = to_density(rapgap_sys['mc'] if flags.reco else rapgap_sys['weights'], binning)
    nominal_closure = to_density(djangoh_sys['mc'], binning)

    # ------------------------------------------------------------------
    # Systematic uncertainty -- EXACT same formula as plot_part_observable
    # ------------------------------------------------------------------
    total_unc = None
    data_stat_unc = np.zeros(len(binning) - 1)

    if flags.sys:
        total_unc = np.zeros(len(binning) - 1)

        all_sources = list(batch_files.keys())
        for sys_label in all_sources:
            if flags.reco and sys_label in ('Rapgap', 'Djangoh', 'data'):
                continue

            print(sys_label)

            if sys_label == 'Rapgap':
                sys_hist = to_density(
                    rapgap_sys['mc'] if flags.reco else rapgap_sys['closure'],
                    binning
                )
                ref_hist = nominal_closure

            elif sys_label == 'Djangoh':
                sys_hist = to_density(djangoh_sys['weights'], binning)
                ref_hist = nominal

            elif sys_label in sys_raw:
                sys_hist = to_density(sys_raw[sys_label], binning)
                ref_hist = nominal

            else:
                continue

            unc = (np.ma.divide(sys_hist, ref_hist).filled(1) - 1) ** 2
            total_unc += unc
            print(f'{sys_label}: max uncertainty = {np.max(np.sqrt(unc)):.4f}')

        # Statistical uncertainties
        if flags.reco:
            unc = 1.0 / (1e-9 + data_raw_counts)
            total_unc += unc
            data_stat_unc = np.sqrt(unc)
            print(f'stat: max uncertainty = {np.max(data_stat_unc):.4f}')
        else:
            if flags.bootstrap and boot_raw is not None:
                print('Running over bootstrap entries')
                valid_boots = [
                    boot_raw[i - 1]
                    for i in range(1, flags.nboot)
                    if np.any(boot_raw[i - 1] > 0)
                ]
                if valid_boots:
                    boot_densities = np.array(
                        [to_density(b, binning) for b in valid_boots]
                    )
                    stat_unc = np.ma.divide(
                        np.std(boot_densities, axis=0),
                        np.mean(boot_densities, axis=0),
                    ).filled(0)
                    data_stat_unc = stat_unc
                    total_unc += stat_unc ** 2
                    print(f'bootstrap: max uncertainty = {np.max(stat_unc):.4f}')

        total_unc = np.sqrt(total_unc)

    # ------------------------------------------------------------------
    # Build feed_dict for utils.HistRoutine / HistRoutinePart
    #
    # Bin-centers trick: pass bin_centers as "data" with
    # weight = density * bin_width, so np.histogram(..., density=True)
    # inside HistRoutinePart recovers the pre-computed density exactly.
    # ------------------------------------------------------------------
    rapgap_mc_density  = to_density(rapgap_plot['mc'],      binning)
    rapgap_unf_density = to_density(rapgap_plot['unfolded'], binning)
    djangoh_mc_density = to_density(djangoh_plot['mc'],      binning)

    feed_dict = {
        data_name: bin_centers,
        'Rapgap':  bin_centers,
        'Djangoh': bin_centers,
    }
    weights_dict = {
        data_name: rapgap_unf_density * bin_widths,
        'Rapgap':  rapgap_mc_density  * bin_widths,
        'Djangoh': djangoh_mc_density * bin_widths,
    }

    if flags.reco and data_raw_counts is not None:
        data_density = to_density(data_raw_counts, binning)
        feed_dict['data']    = bin_centers
        weights_dict['data'] = data_density * bin_widths

    ylabel = (
        r'1/N $\mathrm{dN}/\mathrm{d}$%s' % info.name
        if flags.reco
        else r'$1/\sigma$ $\mathrm{d}\sigma/\mathrm{d}$%s' % info.name
    )

    hist_routine = utils.HistRoutinePart if eec_mode else utils.HistRoutine
    fig, ax = hist_routine(
        feed_dict,
        xlabel=info.name,
        ylabel=ylabel,
        weights=weights_dict,
        logy=info.logy,
        logx=info.logx,
        binning=binning,
        reference_name='data' if flags.reco else data_name,
        label_loc='upper left',
        uncertainty=total_unc,
        stat_uncertainty=data_stat_unc,
        show_stat_points=not flags.blind,
    )

    ax.set_ylim(info.ylow, info.yhigh)
    os.makedirs(flags.plot_folder, exist_ok=True)
    if flags.blind:
        plot_tag = 'closure'
    elif flags.reco:
        plot_tag = 'reco'
    else:
        plot_tag = 'unfolded'
    fig.savefig(os.path.join(flags.plot_folder, f'{version}_{var}_{plot_tag}.pdf'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    flags = parse_arguments()
    opt = utils.LoadJson(flags.config)

    batch_files = get_batch_files(
        data_folder=flags.data_folder,
        period=flags.period,
        niter=flags.niter,
        suffix=flags.suffix,
        use_sys=flags.sys,
        reco=flags.reco,
    )

    for label, files in batch_files.items():
        if not files:
            print(f'WARNING: no batch files found for {label}, skipping')
        else:
            print(f'{label}: {len(files)} file(s)')
            for f in files:
                print(f'  {os.path.basename(f)}')

    version = opt['NAME']

    for var in var_names:
        if 'weight' in var:
            continue
        plot_observable(flags, var, batch_files, version)


if __name__ == '__main__':
    main()
