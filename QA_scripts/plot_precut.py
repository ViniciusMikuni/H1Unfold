"""Pre-cut QA plots of the DIS selection variables, split by simulation run.

Reads the nominal ROOT minitrees directly, including the runs that prepare_data.py
excludes, and fills weighted histograms of Q2, y, ptmiss and Empz for each run.
Runs are compared within each generator, Rapgap is compared with Djangoh, and
MC is compared with data, with no reco selection and with N-1 reco selections.
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "scripts"))
from prepare_data import find_files_with_string  # noqa: E402

RECO_VARS = ["Q2", "y", "ptmiss", "Empz"]
GEN_VARS = ["gen_Q2", "gen_y"]
MC_VARIABLES = {"nocut": RECO_VARS + GEN_VARS, "nminus1": RECO_VARS}
DATA_VARIABLES = {"nocut": RECO_VARS, "nminus1": RECO_VARS}

# Reco selection from prepare_data.convert_to_np as (low, high); None is open
RECO_CUTS = {
    "Q2": (150.0, None),
    "y": (0.08, 0.7),
    "ptmiss": (None, 10.0),
    "Empz": (45.0, 65.0),
}
# Fiducial (pass_gen) selection, only drawn as lines on the gen-level plots
GEN_CUTS = {"gen_Q2": (150.0, None), "gen_y": (0.2, 0.7)}

BINNING = {
    "Q2": np.logspace(1, np.log10(2e4), 41),
    "y": np.linspace(0, 1, 41),
    "ptmiss": np.linspace(0, 40, 41),
    "Empz": np.linspace(0, 80, 41),
}
BINNING["gen_Q2"] = BINNING["Q2"]
BINNING["gen_y"] = BINNING["y"]

VAR_LABELS = {
    "Q2": r"$Q^2$ [GeV$^2$]",
    "y": r"$y$",
    "ptmiss": r"$p_T^{miss}$ [GeV]",
    "Empz": r"$E-p_z$ [GeV]",
    "gen_Q2": r"Generator $Q^2$ [GeV$^2$]",
    "gen_y": r"Generator $y$",
}
CUT_LABELS = {
    "Q2": r"$Q^2>150$",
    "y": r"$0.08<y<0.7$",
    "ptmiss": r"$p_T^{miss}<10$",
    "Empz": r"$45<E-p_z<65$",
}
LOGX = {"Q2", "gen_Q2"}
LOGY = {"Q2", "gen_Q2", "ptmiss"}

LEPTON_LABELS = {"Ep": r"$e^{+}p$ 2006-07", "Em": r"$e^{-}p$ 2006"}
GENERATOR_COLORS = {"Rapgap": "darkorange", "Djangoh": "#8c564b"}  # options.colors
RUN_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
EXCLUDED_COLOR = "#d62728"

FILE_NUMBER = re.compile(r"_(\d+)\.nominal\.root$")

PERFILE_PANELS = [
    ("n_events", "Events after MC preselection", None),
    ("wgt_mean", r"$\Sigma w\,/\,N$", None),
    ("mean_Q2", r"Mean $\log_{10}(Q^2/\mathrm{GeV}^2)$", "sem_Q2"),
    ("mean_y", r"Mean $y$", "sem_y"),
    ("mean_ptmiss", r"Mean $p_T^{miss}$ [GeV]", "sem_ptmiss"),
    ("mean_Empz", r"Mean $E-p_z$ [GeV]", "sem_Empz"),
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "runs.yaml"),
        help="YAML file with the sample and run definitions",
    )
    parser.add_argument(
        "--plot_folder",
        default=os.path.join(SCRIPT_DIR, "..", "plots", "QA", "precut"),
        help="Folder to store plots",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Histogram cache file (default: <plot_folder>/precut_hists.npz)",
    )
    parser.add_argument(
        "--from_cache",
        action="store_true",
        default=False,
        help="Plot from the histogram cache instead of reading the ROOT files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Read at most this many files per run (for debugging)",
    )
    return parser.parse_args()


def assign_runs(name, sample, root_dir, max_files):
    """Return {run: [(file_number, path), ...]} and report missing or unassigned files."""
    directory = os.path.join(root_dir, sample["subdir"])
    by_number = {}
    for filename in find_files_with_string(directory, sample["pattern"]):
        m = FILE_NUMBER.search(filename)
        if m:
            by_number[int(m.group(1))] = os.path.join(directory, filename)

    run_files = {}
    assigned = set()
    for run, info in sample["runs"].items():
        low, high = info["files"]
        expected = range(low, high + 1)
        assigned.update(expected)
        found = [n for n in expected if n in by_number]
        missing = [n for n in expected if n not in by_number]
        message = f"{name} run {run}: {len(found)}/{len(expected)} files"
        if missing:
            message += f", missing file numbers {missing}"
        print(message)
        if max_files is not None:
            found = found[:max_files]
        run_files[str(run)] = [(n, by_number[n]) for n in found]

    unassigned = sorted(set(by_number) - assigned)
    if unassigned:
        print(f"WARNING: {name} files not assigned to any run: {unassigned}")
    return run_files


def read_file(path, tree, is_mc, gen_q2_min):
    branches = RECO_VARS + ["wgt"] + (GEN_VARS if is_mc else [])
    with uproot.open(path) as f:
        arrays = f[tree].arrays(branches, library="np")
    n_raw = arrays["wgt"].shape[0]
    if is_mc:
        keep = arrays["gen_Q2"] > gen_q2_min
        arrays = {key: value[keep] for key, value in arrays.items()}
    return arrays, n_raw


def selection(arrays, var, mode):
    """N-1 selection: every reco cut except the one on var."""
    mask = np.ones(arrays["wgt"].shape[0], dtype=bool)
    if mode == "nocut":
        return mask
    for cut_var, (low, high) in RECO_CUTS.items():
        if cut_var == var:
            continue
        if low is not None:
            mask &= arrays[cut_var] > low
        if high is not None:
            mask &= arrays[cut_var] < high
    return mask


def new_hists(variables):
    return {
        mode: {var: np.zeros((2, len(BINNING[var]) - 1)) for var in var_list}
        for mode, var_list in variables.items()
    }


def fill_histograms(hists, arrays, variables):
    """Accumulate sum(w) and sum(w^2), keeping under/overflow in the edge bins."""
    for mode, var_list in variables.items():
        for var in var_list:
            bins = BINNING[var]
            mask = selection(arrays, var, mode) & np.isfinite(arrays[var])
            values = np.clip(arrays[var][mask], bins[0], bins[-1])
            weights = arrays["wgt"][mask].astype(np.float64)
            hists[mode][var][0] += np.histogram(values, bins, weights=weights)[0]
            hists[mode][var][1] += np.histogram(values, bins, weights=weights**2)[0]


def file_stats(arrays, n_raw):
    n_events = arrays["wgt"].shape[0]
    stats = {
        "n_raw": n_raw,
        "n_events": n_events,
        "wgt_mean": np.sum(arrays["wgt"]) / max(n_events, 1),
    }
    for var in RECO_VARS:
        finite = np.isfinite(arrays[var])
        stats[f"nonfinite_{var}"] = np.sum(~finite)
        values = arrays[var][finite]
        if var == "Q2":
            values = np.log10(values[values > 0])
        if values.shape[0] == 0:
            stats[f"mean_{var}"], stats[f"sem_{var}"] = np.nan, np.nan
            continue
        stats[f"mean_{var}"] = np.mean(values)
        stats[f"sem_{var}"] = np.std(values) / np.sqrt(values.shape[0])
    return stats


def collect(config, max_files):
    root_dir = config["root_dir"]
    gen_q2_min = config["mc_gen_q2_min"]
    hists = {}
    perfile = {}

    for name, sample in config["samples"].items():
        run_files = assign_runs(name, sample, root_dir, max_files)
        hists[name] = {}
        rows = []
        for run, files in run_files.items():
            hists[name][run] = new_hists(MC_VARIABLES)
            for number, path in files:
                print(f"Reading {os.path.basename(path)}")
                arrays, n_raw = read_file(path, sample["tree"], True, gen_q2_min)
                fill_histograms(hists[name][run], arrays, MC_VARIABLES)
                stats = file_stats(arrays, n_raw)
                for var in RECO_VARS:
                    if stats[f"nonfinite_{var}"]:
                        print(
                            f"WARNING: {os.path.basename(path)} has "
                            f"{stats[f'nonfinite_{var}']} non-finite {var} values"
                        )
                rows.append({"file_number": number, "run": run, **stats})
        perfile[name] = {key: np.array([row[key] for row in rows]) for key in rows[0]}

    for lepton, data in config["data"].items():
        name = f"data{lepton}"
        hists[name] = {"data": new_hists(DATA_VARIABLES)}
        for filename in data["files"]:
            path = os.path.join(root_dir, data["subdir"], filename)
            print(f"Reading {filename}")
            arrays, _ = read_file(path, data["tree"], False, gen_q2_min)
            fill_histograms(hists[name]["data"], arrays, DATA_VARIABLES)

    return hists, perfile


def save_cache(path, hists, perfile):
    flat = {f"bins/{var}": bins for var, bins in BINNING.items()}
    for name, runs in hists.items():
        for run, modes in runs.items():
            for mode, variables in modes.items():
                for var, hist in variables.items():
                    flat[f"hist/{name}/{run}/{mode}/{var}"] = hist
    for name, fields in perfile.items():
        for field, values in fields.items():
            flat[f"perfile/{name}/{field}"] = values
    np.savez(path, **flat)
    print(f"Saved histogram cache to {path}")


def load_cache(path):
    hists = {}
    perfile = {}
    with np.load(path) as cache:
        for key in cache.files:
            parts = key.split("/")
            if parts[0] == "bins":
                if not np.array_equal(cache[key], BINNING[parts[1]]):
                    raise ValueError(
                        f"Binning of {parts[1]} changed since {path} was written; "
                        "rerun without --from_cache"
                    )
            elif parts[0] == "hist":
                _, name, run, mode, var = parts
                hists.setdefault(name, {}).setdefault(run, {}).setdefault(mode, {})
                hists[name][run][mode][var] = cache[key]
            else:
                _, name, field = parts
                perfile.setdefault(name, {})[field] = cache[key]
    return hists, perfile


def set_style():
    hep.style.use("CMS")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "legend.frameon": False,
        }
    )


def normalize(hist, bins):
    """Unit-area density and its statistical uncertainty."""
    sumw, sumw2 = hist
    total = np.sum(sumw)
    if total <= 0:
        return np.zeros_like(sumw), np.zeros_like(sumw)
    widths = np.diff(bins)
    return sumw / (total * widths), np.sqrt(sumw2) / (total * widths)


def cut_values(var):
    low, high = {**RECO_CUTS, **GEN_CUTS}.get(var, (None, None))
    return [cut for cut in (low, high) if cut is not None]


def mode_description(var, mode):
    if mode == "gen":
        return "Generator level, no fiducial selection"
    if mode == "nocut":
        return "No reco selection"
    return "N-1: " + ", ".join(CUT_LABELS[v] for v in RECO_CUTS if v != var)


def draw_comparison(
    subfig, var, curves, reference, ratio_label, info_lines, gen_q2_min
):
    """Standalone figure: normalized distributions on top, ratio to reference below."""
    bins = BINNING[var]
    if var in LOGX:
        centers = np.sqrt(bins[1:] * bins[:-1])
    else:
        centers = 0.5 * (bins[1:] + bins[:-1])

    gs = subfig.add_gridspec(
        2,
        1,
        height_ratios=[3, 1],
        hspace=0.06,
        left=0.17,
        right=0.95,
        top=0.95,
        bottom=0.12,
    )
    ax0 = subfig.add_subplot(gs[0])
    ax1 = subfig.add_subplot(gs[1], sharex=ax0)

    ref_val, ref_err = normalize(reference["hist"], bins)
    ymax = 0.0
    ymin = np.inf
    for curve in curves:
        val, err = normalize(curve["hist"], bins)
        if curve.get("data", False):
            ax0.errorbar(
                centers,
                val,
                yerr=err,
                fmt="o",
                color="black",
                markersize=5,
                label=curve["label"],
            )
        else:
            ax0.stairs(
                val,
                bins,
                color=curve["color"],
                linestyle=curve.get("linestyle", "-"),
                linewidth=curve.get("linewidth", 2),
                label=curve["label"],
            )
        if np.any(val > 0):
            ymax = max(ymax, np.max(val))
            ymin = min(ymin, np.min(val[val > 0]))

        if curve is reference:
            continue
        ratio = np.ma.divide(val, ref_val).filled(np.nan)
        ratio_err = np.ma.divide(err, ref_val).filled(np.nan)
        color = "black" if curve.get("data", False) else curve["color"]
        if not curve.get("data", False):
            ax1.stairs(
                ratio,
                bins,
                color=color,
                linestyle=curve.get("linestyle", "-"),
                linewidth=2,
            )
        ax1.errorbar(
            centers,
            ratio,
            yerr=ratio_err,
            fmt="o" if curve.get("data", False) else "none",
            color=color,
            markersize=4,
            elinewidth=1,
        )

    # Statistical uncertainty of the reference
    rel_err = np.ma.divide(ref_err, ref_val).filled(0)
    ax1.stairs(
        1 + rel_err, bins, baseline=1 - rel_err, fill=True, color="gray", alpha=0.3
    )
    ax1.axhline(1.0, color="black", linewidth=1)

    for ax in (ax0, ax1):
        for cut in cut_values(var):
            ax.axvline(cut, color="gray", linestyle="--", linewidth=1.5)
        if var in ("Q2", "gen_Q2"):
            ax.axvline(gen_q2_min, color="gray", linestyle=":", linewidth=1.5)

    if var in LOGX:
        ax0.set_xscale("log")
    ax0.set_xlim(bins[0], bins[-1])
    if ymax == 0:
        ymax, ymin = 1.0, 0.1
    if var in LOGY:
        ymin = max(ymin, ymax * 1e-6)
        decades = np.log10(ymax / ymin)
        ax0.set_yscale("log")
        ax0.set_ylim(ymin * 0.5, ymax * 10 ** (0.8 * decades + 0.5))
    else:
        ax0.set_ylim(0, 1.7 * ymax)
    ax1.set_ylim(0.5, 1.5)

    ax0.tick_params(labelbottom=False)
    ax0.set_xlabel("")
    ax0.set_ylabel("Normalized entries")
    ax1.set_xlabel(VAR_LABELS[var])
    ax1.set_ylabel(ratio_label, fontsize=16)
    ax0.legend(loc="upper right")
    ax0.text(
        0.03,
        0.96,
        "\n".join(info_lines),
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=14,
    )


def info_block(lepton, title, var, mode, gen_q2_min):
    mc_line = rf"MC: gen $Q^2>{gen_q2_min:g}$ GeV$^2$"
    if var in ("Q2", "gen_Q2"):
        mc_line += " (dotted)"
    return [
        f"{LEPTON_LABELS[lepton]}, {title}",
        mode_description(var, mode),
        mc_line,
    ]


def save_page(pdf, variables, draw):
    nrows = (len(variables) + 1) // 2
    fig = plt.figure(figsize=(20, 9 * nrows))
    subfigs = np.atleast_1d(fig.subfigures(nrows, 2)).ravel()
    for subfig, var in zip(subfigs, variables):
        draw(subfig, var)
    pdf.savefig(fig)
    plt.close(fig)


def files_per_run(perfile, run):
    if "run" not in perfile:
        return 0
    return np.sum(perfile["run"] == run)


def run_curves(sample, sample_hists, perfile, mode, var):
    curves = []
    kept = 0
    for run, info in sample["runs"].items():
        run = str(run)
        nfiles = files_per_run(perfile, run)
        if info.get("excluded", False):
            color, linestyle = EXCLUDED_COLOR, "--"
            label = f"Run {run} ({nfiles} files, excluded)"
        else:
            color, linestyle = RUN_COLORS[kept % len(RUN_COLORS)], "-"
            label = f"Run {run} ({nfiles} files)"
            kept += 1
        curves.append(
            {
                "label": label,
                "hist": sample_hists[run][mode][var],
                "color": color,
                "linestyle": linestyle,
            }
        )
    return curves


def combine_kept(sample, sample_hists, mode, var):
    return sum(
        sample_hists[str(run)][mode][var]
        for run, info in sample["runs"].items()
        if not info.get("excluded", False)
    )


def plot_runs(flags, config, hists, perfile, lepton, generator, name):
    sample = config["samples"][name]
    data_hists = hists[f"data{lepton}"]["data"]
    gen_q2_min = config["mc_gen_q2_min"]
    title = f"{generator} runs"
    output = os.path.join(flags.plot_folder, f"{lepton}_{generator}_runs.pdf")

    with PdfPages(output) as pdf:
        for mode in ["nocut", "nminus1"]:

            def draw(subfig, var):
                data = {"label": "Data", "hist": data_hists[mode][var], "data": True}
                curves = run_curves(sample, hists[name], perfile[name], mode, var)
                draw_comparison(
                    subfig,
                    var,
                    curves + [data],
                    data,
                    "MC / Data",
                    info_block(lepton, title, var, mode, gen_q2_min),
                    gen_q2_min,
                )

            save_page(pdf, RECO_VARS, draw)

        def draw_gen(subfig, var):
            kept = {
                "label": "Kept runs combined",
                "hist": combine_kept(sample, hists[name], "nocut", var),
                "color": "black",
                "linewidth": 1,
            }
            curves = run_curves(sample, hists[name], perfile[name], "nocut", var)
            draw_comparison(
                subfig,
                var,
                curves + [kept],
                kept,
                "Run / kept",
                info_block(lepton, title, var, "gen", gen_q2_min),
                gen_q2_min,
            )

        save_page(pdf, GEN_VARS, draw_gen)
    print(f"Saved {output}")


def plot_generators(flags, config, hists, lepton, samples):
    data_hists = hists[f"data{lepton}"]["data"]
    gen_q2_min = config["mc_gen_q2_min"]
    title = "Rapgap vs Djangoh (kept runs)"
    output = os.path.join(flags.plot_folder, f"{lepton}_RapgapVsDjangoh.pdf")

    def generator_curves(mode, var):
        return [
            {
                "label": generator,
                "hist": combine_kept(config["samples"][name], hists[name], mode, var),
                "color": GENERATOR_COLORS[generator],
            }
            for generator, name in samples.items()
        ]

    with PdfPages(output) as pdf:
        for mode in ["nocut", "nminus1"]:

            def draw(subfig, var):
                data = {"label": "Data", "hist": data_hists[mode][var], "data": True}
                draw_comparison(
                    subfig,
                    var,
                    generator_curves(mode, var) + [data],
                    data,
                    "MC / Data",
                    info_block(lepton, title, var, mode, gen_q2_min),
                    gen_q2_min,
                )

            save_page(pdf, RECO_VARS, draw)

        def draw_gen(subfig, var):
            curves = generator_curves("nocut", var)
            draw_comparison(
                subfig,
                var,
                curves,
                curves[0],
                f"X / {curves[0]['label']}",
                info_block(lepton, title, var, "gen", gen_q2_min),
                gen_q2_min,
            )

        save_page(pdf, GEN_VARS, draw_gen)
    print(f"Saved {output}")


def plot_perfile(flags, config, perfile, lepton, generator, name):
    sample = config["samples"][name]
    stats = perfile[name]
    output = os.path.join(flags.plot_folder, f"{lepton}_{generator}_perfile.pdf")

    fig = plt.figure(figsize=(20, 8 * 3))
    subfigs = fig.subfigures(3, 2).ravel()
    for subfig, (field, ylabel, err_field) in zip(subfigs, PERFILE_PANELS):
        ax = subfig.add_subplot()
        subfig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.12)
        kept = 0
        for run, info in sample["runs"].items():
            run = str(run)
            sel = stats["run"] == run
            if info.get("excluded", False):
                color, label = EXCLUDED_COLOR, f"Run {run} (excluded)"
                facecolor = "none"
            else:
                color = RUN_COLORS[kept % len(RUN_COLORS)]
                label, facecolor = f"Run {run}", color
                kept += 1
            ax.errorbar(
                stats["file_number"][sel],
                stats[field][sel],
                yerr=None if err_field is None else stats[err_field][sel],
                fmt="o",
                color=color,
                markerfacecolor=facecolor,
                markersize=7,
                label=label,
            )
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.45 * (high - low))
        ax.set_xlabel("File number")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=2)
        ax.text(
            0.03,
            0.96,
            f"{LEPTON_LABELS[lepton]}, {generator} per-file summary\n"
            f"No reco selection, MC: gen $Q^2>{config['mc_gen_q2_min']:g}$ GeV$^2$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
        )
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    flags = parse_arguments()
    with open(flags.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(flags.plot_folder, exist_ok=True)
    cache = flags.cache or os.path.join(flags.plot_folder, "precut_hists.npz")

    if flags.from_cache:
        hists, perfile = load_cache(cache)
    else:
        hists, perfile = collect(config, flags.max_files)
        save_cache(cache, hists, perfile)

    set_style()
    for lepton in config["data"]:
        samples = {
            sample["generator"]: name
            for name, sample in config["samples"].items()
            if sample["lepton"] == lepton
        }
        for generator, name in samples.items():
            plot_runs(flags, config, hists, perfile, lepton, generator, name)
            plot_perfile(flags, config, perfile, lepton, generator, name)
        plot_generators(flags, config, hists, lepton, samples)


if __name__ == "__main__":
    main()
