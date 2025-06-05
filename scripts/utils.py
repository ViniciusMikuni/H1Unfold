import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import uproot
import os
import json
import yaml
import options
import tensorflow as tf

line_style = {
    'Baseline': 'dotted',
    'Pre-trained': '-',
    'Baseline_Ens0': 'solid',
    'Baseline_Ens1': 'solid',
    'Baseline_Ens2': 'solid',
    'Baseline_Ens3': 'solid',
    'Baseline_Ens4': 'solid',
    'Pre-trained_Ens0': 'dotted',
    'Pre-trained_Ens1': 'dotted',
    'Pre-trained_Ens2': 'dotted',
    'Pre-trained_Ens3': 'dotted',
    'Pre-trained_Ens4': 'dotted',
    'Finetuned_Ens0': 'dashed',
    'Finetuned_Ens1': 'dashed',
    'Finetuned_Ens2': 'dashed',
    'Finetuned_Ens3': 'dashed',
    'Finetuned_Ens4': 'dashed',
    'data': 'dotted',
    'Rapgap reco': '-',
    'Rapgap gen': '-',
    'Rapgap_unfoldedAvg': "-",
}


colors = {
    "Baseline": "black",
    "Pre-trained": "black",
    "data": "black",
    "Rapgap reco": "#7570b3",
    "Rapgap gen": "darkorange",
    'Rapgap_unfoldedAvg': "blue",
}

# Generate colors for each label
color_labels = {
    'Baseline': 'Blues',
    'Pre-trained': 'Greens',
    'Finetuned': 'Reds'}

for label, cmap_name in color_labels.items():
    cmap = plt.get_cmap(cmap_name)
    color_array = cmap(np.linspace(0.5, 0.9, 5))
    for i, color in enumerate(color_array):
        colors[f'{label} E{i+1}'] = tuple(color)  # tuple better readability

event_names = {
    "0": r"$log(Q^2)$",
    "1": "y",
    "2": r"$e_{pT}$/Q",
    "3": r"$e_{\eta}$",
    "4": r"$e_{\phi}$",
}

jet_names = {
    "0": r"Jet $p_{T}$ [GeV]",
    "1": r"Jet $\eta$",
    "2": r"Jet $\phi$",
    "3": r"Jet E [GeV]",
    "4": r"$\mathrm{ln}(\lambda_1^1)$",
    "5": r"$\mathrm{ln}(\lambda_{1.5}^1)$",
    "6": r"$\mathrm{ln}(\lambda_{2.0}^1)$",
    "7": r"$p_\mathrm{T}\mathrm{D}$ $(\sqrt{\lambda_0^2})$",
}


particle_names = {
    "0": r"$\eta_p - \eta_e$",
    "1": r"$\phi_p - \phi_e - \pi$",
    "2": r"$log(p_{T})$",
    "3": r"$log(p_{T}/Q)$",
    "4": "log(E/Q)",
    "5": "log(E)",
    "6": r"$\sqrt{(\eta_p - \eta_e)^2 + (\phi_p - \phi_e)^2}$",
    "7": "Absolute Charge",
}


observable_names = {
    "jet_pt": r"$p_{T}^{jet}$ [GeV]",
    # 'jet_breit_pt': r'$p_{T}^{jet}$ [GeV] Breit frame',
    "jet_breit_pt": r"$p_{T}^{jet}$ [GeV]",
    "deltaphi": r"$\Delta\phi^{jet}$ [rad]",
    "jet_tau10": r"$\mathrm{ln}(\lambda_1^1)$",
    "zjet": r"$z^{jet}$",
    # 'zjet_breit':r'$z^{jet}$ Breit frame'
    "zjet_breit": r"$z^{jet}$ ",
}

dedicated_binning = {
    "jet_pt": np.logspace(np.log10(10), np.log10(100), 7),
    "jet_breit_pt": np.logspace(np.log10(10), np.log10(50), 7),
    "deltaphi": np.linspace(0, 1, 8),
    "jet_tau10": np.array(
        [-4.00, -3.15, -2.59, -2.18, -1.86, -1.58, -1.29, -1.05, -0.81, -0.61, 0.00]
    ),
    "zjet": np.linspace(0.2, 1, 11),
    "zjet_breit": np.linspace(0.2, 1, 11),
}


def get_log(var):
    if "pt" in var:
        return True, True
    # if 'pt' in var:
    #     return False, False
    if "deltaphi" in var:
        return True, True
    if "tau" in var:
        return False, False
    if "zjet" in var:
        return False, False
    if "eec" in var:
        return False, False
    if "theta" in var:
        return True, False
    else:
        print(f"ERROR: {var} not present!")


def get_ylim(var):
    if var == "jet_pt":
        # return 1e-5, 1
        return 1e-5, 12
    if var == "jet_breit_pt":
        return 1e-4, 10
    if "deltaphi" in var:
        return 1e-3, 350
    if "tau" in var:
        return 0, 1.2
    if var == "zjet":
        return 0, 10
    if var == "zjet_breit":
        return 0, 3
    if "eec" in var:
        # return 0, 0.35
        return 0, 4.7
    if "theta" in var:
        return 0, 2.5
    else:
        print("ERROR")


class ObservableInfo:
    def __init__(self, var):
        self.var = var
        self.name = observable_names[var]
        self.binning = dedicated_binning[var]
        self.logx, self.logy = get_log(var)
        self.ylow, self.yhigh = get_ylim(var)


class ScalarFormatterClass(mtick.ScalarFormatter):
    # https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def LoadFromROOT(file_name, var_name, q2_bin=0):
    with uproot.open(file_name) as f:
        if b"DIS_JetSubs;1" in f.keys():
            # Predictions from rivet
            hist = f[b"DIS_JetSubs;1"]
        else:
            hist = f
        if q2_bin == 0:
            var, bins = hist[var_name].numpy()
            # print(bins)
        else:  # 2D slice of histogram
            var = hist[var_name + "2D"].numpy()[0][:, q2_bin - 1]
            bins = hist[var_name + "2D"].numpy()[1][0][0]

        norm = 0
        for iv, val in enumerate(var):
            norm += val * abs(bins[iv + 1] - bins[iv])
    return var


def SetStyle():
    from matplotlib import rc

    rc("text", usetex=True)

    import matplotlib as mpl

    rc("font", family="serif")
    rc("font", size=22)
    rc("xtick", labelsize=15)
    rc("ytick", labelsize=14)
    rc("legend", fontsize=15)

    # #
    mpl.rcParams.update({"font.size": 19})
    # mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams.update({"xtick.labelsize": 18})
    mpl.rcParams.update({"ytick.labelsize": 18})
    mpl.rcParams.update({"axes.labelsize": 18})
    mpl.rcParams.update({"legend.frameon": False})

    import mplhep as hep

    hep.set_style(hep.style.CMS)
    hep.style.use("CMS")


# def SetGrid(npanels=2):
#     fig = plt.figure(figsize=(9, 9))
#     if npanels ==2:
#         gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
#         gs.update(wspace=0.025, hspace=0.1)
#     elif npanels ==3:
#         gs = gridspec.GridSpec(3, 1, height_ratios=[4,1,1])
#         gs.update(wspace=0.025, hspace=0.1)
#     else:
#         gs = gridspec.GridSpec(1, 1)
#     return fig,gs


def SetGrid(ratio=True):
    # fig = plt.figure(figsize=(9, 9))
    fig = plt.figure(figsize=(12, 12))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig, gs


def FormatFig(xlabel, ylabel, ax0, xpos=0.8, ypos=0.95):
    # Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update)
    if "Breit" in xlabel:
        xlabel_strip = xlabel.replace(" Breit frame", "")
        ylabel_strip = ylabel.replace(" Breit frame", "")
        ax0.set_xlabel(xlabel_strip, fontsize=24)
        ax0.set_ylabel(ylabel_strip)

    else:
        ax0.set_xlabel(xlabel, fontsize=24)
        ax0.set_ylabel(ylabel)

    text = r"$\bf{H1 Preliminary}$"
    WriteText(xpos, ypos, text, ax0, align="left")

    second_text = r"$\mathrm{Unfolded\ single\ particle\ dataset}$"
    WriteText(xpos, ypos - 0.06, second_text, ax0, fontsize=18, align="left")

    phasespace_text = r"$Q^2>150~\mathrm{GeV}^2, 0.2<y<0.7$"
    if "Breit frame" in xlabel.strip():
        frame_text = "Breit Frame"
        phasespace_text += "\n" + r"$p_T^{jet} > 5 GeV\ k_{T}, R = 1.0$"
    else:
        frame_text = "Lab Frame"
        phasespace_text += "\n" + r"$p_T^{jet} > 10 GeV\ k_{T}, R = 1.0$"

    WriteText(xpos, ypos - 0.15, frame_text, ax0, fontsize=18, align="left")
    WriteText(xpos, ypos - 0.25, phasespace_text, ax0, fontsize=18, align="left")


def WriteText(xpos, ypos, text, ax0, fontsize=25, align="center"):
    plt.text(
        xpos,
        ypos,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        # fontweight='bold',
        transform=ax0.transAxes,
        fontsize=fontsize,
    )


def LoadJson(file_name, base_path="../JSON"):
    JSONPATH = os.path.join(base_path, file_name)
    return yaml.safe_load(open(JSONPATH))


def SaveJson(data, file_name, base_path="../JSON"):
    JSONPATH = os.path.join(base_path, file_name)
    with open(JSONPATH, "w") as f:
        json.dump(data, f, indent=4)


def make_error_boxes(
    ax, xdata, ydata, xerror, yerror, facecolor="r", edgecolor="None", alpha=0.5
):
    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt="None", ecolor="k")


def PlotRoutine(
    feed_dict, xlabel="", ylabel="", reference_name="gen", plot_ratio=False
):
    if plot_ratio:
        assert reference_name in feed_dict.keys(), (
            "ERROR: Don't know the reference distribution"
        )

    fig, gs = SetGrid(ratio=plot_ratio)
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1], sharex=ax0)

    for ip, plot in enumerate(feed_dict.keys()):
        ax0.plot(
            feed_dict[plot],
            label=plot,
            linewidth=2,
            linestyle=line_style[plot],
            color=colors[plot],
        )
        if reference_name != plot and plot_ratio:
            ratio = 100 * np.divide(
                feed_dict[reference_name] - feed_dict[plot], feed_dict[reference_name]
            )
            ax1.plot(ratio, color=colors[plot], linewidth=2, linestyle=line_style[plot])

    ax0.legend(loc="best", fontsize=16, ncol=1)
    if plot_ratio:
        FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
        plt.ylabel("Difference. (%)")
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color="r", linestyle="--", linewidth=1)
        plt.axhline(y=10, color="r", linestyle="--", linewidth=1)
        plt.axhline(y=-10, color="r", linestyle="--", linewidth=1)
        plt.ylim([-100, 100])

    else:
        FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

    return fig, ax0


def HistRoutine(
    feed_dict,
    xlabel="",
    ylabel="",
    reference_name="data",
    logy=False,
    logx=False,
    binning=None,
    label_loc="best",
    plot_ratio=True,
    weights=None,
    uncertainty=None,
    stat_uncertainty=None,
    axes=None,
    save_str="",
):
    """
    Generate a histogram plot with optional ratio and uncertainties.

    Args:
        feed_dict (dict): Dictionary containing data to be plotted.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        reference_name (str): Key in feed_dict used as the reference distribution.
        logy (bool): Whether to use a logarithmic scale on the y-axis.
        logx (bool): Whether to use a logarithmic scale on the x-axis.
        binning (array-like): Bin edges for the histograms.
        label_loc (str): Location of the legend.
        plot_ratio (bool): Whether to plot the ratio to the reference distribution.
        weights (dict): Optional weights for each distribution in feed_dict.
        uncertainty (array-like): Optional uncertainties for the ratio plot.

    Returns:
        fig, ax0: The generated figure and main axis.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import SetGrid, FormatFig

    assert reference_name in feed_dict, (
        "ERROR: Reference distribution not found in feed_dict."
    )

    # Default styles for plots
    ref_plot_style = {"histtype": "stepfilled", "alpha": 0.2}
    data_plot_style = {"histtype": "step", "alpha": 0.2}
    other_plot_style = {"histtype": "step", "linewidth": 2}

    # Set up the figure and axes
    if axes is not None:
        ax0 = axes[0]
        fig = axes[-1]
    else:
        fig,gs = SetGrid(ratio=plot_ratio)
        ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        if axes is not None:
            ax1 = axes[1]
        else:
            ax1 = plt.subplot(gs[1],sharex=ax0)


    # Define binning if not provided
    if binning is None:
        binning = np.linspace(
            np.quantile(feed_dict[reference_name], 0.01),
            np.quantile(feed_dict[reference_name], 0.99),
            50,
        )

    xaxis = 0.5 * (binning[:-1] + binning[1:])  # Bin centers

    # Compute reference histogram
    ref_weights = weights[reference_name] if weights else None
    reference_hist, _ = np.histogram(
        feed_dict[reference_name], bins=binning, density=True, weights=ref_weights
    )

    max_y = 0
    # Plot each distribution
    ens0_flag = False #avoid saving reference hist constantly in closer

    for plot_name, data in feed_dict.items():

        # Save to npy files for comparison
        if '0' in plot_name:
            ens0_flag = True
            plot_vals = np.array([xaxis, reference_hist])
            np.save(f'../plots/{plot_name}{save_str}_plot_vals.npy', plot_vals)
        if not ens0_flag and (plot_name=='Djangoh'):
            continue
        if 'Avg' in plot_name:
            plot_vals = np.array([xaxis, reference_hist])
            np.save(f'../plots/{plot_name}{save_str}_plot_vals.npy', plot_vals)

        #Set Plot Style
        # plot_style = ref_plot_style if plot_name == reference_name else other_plot_style
        if "data" in plot_name.lower():
            plot_style = data_plot_style

        elif "Rapgap_closure" in plot_name:
            plot_style = ref_plot_style
        else:
            plot_style = other_plot_style

        plot_weights = weights[plot_name] if weights else None

        # if plot_name == reference_name:
        if "data" in plot_name.lower():
            dist, _ = np.histogram(
                data, bins=binning, density=True, weights=plot_weights
            )
            bin_centers = (binning[:-1] + binning[1:]) / 2
            errors_low = dist * (1 - stat_uncertainty)
            errors_high = dist * (1 + stat_uncertainty)
            errors = [dist - errors_low, errors_high - dist]  # Asymmetric error bars
            ax0.errorbar(
                bin_centers,
                dist,
                yerr=errors,
                fmt="o",
                color="black",
                label=options.name_translate[plot_name],
                markersize=8,
            )
        else:
            dist, _, _ = ax0.hist(
                data,
                bins=binning,
                density=True,
                weights=plot_weights,
                label=options.name_translate[plot_name],
                color=options.colors[plot_name],
                **plot_style,
            )

        max_y = max(max_y, np.max(dist))

        # Plot ratio if applicable
        if plot_ratio and plot_name != reference_name:
            ratio = np.ma.divide(dist, reference_hist).filled(0)

            bin_edges = np.zeros(len(binning))
            for i in range(len(binning)):
                bin_edges[i] = binning[i]

            # Create extended ratio array for steps-post style
            extended_ratio = np.zeros(len(bin_edges))
            for i in range(len(ratio)):
                extended_ratio[i] = ratio[i]

            ax1.plot(
                bin_edges,
                extended_ratio,
                color=options.colors[plot_name],
                drawstyle="steps-post",
                linestyle="-",
                lw=3,
                ms=10,
                markerfacecolor="none",
                markeredgewidth=3,
            )

            # Add uncertainties
            if uncertainty is not None:
                for ibin in range(len(binning) - 1):
                    xup = binning[ibin + 1]
                    xlow = binning[ibin]
                    ax1.fill_between(
                        np.array([xlow, xup]),
                        1.0 + uncertainty[ibin],
                        1.0 - uncertainty[ibin],
                        alpha=0.1,
                        color="k",
                    )
                    # Overlay hatch using bar
                    ax1.bar(
                        (xlow + xup) / 2,
                        2 * uncertainty[ibin],
                        width=(xup - xlow),
                        bottom=1.0 - uncertainty[ibin],
                        hatch="//",
                        color="none",
                        edgecolor="grey",
                        label="Systematic Uncertainty",
                    )

    grey_patch = Patch(
        facecolor="grey",
        alpha=0.5,
        hatch="//",
        edgecolor="black",
        label="Systematic Uncertainty",
    )

    ## draw data points at 1 with stat uncertainties on the bottom panel
    if "data" in reference_name.lower():
        ax1.errorbar(
            xaxis,
            np.ones_like(xaxis),
            yerr=stat_uncertainty,
            marker="o",
            linestyle="None",
            markersize=8,
            color="black",
            capsize=3,
        )
    # Adjust y-axis scale
    if logy:
        ax0.set_yscale("log")
        ax0.set_ylim(1e-5, 10 * max_y)
    else:
        ax0.set_ylim(0, 1.3 * max_y)

    # Adjust x-axis scale
    if logx:
        ax0.set_xscale("log")
        if plot_ratio:
            ax1.set_xscale("log")

    # Add legend and format axes
    # ax0.legend(loc=label_loc, fontsize=16, ncol=2)
    handles, labels = ax0.get_legend_handles_labels()
    handles.append(grey_patch)
    labels.append("Systematic Uncertainty")
    ax0.legend(handles, labels, loc=label_loc, fontsize=20, ncol=1)

    # ax0.legend(loc=label_loc, fontsize=20, ncol=1)

    if plot_ratio:
        FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)
        ax1.set_ylabel("Pred./Ref.")
        ax1.axhline(y=1.0, color="r", linestyle="-", linewidth=1)
        ax1.set_ylim([0.5, 1.5])
        if "Breit" in xlabel:
            xlabel_strip = xlabel.replace(" Breit frame", "")
            ax1.set_xlabel(xlabel_strip)
        else:
            ax1.set_xlabel(xlabel)
    else:
        FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

    return fig, ax0


def FormatFigPart(xlabel, ylabel, ax0, xpos=0.8, ypos=0.95):
    # Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update)
    if "Breit" in xlabel:
        xlabel_strip = xlabel.replace(" Breit frame", "")
        ylabel_strip = ylabel.replace(" Breit frame", "")
        ax0.set_xlabel(xlabel_strip, fontsize=24)
        ax0.set_ylabel(ylabel_strip)

    else:
        ax0.set_xlabel(xlabel, fontsize=24)
        ax0.set_ylabel(ylabel)

    text = r"$\bf{H1 Preliminary}$"
    WriteText(xpos, ypos, text, ax0, align="left")

    second_text = r"$\mathrm{Unfolded\ single\ particle\ dataset}$"
    WriteText(xpos, ypos - 0.06, second_text, ax0, fontsize=18, align="left")

    phasespace_text = r"$Q^2>150~\mathrm{GeV}^2, 0.2<y<0.7$"
    if "Breit frame" in xlabel.strip():
        frame_text = "Breit Frame"
    else:
        frame_text = "Lab Frame"

    WriteText(xpos, ypos - 0.15, frame_text, ax0, fontsize=18, align="left")
    WriteText(xpos, ypos - 0.25, phasespace_text, ax0, fontsize=18, align="left")


def HistRoutinePart(
    feed_dict,
    xlabel="",
    ylabel="",
    reference_name="data",
    logy=False,
    logx=False,
    binning=None,
    label_loc="best",
    plot_ratio=True,
    weights=None,
    uncertainty=None,
    stat_uncertainty=None,
    axes=None,
    save_str="",
):
    """
    Generate a histogram plot with optional ratio and uncertainties.

    Args:
        feed_dict (dict): Dictionary containing data to be plotted.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        reference_name (str): Key in feed_dict used as the reference distribution.
        logy (bool): Whether to use a logarithmic scale on the y-axis.
        logx (bool): Whether to use a logarithmic scale on the x-axis.
        binning (array-like): Bin edges for the histograms.
        label_loc (str): Location of the legend.
        plot_ratio (bool): Whether to plot the ratio to the reference distribution.
        weights (dict): Optional weights for each distribution in feed_dict.
        uncertainty (array-like): Optional uncertainties for the ratio plot.

    Returns:
        fig, ax0: The generated figure and main axis.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import SetGrid, FormatFigPart

    assert reference_name in feed_dict, (
        "ERROR: Reference distribution not found in feed_dict."
    )

    # Default styles for plots
    ref_plot_style = {"histtype": "stepfilled", "alpha": 0.2}
    data_plot_style = {"histtype": "step", "alpha": 0.2}
    other_plot_style = {"histtype": "step", "linewidth": 2}

    # Set up the figure and axes
    if axes is not None:
        ax0 = axes[0]
        fig = axes[-1]
    else:
        fig,gs = SetGrid(ratio=plot_ratio)
        ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        if axes is not None:
            ax1 = axes[1]
        else:
            ax1 = plt.subplot(gs[1],sharex=ax0)

    # Define binning if not provided
    if binning is None:
        binning = np.linspace(
            np.quantile(feed_dict[reference_name], 0.01),
            np.quantile(feed_dict[reference_name], 0.99),
            50,
        )

    xaxis = 0.5 * (binning[:-1] + binning[1:])  # Bin centers

    # Compute reference histogram
    ref_weights = weights[reference_name] if weights else None
    reference_hist, _ = np.histogram(
        feed_dict[reference_name], bins=binning, density=True, weights=ref_weights
    )

    max_y = 0
    # Plot each distribution

    ens0_flag = False

    for plot_name, data in feed_dict.items():

        # Save to npy files for comparison
        if '0' in plot_name:
            ens0_flag = True
            plot_vals = np.array([xaxis, reference_hist])
            np.save(f'../plots/{plot_name}{save_str}_plot_vals.npy', plot_vals)
        if not ens0_flag and (plot_name=='Djangoh'):
            continue
        if 'Avg' in plot_name:
            plot_vals = np.array([xaxis, reference_hist])
            np.save(f'../plots/{plot_name}{save_str}_plot_vals.npy', plot_vals)


        #Set Plot Style
        if "data" in plot_name.lower():
            plot_style = data_plot_style

        elif "Rapgap_closure" in plot_name:
            plot_style = ref_plot_style
        else:
            plot_style = other_plot_style

        plot_weights = weights[plot_name] if weights else None

        # if plot_name == reference_name:
        if "data" in plot_name.lower():
            dist, _ = np.histogram(
                data, bins=binning, density=True, weights=plot_weights
            )
            bin_centers = (binning[:-1] + binning[1:]) / 2
            errors_low = dist * (1 - stat_uncertainty)
            errors_high = dist * (1 + stat_uncertainty)
            errors = [dist - errors_low, errors_high - dist]  # Asymmetric error bars
            ax0.errorbar(
                bin_centers,
                dist,
                yerr=errors,
                fmt="o",
                color="black",
                label=options.name_translate[plot_name],
                markersize=8,
            )
        else:
            dist, _, _ = ax0.hist(
                data,
                bins=binning,
                density=True,
                weights=plot_weights,
                label=options.name_translate[plot_name],
                color=options.colors[plot_name],
                **plot_style,
            )

        max_y = max(max_y, np.max(dist))

        # Plot ratio if applicable
        if plot_ratio and plot_name != reference_name:
            ratio = np.ma.divide(dist, reference_hist).filled(0)

            bin_edges = np.zeros(len(binning))
            for i in range(len(binning)):
                bin_edges[i] = binning[i]

            # Create extended ratio array for steps-post style
            extended_ratio = np.zeros(len(bin_edges))
            for i in range(len(ratio)):
                extended_ratio[i] = ratio[i]

            ax1.plot(
                bin_edges,
                extended_ratio,
                color=options.colors[plot_name],
                drawstyle="steps-post",
                linestyle="-",
                lw=3,
                ms=10,
                markerfacecolor="none",
                markeredgewidth=3,
            )

            # Add uncertainties
            if uncertainty is not None:
                for ibin in range(len(binning) - 1):
                    xup = binning[ibin + 1]
                    xlow = binning[ibin]
                    ax1.fill_between(
                        np.array([xlow, xup]),
                        1.0 + uncertainty[ibin],
                        1.0 - uncertainty[ibin],
                        alpha=0.1,
                        color="k",
                    )
                    # Overlay hatch using bar
                    ax1.bar(
                        (xlow + xup) / 2,
                        2 * uncertainty[ibin],
                        width=(xup - xlow),
                        bottom=1.0 - uncertainty[ibin],
                        hatch="//",
                        color="none",
                        edgecolor="grey",
                        label="Systematic Uncertainty",
                    )

    grey_patch = Patch(
        facecolor="grey",
        alpha=0.5,
        hatch="//",
        edgecolor="black",
        label="Systematic Uncertainty",
    )

    ## draw data points at 1 with stat uncertainties on the bottom panel
    if "data" in reference_name.lower():
        ax1.errorbar(
            xaxis,
            np.ones_like(xaxis),
            yerr=stat_uncertainty,
            marker="o",
            linestyle="None",
            markersize=8,
            color="black",
            capsize=3,
        )
    # Adjust y-axis scale
    if logy:
        ax0.set_yscale("log")
        ax0.set_ylim(1e-5, 10 * max_y)
    else:
        ax0.set_ylim(0, 1.3 * max_y)

    # Adjust x-axis scale
    if logx:
        ax0.set_xscale("log")
        if plot_ratio:
            ax1.set_xscale("log")

    # Add legend and format axes
    # ax0.legend(loc=label_loc, fontsize=16, ncol=2)
    handles, labels = ax0.get_legend_handles_labels()
    handles.append(grey_patch)
    labels.append("Systematic Uncertainty")
    ax0.legend(handles, labels, loc=label_loc, fontsize=20, ncol=1)

    # ax0.legend(loc=label_loc, fontsize=20, ncol=1)

    if plot_ratio:
        FormatFigPart(xlabel=xlabel, ylabel=ylabel, ax0=ax0)
        ax1.set_ylabel("Pred./Ref.")
        ax1.axhline(y=1.0, color="r", linestyle="-", linewidth=1)
        ax1.set_ylim([0.5, 1.5])
        if "Breit" in xlabel:
            xlabel_strip = xlabel.replace(" Breit frame", "")
            ax1.set_xlabel(xlabel_strip)
        else:
            ax1.set_xlabel(xlabel)
    else:
        FormatFigPart(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

    return fig, ax0


def setup_gpus(local_rank):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], "GPU")
