import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import uproot
import os
import json,yaml
import options
import tensorflow as tf

line_style = {
    'Baseline':'dotted',
    'Pre-trained':'-',
    'data':'dotted',
    'Rapgap reco':'-',
    'Rapgap gen':'-',
}

colors = {
    'Baseline':'black',
    'Pre-trained':'black',    
    'data':'black',
    'Rapgap reco':'#7570b3',
    'Rapgap gen':'darkorange',
}


event_names = {
    '0': 'log(Q)',
    '1': 'y',
    '2': r'$e_{pT}$/Q',
    '3': r'$e_{\eta}$',
    '4': r'$e_{\phi}$',
    }
particle_names = {
    '0': r'$\eta_e - \eta_p$',
    '1': r'$\phi_e - \phi_p - \pi$',
    '2': r'$log(p_{T})$',
    '3': r'$log(p_{T}/Q)$',
    '4': 'log(E/Q)',
    '5': 'log(E)',
    '6':r'$\sqrt{(\eta_e - \eta_p)^2 + (\phi_e - \phi_p)^2}$',
    '7': 'Charge',
    }


class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"

def LoadFromROOT(file_name,var_name,q2_bin=0):
    with uproot.open(file_name) as f:
        if b'DIS_JetSubs;1' in f.keys():
            #Predictions from rivet
            hist = f[b'DIS_JetSubs;1']            
        else:
            hist = f
        if q2_bin ==0:
            var, bins =  hist[var_name].numpy()
            #print(bins)
        else: #2D slice of histogram
            var =  hist[var_name+"2D"].numpy()[0][:,q2_bin-1]
            bins = hist[var_name+"2D"].numpy()[1][0][0]
            
        norm = 0
        for iv, val in enumerate(var):
            norm += val*abs(bins[iv+1]-bins[iv])
    return var
        
def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=14)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    
    import matplotlib.pyplot as plt
    # import mplhep as hep
    # hep.set_style(hep.style.CMS)
    # hep.style.use("CMS") 

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
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



def FormatFig(xlabel,ylabel,ax0,xpos=0.9,ypos=0.9):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=24)
    ax0.set_ylabel(ylabel)
        

    # text = r'$\bf{H1}$'
    # WriteText(xpos,ypos,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             #fontweight='bold',
             transform = ax0.transAxes, fontsize=25)


def LoadJson(file_name,base_path='../JSON'):
    JSONPATH = os.path.join(base_path,file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(data,file_name,base_path='../JSON'):
    JSONPATH = os.path.join(base_path,file_name)
    with open(JSONPATH, 'w') as f:
        json.dump(data, f, indent=4)



def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None', ecolor='k')

def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen',plot_ratio = False):
    if plot_ratio:
        assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        ax0.plot(feed_dict[plot],label=plot,linewidth=2,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot and plot_ratio:
            ratio = 100*np.divide(feed_dict[reference_name] -feed_dict[plot],feed_dict[reference_name])
            ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])

    ax0.legend(loc='best',fontsize=16,ncol=1)            
    if plot_ratio:        
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])

    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)    
        
    return fig,ax0


def HistRoutine(feed_dict,
                xlabel='',
                ylabel='',
                reference_name='data',
                logy=False,
                logx = False,
                binning=None,
                label_loc='best',
                plot_ratio=True,
                weights=None,
                uncertainty=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.01),np.quantile(feed_dict[reference_name],0.99),50)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]

    if weights is not None:
        reference_hist,_ = np.histogram(feed_dict[reference_name],weights=weights[reference_name],bins=binning,density=True)
    else:
        reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,color=options.colors[plot],density=True,weights=weights[plot],**plot_style)
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,color=options.colors[plot],density=True,**plot_style)

        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,color=options.colors[plot],
                         marker=options.markers[plot],ms=10,
                         lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')
        ax0.set_ylim(1e-5,10*maxy)
    else:
        ax0.set_ylim(0,1.3*maxy)

    if logx:
        #ax0.set_xscale('log')
        ax1.set_xscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Data')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        plt.ylim([0.5,1.5])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0


def setup_gpus(local_rank):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')
