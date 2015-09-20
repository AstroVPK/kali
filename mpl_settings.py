import matplotlib as mpl

def set_plot_params(fontsize=20):
    #make ticks longer and thicker
    mpl.rcParams['xtick.major.size']=8
    mpl.rcParams['xtick.minor.size']=6
    mpl.rcParams['xtick.major.width']=2
    mpl.rcParams['xtick.minor.width']=2 
    mpl.rcParams['ytick.major.size']=8
    mpl.rcParams['ytick.minor.size']=6
    mpl.rcParams['ytick.major.width']=2
    mpl.rcParams['ytick.minor.width']=2
    #make border thicker
    mpl.rcParams['axes.linewidth']=2
    #make plotted lines thicker
    mpl.rcParams['lines.linewidth']=2
    #make fonts bigger
    mpl.rcParams['xtick.labelsize']=fontsize
    mpl.rcParams['ytick.labelsize']=fontsize
    mpl.rcParams['legend.fontsize']=fontsize
    mpl.rcParams['axes.titlesize']=fontsize
    mpl.rcParams['axes.labelsize']=fontsize
    #save figure settings
    mpl.rcParams['savefig.bbox']='tight'
