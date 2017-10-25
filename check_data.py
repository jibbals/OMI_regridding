'''
File to check data files created throughout the calculation process.
'''
## Modules
# these 2 lines make plots not show up ( can save them as output faster )
# use needs to be called before anythin tries to import matplotlib modules
#import matplotlib
#matplotlib.use('Agg')

# import my stuff:
from utilities import fio
from utilities import plotting as pp
from utilities import utilities as util
from utilities import GMAO
from classes.omhchorp import omhchorp

# maths and dates
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt


##############################
########## GLOBALS ###########
##############################

figpath='./Figs/Checks'


##############################
########## FUNCTIONS #########
##############################

def check_swaths(date=datetime(2005,1,1),suffix=None):
    '''
    how many entries/day, good entries/day, etc.
    '''
    day=datetime(2005,1,1)
    paths=fio.determine_filepath(day)
    print(day)
    print(paths)
    omhcho=fio.read_omhcho(paths[0])
    h=omhcho['HCHO']
    entries=np.cumprod(np.shape(h))[-1]
    print('%d entries in first swath'% entries)
    print("%d good entries in first swath"%np.sum(~np.isnan(omhcho['HCHO'])))

def check_omhchorp(date=datetime(2005,1,1),suffix='',ignorePP=True):
    ''' '''
    dstr=date.strftime("%Y%m%d") # yyyymmdd
    mstr=date.strftime("%Y, %B") # yyyy, Month
    pname="%s/omhchorp_%s_%s.png"%(figpath,dstr,suffix)

    # read month average
    day0=datetime(date.year,date.month,1)
    dayn=util.last_day(date)
    om=omhchorp(day0=day0,dayn=dayn,ignorePP=ignorePP)
    data_day=om.time_averaged(day0=date,keys=['VCC','gridentries'])
    data_month=om.time_averaged(day0=date,month=True,keys=['VCC','gridentries'])

    lats=om.lats; lons=om.lons

    #print(np.shape(data_day['gridentries']))
    #print(type(data_day['gridentries']))
    #print(type(data_day['gridentries'][0,0]))
    #print(data_day['gridentries'][10:20,10:20])

    #print(np.shape(data_day['VCC']))
    #print(type(data_day['VCC']))
    #print(type(data_day['VCC'][0,0]))
    #print(data_day['VCC'][10:20,10:20])

    # plot map of day hcho columns
    # next to map of month averaged hcho columns
    f=plt.figure(figsize=[14,14])
    plt.subplot(221)
    pp.createmap(data_day['VCC'],lats,lons,linear=True,clabel='molec/cm2')
    plt.title("VCC %s"%dstr)
    plt.subplot(222)
    pp.createmap(data_month['VCC'],lats,lons,linear=True,clabel='molec/cm2')
    plt.title("VCC %s"%mstr)

    # plot entry counts
    plt.subplot(223)

    pp.createmap(data_day['gridentries'],lats,lons,linear=True)
    plt.title("omi pixels %s"%dstr)
    plt.subplot(224)
    pp.createmap(data_month['gridentries'],lats,lons,linear=True)
    plt.title("omi pixels %s"%mstr)

    plt.savefig(pname)
    print("FIGURE SAVED: %s"%pname)


def check_HEMCO_restarts(date=datetime(2005,1,1),suffix=''):
    '''
        Check how different the hemco restarts are between UCX and tropchem
    '''
    dstr=date.strftime("%Y%m%d")
    pname='%s/HEMCO_restart_%s_%s.png'%(figpath,dstr,suffix)
    # Read the restart files:
    fpat='Data/GC_Output/%s/restarts/HEMCO_restart.%s0000.nc'
    fucx=fpat%('UCX_geos5_2x25',dstr)
    ftrp=fpat%('geos5_2x25_tropchem',dstr)
    ucx=fio.read_netcdf(fucx)
    trp=fio.read_netcdf(ftrp)

    # Try plotting a table for easy reading:
    col_labels= ('UCX','Trop')
    row_labels=[]
    row_colour=[]
    tabledat=[]
    cmap=matplotlib.cm.get_cmap("RdBu")
    blue=cmap(0.9);red=cmap(0.2)
    for name in ucx.keys():
        ucx_mean=np.nanmean(ucx[name])
        if name in trp.keys():
            trp_mean=np.nanmean(trp[name])
            row_labels.append(name)
            equal=np.isclose(trp_mean,ucx_mean)
            row_colour.append([red,blue][equal])
            tabledat.append(["%.2e"%ucx_mean,"%.2e"%trp_mean])

    # Figure stuff:
    fig,ax=plt.subplots()
    plt.title("UCX vs Tropchem HEMCO Restarts %s"%dstr)

    table=plt.table(cellText=tabledat, rowLabels=row_labels, colLabels=col_labels,
                    rowColours=row_colour,  loc='center')

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    #ax.axis('tight')
    # change cellwidths
    fig.tight_layout()
    fig.subplots_adjust(left=0.3)
    plt.savefig(pname)
    print("FIGURE SAVED: %s "%pname)

    return None




def plot_swaths(day):
    '''  Plot a swath/day/8day picture '''
    print("TBA")

def grid_comparison():
    '''
    '''

    # Get resolution for GC and satellite:
    GC_y   = GMAO.lats_e
    GC_x   = GMAO.lons_e
    GC_xy0 = (GC_x[2],GC_y[2])
    GC_res = (GC_x[3]-GC_x[2],GC_y[3]-GC_y[2])

    #S=omhchorp(datetime(2005,1,1),ignorePP=True)
    #S_x    = S.lons_e
    #S_y    = S.lats_e
    #S_xy0  = GC_xy0#(S_x[1],S_y[1])
    #S_res  = (0.3125,0.25) #(S_x[3]-S_x[2],S_y[3]-S_y[2])

    # Make bluemarble display map:
    m=pp.displaymap()

    # Add GC resolution and then my satellite resolution
    #m, xy0=(-181.25,-89.), xyres=(2.5,2.), color='k', linewidth=1.0, dashes=[1000,1], labels=[0,0,0,0]
    pp.add_grid_to_map(m,xy0=GC_xy0,xyres=GC_res,color='white',linewidth=1,labels=[0,0,0,0])
    #pp.add_grid_to_map(m,xy0=S_xy0,xyres=S_res,color='orange',linewidth=1,labels=[0,0,0,0])

    pname=figpath+'/GridSizes.png'
    plt.title("GEOS-Chem Resolution")
    plt.savefig(pname)
    print("Saved "+pname)

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running check_data.py")

    grid_comparison()
    #for date in [datetime(2005,1,1),datetime(2005,2,1),datetime(2005,7,1),datetime(2006,1,1),]:
    #    check_HEMCO_restarts(date=date)

    #for date in [datetime(2005,1,1),datetime(2005,1,2), datetime(2005,3,1)]:
    #    check_omhchorp(date)


