###
# Created 31/5/18 by jwg366
#   Loook at RSC and what affect it has
###

## Modules

# my file reading and writing module
from utilities import fio
import reprocess
from utilities import plotting as pp
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes.omhchorp import omhchorp as omrp
from classes.GC_class import GC_tavg, GC_sat

# General use modules
import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy
from datetime import datetime#, timedelta

import matplotlib.pyplot as plt

#from matplotlib.colors import LogNorm # for lognormal colour bar
#from matplotlib.ticker import FormatStrFormatter # tick formats
#import matplotlib.patches as mpatches
#import seaborn # kdeplots
#import random # random number generation
#import timeit

#######################
##      GLOBALS
#######################

# STRINGS FOR THINGS
Ohcho='$\Omega_{HCHO}$' # VC of HCHO
Ovc='\Omega'
Og='$\Omega_{G}$'       # GEOS-Chem VC
Ogc='$\Omega_{GC}$'     # GC VCC
Op='$\Omega_{P}$'       # Palmer VC
Opp='$\Omega_{PC}$'     # Palmer VCC
Oo='$\Omega_{O}$'       # OMI VC
Ooc='$\Omega_{OC}$'     # OMI VCC







########################
###     TESTS/Plots
########################
def new_vs_old(month=datetime(2005,1,1)):
    '''
        VCC_OMI vs VCC_OMI_newres
    '''
    # plot name
    ymstr=month.strftime('%Y%m')
    pname='Figs/RSC_new_vs_old_%s.png'%ymstr

    # read in VCC_OMI and VCC_OMI_newrsc for month
    d0=util.first_day(month)
    dn=util.last_day(month)
    titles=['VCC_OMI','VCC_OMI_newrsc']
    om=omrp(d0,dayn=dn, keylist=titles)

    # Compare old vs new avg for one month
    datas=[np.nanmean(getattr(om,title),axis=0) for title in titles]
    lats=[om.lats,om.lats]
    lons=[om.lons,om.lons]
    pp.compare_maps(datas,lats,lons,pname=pname, linear=True,
                    titles=titles,rmin=-50,rmax=50,
                    suptitle='RSC effect on OMI VCC (%s)'%ymstr)

def Summary_RSC(month=datetime(2005,1,1)):
    '''
    Print and plot a summary of the effect of our remote sector correction
    Plot 1: Reference Correction
        showing VCs before and after correction, with rectangle around region

    Plot 2: OMI Sensors difference from apriori over RSC
        Contourf of RSC correction function [sensor(X) vs latitude(Y)]

    '''
    date=datetime(month.year,month.month,1)
    yms=month.strftime('%Y%m')
    dayn=util.last_day(month)
    cbarname='plasma'

    # read reprocessed data
    dat=omrp(day0=date,dayn=dayn)
    VCC_GC=np.nanmean(dat.VCC_GC,axis=0) # avg over month
    VC_GC =np.nanmean(dat.VC_GC, axis=0)
    lats,lons = dat.lats, dat.lons
    # read geos chem data
    gcdat=GC_sat(day0=date,dayN=dayn)
    gchcho=np.nanmean(gcdat.O_hcho,axis=0) # molec/cm2
    gclats,gclons=gcdat.lats,gcdat.lons
    # plot 1) showing VCs before and after correction
    vmin,vmax=1e14,4e16
    f=plt.figure(0,figsize=(17,16))
    lims=(-60,30,45,160)
    lims2=(-65,-185,65,-115)
    for i,arr in enumerate([VC_GC,VCC_GC]):
        #2 rows, 6 columns, this plot will take up space of subplots 0,1,2 and 3,4,5
        plt.subplot2grid((2, 6), (0, 3*i), colspan=3)
        m,cs,cb=pp.createmap(arr,lats,lons,
                             colorbar=False,vmin=vmin,vmax=vmax,
                             region=lims)
        plt.title([Og,Ogc][i],fontsize=25)
        m.drawparallels([-40,0,40],labels=[1-i,0,0,0],linewidth=0.0)

    # print some stats of changes
    diffs=dat.VCC_GC-dat.VC_GC
    print ("Mean difference VC - VCC:%7.5e "%np.nanmean(diffs))
    print ("%7.2f%%"%(np.nanmean(diffs)*100/np.nanmean(dat.VC_GC)))
    print ("std VC - VCC:%7.5e "%np.nanstd(diffs))

    # plot c) RSC by sensor and latitude
    plt.subplot2grid((2, 6), (1, 0), colspan=2)
    RSC_GC=np.nanmean(dat.RSC[:,:,:,1],axis=0) # average over time
    norm = plt.cm.colors.Normalize(vmin=-4e14,vmax=4e14)
    cmap=plt.cm.coolwarm

    cp=plt.contourf(np.arange(1,60.1,1),dat.RSC_latitude,RSC_GC, cmap=cmap,norm=norm)
    plt.colorbar(cp)
    plt.xlabel('sensor'); plt.ylabel('latitude')
    plt.title('OMI corrections')
    plt.xlim([-1,61]);plt.ylim([-70,70])
    plt.yticks(np.arange(-60,61,15))
    # plt.imshow(dat.RSC, extent=(0,60,-65,65), interpolation='nearest', cmap=cm.jet, aspect="auto")

    # plot d,e) RSC effect
    for i,arr in enumerate([gchcho, VCC_GC]):
        plt.subplot2grid((2, 6), (1, 2*i+2), colspan=2)
        m,cs,cb=pp.createmap(arr,[gclats,lats][i],[gclons,lons][i],
                          colorbar=False,vmin=vmin,vmax=vmax,
                          region=lims2)
        plt.title([Og,Ogc][i],fontsize=25)
        # rectangle around RSC
        #plot_rec(m,dat.RSC_region,color='k',linewidth=4)
        meridians=m.drawmeridians([-160,-140],labels=[0,0,0,1], linewidth=4.0, dashes=(None,None))
        m.drawparallels([-60,0,60],labels=[1,0,0,0],linewidth=0.0)
        for m in meridians:
            try:
                meridians[m][1][0].set_rotation(45)
            except:
                pass

    f.suptitle('Reference Sector Correction '+yms,fontsize=30)
    # Add colourbar to the right
    f.tight_layout()
    f.subplots_adjust(top=0.95)
    f.subplots_adjust(right=0.84)
    cbar_ax = f.add_axes([0.87, 0.20, 0.04, 0.6])
    cb=f.colorbar(cs,cax=cbar_ax)
    cb.set_ticks(np.logspace(13,17,5))
    cb.set_label('molec/cm$^2$',fontsize=24)

    pltname='Figs/Summary_RSC_Effect_%s.png'%(yms)
    f.savefig(pltname)
    print ('%s saved'%pltname)
    plt.close(f)