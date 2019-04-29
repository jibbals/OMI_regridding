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
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D # 3d scatter
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


######################
##      plotting functions
######################

def plot_tracks(RSC,RSC_lats=np.linspace(-90,90,500),
                linestyle='.',
                cmapname='jet', labels=True, colorbar=True):
    ## plot each track with a slightly different colour

    cmap=plt.cm.cmap_d[cmapname]
    colors=[cmap(i) for i in np.linspace(0, 1, 60)]
    for track in range(60):
        plt.plot(RSC[:,track], RSC_lats, linestyle, color=colors[track])

    if labels:
        plt.ylabel('latitude')
        plt.xlabel('molec/cm2')

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=plt.Normalize(vmin=0,vmax=60))
        sm._A=[]
        cb=plt.colorbar(sm)
        cb.set_label('track')

def RSC_map(data,lats,lons, vmin=5e14, vmax=1e16, region=[-65, -171, 65, -129],
            **createmapargs):
    ''' Draw standard map around RSC region '''

    m,cs,cb=pp.createmap(data, lats, lons, region=region,
                         vmin=vmin, vmax=vmax, linear=False,
                         **createmapargs)

    m.drawmeridians([ -160, -140],labels=[0,0,0,0])
    return m, cs, cb

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
    lables=[Ooc,r'$\Omega_{Oc,new}$']
    om=omrp(d0,dayn=dn, keylist=titles)

    # Compare old vs new avg for one month
    datas=[np.nanmean(getattr(om,title),axis=0) for title in titles]
    lats=[om.lats,om.lats]
    lons=[om.lons,om.lons]
    pp.compare_maps(datas,lats,lons,pname=pname, linear=True,
                    vmin=1e14,vmax=2e16,
                    titles=lables,rmin=-50,rmax=50, amin=-5e15,amax=5e15,
                    suptitle='%s with and without updated RSC (%s)'%(Ooc,ymstr))

def intercomparison(month=datetime(2005,1,1)):
    '''
        Look at RSC in a month between AMF types used to turn model VC into SC
    '''
     # plot name
    ymstr=month.strftime('%Y%m')
    pname='Figs/RSC_intercomparison_%s.png'%ymstr
    cmapname='plasma'

    # read in VCC_OMI and VCC_OMI_newrsc for month
    d0=util.first_day(month)
    dn=util.last_day(month)
    titles=['RSC','RSC_latitude']
    lables=['RSC$_O$','RSC$_G$','RSC$_P$']
    colours=['orange','green','purple']
    om=omrp(d0,dayn=dn, keylist=titles) #
    lats=om.RSC_latitude
    tracks=np.arange(1,61,1) # excludes last point
    RSC=np.nanmean(om.RSC,axis=0) # RSC=[time, lats, track, type]

    vmin=-1e16
    vmax=1e16
    #norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax,clip=True)
    plt.figure(figsize=(15,15))
    for i in range(3):
        plt.subplot(2,3,1+i)

        plt.pcolormesh(tracks,lats,np.ma.masked_invalid(RSC[:,:,i]),
                      cmap=cmapname,vmin=vmin,vmax=vmax)
        plt.title(lables[i])
        if i==0:
            plt.ylabel('latitude')
        if i==1:
            plt.xlabel('track')
        if i==2:
            cb=plt.colorbar(orientation='vertical')
            cb.set_label('molec/cm2')
        # plot the 60 tracks as lines
        plt.subplot(2,3,4+i)
        plot_tracks(RSC[:,:,i],labels=False, colorbar=i==2)
        if i==0:
            plt.ylabel('latitude')
        if i==1:
            plt.xlabel('molec/cm2')
        plt.title(lables[i])


    #[[plt.plot(lats, RSC[:,i,j],color=colours[j],label=[None,lables[j]][i==0]) for i in range(60)] for j in range(3)]
    #plt.legend()


    plt.savefig(pname)
    print ('%s saved'%pname)
    plt.close()

def check_RSC(day=datetime(2005,1,1), track_corrections=True):
    '''
    Grab the RSC from both GEOS-Chem and OMI for a particular day
    Plot and compare the RSC region
    Plot the calculated corrections
    '''
    print("Running check_RSC")
    # Read in one day average
    omhchorp1=fio.read_omhchorp(day)
    yyyymmdd=day.strftime("%Y%m%d")
    rsc=omhchorp1['RSC']
    ref_lat_bins=omhchorp1['RSC_latitude']
    rsc_gc=omhchorp1['RSC_GC'] # RSC_GC is in molecs/cm2 as of 11/08/2016


    ## plot the reference region,
    # need to make 500x3 array for basemap not to lose its shit
    rsc_gc_new = np.transpose(np.array([ rsc_gc, rsc_gc, rsc_gc] ))
    # Make the GC map:
    f2=plt.figure(1, figsize=(14,10))
    plt.subplot(141)
    rsc_lons=np.array([-160,-150,-140])
    m,cs,cb = RSC_map(rsc_gc_new,ref_lat_bins,rsc_lons, GC_shift=False,colorbar=False)
    m.drawparallels([-60,0,60],labels=[1,0,0,0])
    plt.title('RSC_GC')

    lats=omhchorp1['lats']
    lons=omhchorp1['lons']
    loni= (lons > -160) * (lons < -140)

    # Plot the VC_OMI,GC, and the VCC_GC
    for i,title in enumerate(['VC_OMI','VC_GC','VCC_GC']):
        data=omhchorp1[title] # [ 720, 1152 ] molecs/cm2
        # new map with OMI SC data
        plt.subplot(142+i)
        plt.title(title,fontsize=18)
        m,cs,cb=RSC_map(data[:,loni],lats,lons[loni],
                        colorbar=False)
        m.drawparallels([-60,0,60],labels=[0,0,0,0])

    #cb=m.colorbar(cs,"right",size="8%", pad="3%")
    #f2.suptitle('RSC %s'%yyyymmdd,fontsize=20)
    cb=pp.add_colourbar(f2,cs,ticks=None,label=None,fontsize=15)
    cb.set_label('molec/cm2')




    outfig2='Figs/RSC_GC_%s.png'%yyyymmdd
    plt.savefig(outfig2)
    print(outfig2+' saved')
    plt.close(f2)

    if track_corrections:
        f1=plt.figure(figsize=(12,12))
        plot_tracks(rsc[:,:,0])
        plt.title('Reference sector corrections for %s'%yyyymmdd,fontsize=20)
        pname='Figs/RSC_track_corrections%s.png'%yyyymmdd
        plt.savefig(pname)
        print('SAVED: ',pname)
        plt.close(f1)

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
    cbarname='jet'
    
    matplotlib.rcParams['image.cmap'] = cbarname #'PuRd' #'inferno_r'       # Colormap default
    
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
        plt.title(["OMI VC","OMI corrected"][i],fontsize=25)
        m.drawparallels([-40,0,40],labels=[1-i,0,0,0],linewidth=0.0)

    # print some stats of changes
    diffs=dat.VCC_GC-dat.VC_GC
    print ("Mean difference VC - VCC:%7.5e "%np.nanmean(diffs))
    print ("%7.2f%%"%(np.nanmean(diffs)*100/np.nanmean(dat.VC_GC)))
    print ("std VC - VCC:%7.5e "%np.nanstd(diffs))

    # plot c) RSC by sensor and latitude
    plt.subplot2grid((2, 6), (1, 0), colspan=2)
    RSC_GC=np.nanmean(dat.RSC[:,:,:,1],axis=0) # average over time
    norm = plt.cm.colors.Normalize(vmin=-1e16,vmax=2e16)
    cmap=plt.cm.cmap_d[cbarname]

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
        plt.title(["GEOS-Chem VC","OMI corrected"][i],fontsize=25)
        # rectangle around RSC
        #plot_rec(m,dat.RSC_region,color='k',linewidth=4)
        meridians=m.drawmeridians([-160,-140],labels=[0,0,0,1], linewidth=4.0, dashes=(None,None))
        m.drawparallels([-60,0,60],labels=[1,0,0,0],linewidth=0.0)
        for m in meridians:
            try:
                meridians[m][1][0].set_rotation(45)
            except:
                pass

    f.suptitle('HCHO reference sector correction '+yms,fontsize=30)
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