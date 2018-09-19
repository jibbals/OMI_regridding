# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
#import matplotlib
#matplotlib.use('Agg') # don't actually display any plots, just create them
from matplotlib import gridspec
import matplotlib

# module for hdf eos 5
#import h5py
#import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from scipy import interpolate
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn # Plotting density function
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings # To catch annoying warnings
import timeit

# local imports:
import utilities.plotting as pp
from utilities import GC_fio
from utilities import fio
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes import GC_class
from classes.omhchorp import omhchorp
from classes.campaign import campaign
import reprocess


##################
#####GLOBALS######
##################


NA     = util.NA
SWA    = util.SWA
SEA    = util.SEA
subs   = [SWA,NA,SEA]
labels = ['SWA','NA','SEA']
colours = ['chartreuse','magenta','aqua']

################
###FUNCTIONS####
################

def check_modelled_background(month=datetime(2005,1,1)):
    '''
        plot map of HCHO over remote pacific
        also plot map of HCHO when isop emissions are scaled to zero globally
    '''
    day0=month
    dayN=util.last_day(month)

    # DAILY OUTPUT
    gc=GC_class.GC_tavg(day0,dayN,)
    gc_noisop=GC_class.GC_tavg(day0,dayN,run='noisop')

    lats=gc.lats
    lons=gc.lons

    hcho1 = np.nanmean(gc.O_hcho,axis=0) # average the month
    bg1,bglats,bglons = util.remote_pacific_background(hcho1,lats,lons,)

    hcho2 = gc_noisop.O_hcho
    bg2, bglats,bglons = util.remote_pacific_background(hcho2,lats,lons)

    # plot with  panels, hcho over aus, hcho over remote pacific matching lats
    #                    hcho over both with no isop emissions
    vmin=1e15
    vmax=1e16
    ausregion=pp.__AUSREGION__
    bgregion=util.__REMOTEPACIFIC__
    bgregion[0]=ausregion[0]
    bgregion[2]=ausregion[2]
    clabel=r'$\Omega_{HCHO}$ [molec cm$^{-2}$]'

    plt.figure(figsize=[15,15])
    plt.subplot(2,2,1)
    pp.createmap(hcho1,lats,lons,region=ausregion, vmin=vmin,vmax=vmax, clabel=clabel, title='tropchem')
    plt.subplot(2,2,2)
    pp.createmap(bg1,bglats,bglons,region=bgregion, vmin=vmin,vmax=vmax, clabel=clabel, title='tropchem')
    plt.subplot(2,2,3)
    pp.createmap(hcho2,lats,lons,region=ausregion, vmin=vmin,vmax=vmax, clabel=clabel, title='no isoprene emitted')
    plt.subplot(2,2,4)
    pp.createmap(bg2,bglats,bglons,region=bgregion, vmin=vmin,vmax=vmax, clabel=clabel, title='no isoprene emitted',pname='Figs/GC/GC_background_hcho_%s.png'%month.strftime('%Y%m'))

def check_rsc_interp(d0=datetime(2005,1,1)):
    '''
    Look at calc of RSC from month of sat output, and it's interpolation
    by scipy.interpolation.interp1d
    '''

    # start timer
    start1=timeit.default_timer()

    ##########
    ### DO STUFFS
    ##########

    # get ref sector


    ref, lats = reprocess.GC_ref_sector(d0)
    ref_interp = interpolate.interp1d(lats, ref, kind='nearest')

    plt.plot(lats, ref, label='original')
    plt.plot(lats, ref_interp(lats),'+k',label='interpolation')
    lats_unordered=np.random.random_integers(-5,50,50)
    plt.plot(lats_unordered,ref_interp(lats_unordered), 'xr', label='interpolated (lats unordered)')
    plt.legend()
    ###########
    ### Record and time STUJFFS
    ###########
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for stuff"%((end-start1)/60.0))

    plt.savefig('Figs/GC/interp.png')
    plt.close()

def HCHO_vs_temp(d0=datetime(2005,1,1),d1=None,
                 region=SEA,regionlabel='SEA',regionplus=pp.__AUSREGION__):
    '''
    Plot comparison of temperature over region
    and over time in the region
    Plots look at surface level only

    d1 not yet implemented, just looks at month of d0
    '''
    if d1 is None:
        d1=util.last_day(d0)

    # ymd string and plot name
    ymd=d0.strftime('%Y%m%d') + '-' + d1.strftime('%Y%m%d')
    pname='Figs/GC/HCHO_vs_temp_%s_%s.png'%(regionlabel,ymd)

    # read hcho, and temperature
    gc=GC_class.GC_sat(d0,d1,keys=['IJ-AVG-$_CH2O','DAO-FLDS_TS'],run='tropchem')
    # Also read hcho from biogenic only run, should be better correlated
    gcb=GC_class.GC_sat(d0,d1,keys=['IJ-AVG-$_CH2O'],run='biogenic')
    fig=plt.figure(figsize=[12,16])

    # Read omhcho (reprocessed)
    omi=omhchorp(d0,d1,keylist=['latitude','longitude','VCC','gridentries'],ignorePP=True)
    omivcc=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    for ti in range(omi.n_times):
        omivcc[ti,:,:]=util.regrid(omi.VCC[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)

    # region + view area
    if regionplus is None:
        regionplus=np.array(region)+np.array([-10,-15,10,15])

    # Temperature avg:
    temp=gc.surftemp[:,:,:,0] # surface temp in Kelvin
    surfmeantemp=np.mean(temp,axis=0)

    lati,loni=util.lat_lon_range(gc.lats,gc.lons,regionplus)
    smt=surfmeantemp[lati,:]
    smt=smt[:,loni]
    tmin,tmax=np.min(smt),np.max(smt)
    oceanmask=util.get_mask(surfmeantemp,gc.lats,gc.lons,maskocean=True)

    # First plot temp map of region
    ax0a=plt.subplot(321)
    m,cs,cb=pp.createmap(surfmeantemp, gc.lats, gc.lons,
                         region=regionplus, cmapname='rainbow',
                         cbarorient='right',
                         vmin=tmin,vmax=tmax,
                         GC_shift=True, linear=True,
                         title='Temperature '+ymd, clabel='Kelvin')
    # Add rectangle around where we are correlating
    pp.add_rectangle(m,region,linewidth=2)

    # two plots, one for tropchem and one for biogenic hcho runs
    ax1=plt.subplot(312)
    ax2=plt.subplot(313)

    lati,loni=util.lat_lon_range(gc.lats,gc.lons,region)
    colors = cm.rainbow(np.linspace(0, 1, len(lati)*len(loni)))

    # Find area averaged regressions:
    #plot against Satellite hcho...
    ax0b=plt.subplot(322)
    for ax,areat,areah in zip(
            [ax0b,ax1,ax2],
            [np.copy(gc.surftemp[:,:,:,0]), np.copy(gc.surftemp[:,:,:,0]), np.copy(gc.surftemp[:,:,:,0])],
            [np.copy(omivcc), np.copy(gc.hcho[:,:,:,0]), np.copy(gcb.hcho[:,:,:,0])]):
        #areat=np.copy(gc.surftemp[:,:,:,0])
        #areah=np.copy(gc.hcho[:,:,:,0])

        for ti in range(len(areat[:,0,0])): # for each timestep
            # Remove oceansquares
            areat[ti][oceanmask] = np.NaN
            areah[ti][oceanmask] = np.NaN
        print (np.shape(areah))
        print (np.nanmean(areah))
        # subset and average spatially
        areat=areat[:,:,loni]
        areat=areat[:,lati,:]
        areat=np.nanmean(areat,axis=(1,2))


        areah=areah[:,:,loni]
        areah=areah[:,lati,:]
        areah=np.nanmean(areah,axis=(1,2))
        print (np.shape(areah))
        print (np.nanmean(areah))

        # Plot the area averaged regressions
        plt.sca(ax)
        plt.scatter(areat, areah, color='k')
        pp.add_regression(areat, areah, addlabel=True,
                                  exponential=True, color='k',
                                  linewidth=3)


    ## Next we add regressions in each gridsquare over the time dimension
    #

    # iterator ii, one for each land grid square we will regress
    ii=0
    # temp, hcho, exponential regression and slope for tropchem,biogenic
    tt,hh,e_r,e_m,=[],[],[],[]
    hhb,eb_r,eb_m,=[],[],[]
    for y in lati:
        for x in loni:

            # Don't correlate oceanic squares
            if oceanmask[y,x]:
                ii=ii+1
                continue


            # Add dot to map
            plt.sca(ax0a)
            mx,my = m(gc.lons[x], gc.lats[y])
            m.plot(mx, my, 'o', markersize=5, color=colors[ii])

            # Gridbox scatter plot
            iitemp=gc.surftemp[:,y,x,0]
            iihcho=gc.hcho[:,y,x,0]
            iihchob=gcb.hcho[:,y,x,0]

            plt.sca(ax1)
            plt.scatter(iitemp, iihcho, color=colors[ii])
            e_reg=pp.add_regression(iitemp, iihcho, addlabel=False,
                                    exponential=True, color=colors[ii],
                                    linewidth=0)

            plt.sca(ax2)
            plt.scatter(iitemp, iihchob, color=colors[ii])
            eb_reg=pp.add_regression(iitemp, iihchob, addlabel=False,
                                    exponential=True, color=colors[ii],
                                    linewidth=0)

            # create full list for regression
            tt.extend(list(iitemp))
            hh.extend(list(iihcho))
            hhb.extend(list(iihchob))
            # Exponential slopes and regressions
            e_r.append(e_reg[2])
            e_m.append(e_reg[0])
            eb_r.append(eb_reg[2])
            eb_m.append(eb_reg[0])

            # Next colour iterator
            ii=ii+1

    # Add exponential regressions:
    plt.sca(ax1)
    plt.title('Scatter (coloured by gridsquare)')
    plt.ylabel('HCHO$_{tropchem}$ ppbv')
    pp.add_regression(tt,hh,exponential=True,color='m',linewidth=3)

    plt.sca(ax2)
    plt.ylabel('HCHO$_{biogenic}$ ppbv')
    pp.add_regression(tt,hhb,exponential=True,color='m',linewidth=3)

    for ax in ax1,ax2:
        # reset lower bounds
        ylims=ax.get_ylim()
        plt.ylim([max([-2,ylims[0]]), ylims[1]])
        xlims=ax.get_xlim()
        plt.xlim([max([xlims[0],270]),xlims[1]])

        # add legend
        plt.sca(ax)
        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('Kelvin')

    # set fontsizes for plot
    fs=10
    for attr in ['ytick','xtick','axes']:
        plt.rc(attr, labelsize=fs)
    plt.rc('font',size=fs)


    # Add little density plots for both
    # show distribution of regressions  # x0,y0,xwid,ywid
    lil_axa  = fig.add_axes([0.17, 0.5, 0.19, 0.08])
    lil_axb = fig.add_axes([0.17, 0.2, 0.19, 0.11])

    for lil_e_r, lil_e_m, lil_ax in zip([e_r,eb_r],[e_m,eb_m],[lil_axa,lil_axb]):
        plt.sca(lil_ax)

        seaborn.set_style('whitegrid')
        seaborn.kdeplot(np.array(lil_e_r), linestyle='--',color='m',ax=lil_ax)
        #plt.xlim([0.3,1.2]); plt.ylim([0.0,5.0])
        plt.xticks(np.arange(0.3,1.21,.2))
        #plt.yticks([1,2,3,4])
        plt.xlabel('r (dashed)'); plt.ylabel('density (r)')
        plt.title('')
        plt.text(0.8,0.8,'n=%d'%len(lil_e_r),transform = lil_ax.transAxes)

        # show distribution of m values:
        lil_ax2 = lil_ax.twinx().twiny() # another density plot on same axes
        plt.sca(lil_ax2)

        seaborn.kdeplot(np.array(lil_e_m), linestyle='-',color='m',ax=lil_ax2)
        plt.xlabel('slope (m)')
        plt.ylabel('m density')
        plt.xticks(np.arange(0,0.401,0.1))
        plt.xlim(0,0.4)

    # ax0b also wants label
    plt.sca(ax0b)
    plt.legend()
    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)



def GC_vs_OMNO2d(d0=datetime(2005,1,1), d1=None,
                 region=pp.__AUSREGION__, regionlabel='AUS',
                 map_cmap='PuRd' ,reg_cmap='YlOrRd',dmap_cmap='RdBu_r'):
    '''
        Plot differences between GC and OMI NO2 columns...
    '''

    # Set up time bounds: one month if d1 is missing
    if d1 is None:
        d1=util.last_day(d0)
    dstr="%s-%s"%(d0.strftime("%Y%m%d"),d1.strftime("%Y%m%d"))

    ## Read our data:

    # Read omno2d
    OMNO2d,OM_attrs=fio.read_omno2d(day0=d0,dayN=d1)
    OM_tropno2=OMNO2d['tropno2']
    OM_lats=OMNO2d['lats']
    OM_lons=OMNO2d['lons']

    # Read satellite output from GC troprun
    # Keys needed for satellite tropNO2
    keys=['IJ-AVG-$_NO2','BXHGHT-$_BXHEIGHT','TIME-SER_AIRDEN','TR-PAUSE_TPLEV',]
    GC=GC_class.GC_sat(d0,dayN=d1,keys=keys)
    GC_tropno2=GC.get_trop_columns(['NO2'])['NO2']
    GC_lats,GC_lons=GC.lats,GC.lons

    # average them over time
    [GC_tropno2, OM_tropno2] = [ np.nanmean(arr,axis=0) for arr in [GC_tropno2, OM_tropno2] ]
    # get a lower resolution version of OMI tropno2
    OM_tropno2_low=util.regrid_to_lower(OM_tropno2,OM_lats,OM_lons,GC.lats_e,GC.lons_e)
    assert OM_tropno2_low.shape == GC_tropno2.shape, 'Reduced OMI Grid should match GC'

    ## PLOTTING First two rows
    ##

    # plot scale limits
    rmin,rmax=-50,50 # limits for percent relative difference in plots
    amin,amax=-1e15,1e15 # limits for absolute diffs
    vmin,vmax=1e14,14e14 # limits for molec/cm2 plots
    linear=True # linear or logarithmic scale
    regression_dot_size=30

    # set up axes for 3x2
    fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(14,18))

    # First plot, maps
    plt.sca(axes[0,0])
    pp.createmap(GC_tropno2,GC_lats,GC_lons,
                 title='GEOS-Chem', colorbar=False,
                 linear=linear, vmin=vmin,vmax=vmax,
                 cmapname=map_cmap, region=region)

    plt.sca(axes[0,1])
    pp.createmap(OM_tropno2, OM_lats, OM_lons,
                 title='OMI', colorbar=False,
                 linear=linear, vmin=vmin,vmax=vmax,
                 cmapname=map_cmap, region=region)

    plt.sca(axes[0,2])
    m,cs,cb= pp.createmap(OM_tropno2_low, GC_lats, GC_lons,
                 title='OMI (low res)', colorbar=False,
                 linear=linear, vmin=vmin,vmax=vmax,
                 cmapname=map_cmap, region=region)

    # Add colorbar to the right:
    #cax = fig.add_axes([0.9,0.375,0.04,0.25])
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cs,cax=cax, orientation='vertical')

    # Plot differences and corellation
    # first abs diff
    plt.sca(axes[1,0])
    pp.createmap(GC_tropno2-OM_tropno2_low, GC_lats, GC_lons,
                 title='GC - OMI', colorbar=True, clabel='molec/cm2',
                 linear=True, vmin=amin, vmax=amax,
                 cmapname=dmap_cmap, region=region)

    # then rel diff
    plt.sca(axes[1,1])
    pp.createmap(100*(GC_tropno2-OM_tropno2_low)/OM_tropno2_low, GC_lats, GC_lons,
                 title='100(GC - OMI)/OMI', colorbar=True, clabel='%',
                 linear=True, vmin=rmin, vmax=rmax,
                 cmapname=dmap_cmap, region=region)


    # now we must pull out the land data for corellations:

    subsets=util.lat_lon_subset(GC_lats,GC_lons,region,data=[GC_tropno2,OM_tropno2_low])
    lati,loni=subsets['lati'],subsets['loni']
    lats,lons=subsets['lats'],subsets['lons'] # GC_lats[lati],GC_lons[loni]
    [GC_tropno2,OM_tropno2_low]=subsets['data']

    # Lets mask the oceans at this point:
    oceanmask=util.get_mask(GC_tropno2, lats=lats,lons=lons,maskocean=True)
    for arr in [GC_tropno2,OM_tropno2_low]:
        arr[oceanmask] = np.NaN


    # then corellations
    plt.sca(axes[1,2])
    pp.plot_regression(OM_tropno2_low.flatten(),GC_tropno2.flatten(),
                       limsx=[vmin,vmax], limsy=[vmin,vmax],
                       logscale=False, legendfont=12, size=regression_dot_size)
    plt.title('GC vs OMI')
    plt.ylabel('GC')
    plt.xlabel('OMI')

    pname='Figs/GC/GC_vs_OMNO2_%s_%s.png'%(regionlabel,dstr)
    plt.suptitle('GC NO vs OMINO2d %s'%dstr, fontsize=35)
    plt.savefig(pname)
    print('Saved ',pname)
    plt.close()

def GCe_vs_OMNO2d(d0=datetime(2005,1,1), d1=None,
             region=pp.__AUSREGION__, regionlabel='AUS',
             soil=True, dstr_lab=None,
             map_cmap='PuRd' ,reg_cmap='YlOrRd',dmap_cmap='RdBu_r'):
    '''
        Plot regressions between soil NO vs bias from GC-OMI
            also anthro nox can be plotted if soil==False
    '''

    # Set up time bounds: one month if d1 is missing
    if d1 is None:
        d1=util.last_day(d0)
    dstr="%s-%s"%(d0.strftime("%Y%m%d"),d1.strftime("%Y%m%d"))
    if dstr_lab is None:
        dstr_lab=dstr

    ## Read our data:

    # Read omno2d
    OMNO2d,OM_attrs=fio.read_omno2d(day0=d0,dayN=d1)
    OM_tropno2=OMNO2d['tropno2']
    OM_lats=OMNO2d['lats']
    OM_lons=OMNO2d['lons']

    # Read satellite output from GC troprun
    # Keys needed for satellite tropNO2
    keys=['IJ-AVG-$_NO2','BXHGHT-$_BXHEIGHT','TIME-SER_AIRDEN','TR-PAUSE_TPLEV',]
    GC=GC_class.GC_sat(d0,dayN=d1,keys=keys)
    GC_tropno2=GC.get_trop_columns(['NO2'])['NO2']
    GC_lats,GC_lons=GC.lats,GC.lons

    # Read emissions from GC:
    GC_tavg=GC_class.GC_tavg(d0,d1,keys=['NO-SOIL_NO','ANTHSRCE_NO',], run='nochem')
    anthrono=GC_tavg.ANTHSRCE_NO
    soilno=GC_tavg.NO_soil

    # remove near zero emissions squares
    for arr in [anthrono, soilno]:
        arr[arr < 1]=np.NaN

    # average over time:
    to_average = [anthrono, soilno ,GC_tropno2, OM_tropno2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        [anthrono, soilno ,GC_tropno2, OM_tropno2] = [ np.nanmean(arr,axis=0) for arr in to_average ]
    OM_tropno2_low=util.regrid_to_lower(OM_tropno2,OM_lats,OM_lons,GC.lats_e,GC.lons_e)
    assert OM_tropno2_low.shape == GC_tropno2.shape, 'Reduced OMI Grid should match GC'
    ## PLOTTING
    ##

    # plot scale limits
    rmin,rmax=-50,50 # limits for percent relative difference in plots
    amin,amax=-1e15,1e15 # limits for absolute diffs
    vmin,vmax=1e14,14e14 # limits for molec/cm2 plots
    linear=True # linear or logarithmic scale
    elinear=[False,True][soil] # for emissions
    soilmin,soilmax=1e10,6e10 # limits for soil no emissions
    anthmin,anthmax=1e5,1.5e11 # limits for anthro emiss
    emiss=[anthrono,soilno][soil]
    emimin,emimax=[[anthmin,anthmax],[soilmin,soilmax]][soil]
    emisstitle=['$E_{anthro NO}$','$E_{soil NO}$'][soil]
    regression_dot_size=30

    # set up axes for 3,3,2
    fig = plt.figure(figsize=(15,17))
    axes=[ plt.subplot(331), plt.subplot(332), plt.subplot(333),
           plt.subplot(334), plt.subplot(335), plt.subplot(336),
                   plt.subplot(325), plt.subplot(326) ]

    # First plot GC map
    plt.sca(axes[0])
    m,cs,cb = pp.createmap(GC_tropno2,GC_lats,GC_lons,
                 title='$\Omega_{GEOS-Chem NO2}$', colorbar=False,
                 linear=linear, vmin=vmin,vmax=vmax,
                 cmapname=map_cmap, region=region)

    # Add colorbar to the right:
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cs,cax=cax, orientation='vertical')

    # Plot the omi map
    plt.sca(axes[1])
    pp.createmap(OM_tropno2_low, GC_lats, GC_lons,
                 title='$\Omega_{OMI NO2}$', colorbar=False,
                 linear=linear, vmin=vmin,vmax=vmax,
                 cmapname=map_cmap, region=region)

    # plot the emissions:
    plt.sca(axes[2])
    pp.createmap(emiss, GC_lats, GC_lons,
                 title=emisstitle, colorbar=True,
                 linear=elinear, vmin=emimin, vmax=emimax,
                 cmapname=reg_cmap, region=region)

    # Plot differences and corellation
    # first abs diff
    plt.sca(axes[3])
    pp.createmap(GC_tropno2-OM_tropno2_low, GC_lats, GC_lons,
                 title='GC - OMI', colorbar=True, clabel='molec/cm2',
                 linear=True, vmin=amin, vmax=amax,
                 cmapname=dmap_cmap, region=region)

    # then rel diff
    plt.sca(axes[4])
    pp.createmap(100*(GC_tropno2-OM_tropno2_low)/OM_tropno2_low, GC_lats, GC_lons,
                 title='100(GC - OMI)/OMI', colorbar=True, clabel='%',
                 linear=True, vmin=rmin, vmax=rmax,
                 cmapname=dmap_cmap, region=region)


    # now we must pull out the land data for corellations:

    subsets=util.lat_lon_subset(GC_lats,GC_lons,region,data=[GC_tropno2,OM_tropno2_low, emiss])
    lati,loni=subsets['lati'],subsets['loni']
    lats,lons=subsets['lats'],subsets['lons'] # GC_lats[lati],GC_lons[loni]
    [GC_tropno2,OM_tropno2_low, emiss]=subsets['data']

    # Lets mask the oceans at this point:
    oceanmask=util.get_mask(GC_tropno2, lats=lats,lons=lons,maskocean=True)
    for arr in [GC_tropno2,OM_tropno2_low, emiss]:
        arr[oceanmask] = np.NaN

    # bias between GC and OMI
    bias= (GC_tropno2 - OM_tropno2_low)

    # then corellations
    plt.sca(axes[5])
    colours = emiss
    clabel = 'molec/cm2/s'
    pp.plot_regression(OM_tropno2_low.flatten(),GC_tropno2.flatten(),
                       limsx=[vmin,vmax], limsy=[vmin,vmax],
                       logscale=False, legendfont=12, size=regression_dot_size,
                       showcbar=True, colours=colours.flatten(), clabel=clabel,
                       cmap=reg_cmap)
    plt.title('GC vs OMI')
    plt.ylabel('GC')
    plt.xlabel('OMI')

    #emiss vs GC
    plt.sca(axes[6])
    pp.plot_regression(emiss.flatten(), GC_tropno2.flatten(),
                       logscale=False, legendfont=12, diag=False,
                       cmap=reg_cmap, showcbar=False,
                       colours=colours.flatten(),
                       size=regression_dot_size)

    plt.title('GEOS-Chem $E_{NO}$ vs $\Omega_{NO2}$')
    plt.xlabel(['anthro','soil'][soil])
    plt.ylabel('GC')

    #emiss vs bias
    plt.sca(axes[7])
    pp.plot_regression(emiss.flatten(), bias.flatten(),
                       logscale=False, legendfont=12, diag=False,
                       cmap=reg_cmap, showcbar=False,
                       colours=colours.flatten(),
                       size=regression_dot_size)

    plt.title('$E_{NO}$ vs (GC-OMI)')
    plt.xlabel('emissions')
    plt.ylabel('GC-OMI')

    emisstype=['anthro','soil'][soil]
    pname='Figs/GC/GC%s_vs_OMNO2_%s_%s.png'%(emisstype,regionlabel,dstr)
    plt.suptitle('GEOS-Chem vs OMINO2d %s'%dstr_lab, fontsize=32)
    plt.savefig(pname)
    print('Saved ',pname)
    plt.close()


def GC_vs_OMI(month=datetime(2005,1,1),region=pp.__AUSREGION__):
    '''
    Plot comparison of month of GC output vs month of omhcho
    '''
    # READ OMI
    dayn=util.last_day(month)
    OMI=omhchorp(month,dayn=dayn)
    # READ GC
    GC=GC_class.GC_sat(month)

    # Check data
    print ('OMI (VCC) molec/cm2',OMI.VCC.shape)
    OMIhcho=OMI.time_averaged(month,dayn,keys=['VCC'])['VCC']# molec/cm2
    print('month average globally:',np.nanmean(OMIhcho))

    print("GC (O_hcho)",GC.attrs['O_hcho']['units'], GC.O_hcho.shape)
    GChcho=np.nanmean(GC.O_hcho,axis=0) # time averaged for the month
    print("month average globally:",np.nanmean(GChcho))

    plt.figure(figsize=(12,12))
    plt.subplot(221)
    pp.createmap(GChcho,GC.lats,GC.lons,aus=True,GC_shift=True, linear=True,
                 title='GC O_hcho', clabel=GC.attrs['O_hcho']['units'])

    plt.subplot(222)
    pp.createmap(OMIhcho,OMI.lats,OMI.lons,aus=True, linear=True,
                 title='VCC',clabel='molec/cm2')

    OMI_lr = OMI.lower_resolution(key='VCC',dates=[month,dayn])
    OMIhcho= OMI_lr['VCC']
    lats,lons=OMI_lr['lats'],OMI_lr['lons']
    assert lats==GC.lats, 'lats mismatch..'
    assert lons==GC.lons, 'lons mismatch..'

    diff=OMIhcho-GChcho
    rdiff=(OMIhcho-GChcho)/GChcho

    plt.subplot(223)
    pp.createmap(diff,lats,lons, aus=True, GC_shift=True, linear=True,
                 title='OMI - GC')

    plt.subplot(224)
    pp.createmap(rdiff,lats,lons, aus=True, GC_shift=True,
                 vmin=-2.0, vmax=2.0, linear=True,
                 title='(OMI - GC)/GC')

    pname='Figs/GC_vs_OMI_hcho.png'
    plt.savefig(pname)
    print("SAVED ",pname)


def compare_to_campaigns(d0=datetime(2005,1,31), de=datetime(2005,6,1), dfmt='%b %d'):
    ''' compare to SPS, MUMBA, more for GC season vs time shifted campaigns '''

    # Read campaigns:
    SPS1=campaign()
    SPS2=campaign()
    SPS1.read_SPS(1)
    SPS2.read_SPS(2)

    # Read GEOS-Chem:
    GC_paths=GC_fio.paths[0]+'/trac_avg*2005*0000'
    GCd,GCa=GC_fio.read_bpch(GC_paths,['IJ-AVG-$_ISOP','IJ-AVG-$_CH2O'],multi=True)

    # GC dates in datetime format
    GCdates=util.datetimes_from_np_datetime64(GCd['time'])
    # colocated with sydney gridbox
    lati,loni=util.lat_lon_index(SPS1.lat,SPS1.lon,GCd['lat'],GCd['lon'])
    GChcho=GCd['IJ-AVG-$_CH2O'][:,loni,lati,0]
    GCisop=GCd['IJ-AVG-$_ISOP'][:,loni,lati,0]/5.0 # ppbC to ppb

    # plot setup
    f=plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 3])
    ax1,ax2=plt.subplot(gs[0,0]),plt.subplot(gs[0,1])
    ax3,ax4=plt.subplot(gs[1,0],sharey=ax2),plt.subplot(gs[1,1],sharey=ax2)

    # GC data:
    # first plot is HCHO
    plt.sca(ax1)
    pp.plot_time_series(GCdates,GChcho,ylabel='HCHO [ppb]', legend=False, title='HCHO', xtickrot=30, dfmt=dfmt,
                     label='GEOS-Chem',color='red')

    # second plot is ISOP
    for ax in [ax2,ax3,ax4]:
        plt.sca(ax)
        pp.plot_time_series(GCdates,GCisop,ylabel='ISOP [ppb]', legend=False, title='Isop', xtickrot=30, dfmt=dfmt,
                         label='GEOS-Chem',color='red',linewidth=3)

    # loop over plotting hcho and isop
    labels=['SPS1','SPS2']
    markers=['x','+']
    for i,c in enumerate([SPS1,SPS2]):
        # shift dates to 2005
        dates= [d.replace(year=2005) for d in c.dates]

        #plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, xtickrot=30, dfmt='%Y%m', **pltargs)
        # campaign hcho
        plt.sca(ax1)
        pp.plot_time_series(dates,c.hcho,ylabel='HCHO [ppb]', legend=False, title='HCHO', xtickrot=30, dfmt=dfmt,
                     label=labels[i],color='k',marker=markers[i])

        # campaign isop
        plt.sca(ax2)
        pp.plot_time_series(dates,c.isop,ylabel='Isop [ppb]', legend=False, title='Isoprene', xtickrot=30, dfmt=dfmt,
                     label=labels[i], color='k', marker=markers[i])

        # isoprene closeup
        plt.sca([ax3,ax4][i==1])
        pp.plot_time_series(dates,c.isop,ylabel='Isop [ppb]', legend=False, title='Isoprene '+labels[i], xtickrot=30, dfmt=dfmt,
                     color='k', marker=markers[i], linewidth=2)
        plt.gca().set_xlim([dates[0],dates[-1]])

    # Legend and prettiness
    plt.sca(ax1)
    plt.legend(loc='best',fontsize=12)
    #plt.suptitle("GEOS-Chem vs campaigns, 2005",fontsize=30)
    f.subplots_adjust(hspace=.35)

    # subset to desired dates:
    for ax in [ax1,ax2]:
        ax.set_xlim([d0,de])

    # Halve how many ticks are shown
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xticks(ax.get_xticks()[::2])

    # save the figure:
    pname='Figs/GC_vs_Campaigns_%s-%s.png'%(d0.strftime("%Y%m%d"),de.strftime("%Y%m%d"))
    plt.savefig(pname)
    plt.close()
    print('SAVED: ',pname)

def compare_to_campaigns_daily_cycle():

    # Read campaigns:
    SPS1=campaign()
    SPS2=campaign()
    SPS1.read_SPS(1)
    SPS2.read_SPS(2)
    lat,lon=SPS1.lat,SPS1.lon

    d0,d1=SPS1.dates[0],SPS1.dates[-1]
    d2,d3=SPS2.dates[0],SPS2.dates[-1]

    print('SPS1:',d0,d1)
    print('SPS2:',d2,d3)

    d0 = d0.replace(year=2005)
    d1 = d1.replace(year=2005)
    d2 = d2.replace(year=2005)
    d3 = d3.replace(year=2005)

    print('GC1:',d0,d1)
    print('GC2:',d2,d3)

    # Read GEOS-Chem:
    GC1=GC_class.Hemco_diag(d0,d1,month=False)
    lati,loni=GC1.lat_lon_index(lat,lon)
    GC1_E_isop=GC1.E_isop_bio[:,lati,loni]
    gcoffset=GC1.local_time_offset[lati,loni]
    gcdates=[]
    for date in GC1.dates:
        gcdates.append(date+timedelta(seconds=int(3600*gcoffset)))
    # figure, first do whole timeline:
    f, axes = plt.subplots(2,2, gridspec_kw = {'height_ratios':[1, 4]})
    a0, a1, a2, a3 = axes[0,0],axes[0,1],axes[1,0],axes[1,1]
    plt.sca(a0)
    plt.plot(SPS1.isop, color='k')
    plt.sca(a1)
    plt.plot(np.arange(gcoffset,len(GC1_E_isop)+gcoffset), GC1_E_isop, color='r')

    for ax in [a0,a1]:
        plt.sca(ax)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off

    # then show daily cycle
    plt.sca(a2)
    print(len(SPS1.dates),SPS1.isop.shape)
    pp.plot_daily_cycle(SPS1.dates,SPS1.isop,houroffset=0, color='k') # already local time
    plt.sca(a3)
    pp.plot_daily_cycle(GC1.dates,GC1_E_isop,houroffset=gcoffset, color='r')

    pname='Figs/SPS1_DailyCycle.png'
    plt.savefig(pname)
    print('Saved ',pname)




def biogenic_vs_tavg():
    '''    '''
    # plot stuff
    d0=datetime(2005,1,1)
    args={'region':[-60,0,20,170],
          'clabel':'atom C/cm2/s',
          'linear':False,
          'make_edges':False,
          'GC_shift':True,
          'smoothed':False,
          'vmin':1e5, 'vmax':1e13,
          'cmapname':'rainbow',}

    dstr=d0.strftime("%Y%m%d")
    yyyymm=d0.strftime("%Y%m")

    # GC tavg for same day:
    tavg=GC_class.GC_tavg(d0)
    e_isop_bio=tavg.E_isop_bio[0] # average over time axis
    lats=tavg.lats
    lons=tavg.lons
    plt.figure(figsize=(11,10))
    plt.subplot(221)
    m,cs,cb=pp.createmap(e_isop_bio,lats,lons, **args)

    plt.subplot(222)
    # GC hemco diagnostics for one day:
    HD=GC_class.Hemco_diag(d0)
    days,hdisop=HD.daily_LT_averaged()
    hdisop=hdisop*HD.kgC_per_m2_to_atomC_per_cm2
    m,cs,cb=pp.createmap(hdisop,lats,lons,**args)

    plt.subplot(212)
    with np.errstate(divide='ignore',invalid='ignore'):
        ratio=hdisop/e_isop_bio
    args['linear']=True; args['clabel']='overpass/dayavg'
    args['vmin']=0.0; args['vmax']=5
    pp.createmap(ratio,lats,lons,**args)

    pname='Figs/Hemco_Vs_tavg.png'
    plt.savefig(pname)
    print('Saved figure ',pname)




def check_smearing():
    '''
        Calculate smearing using halfisoprene and tropchem trac_avg files..
    '''
    print("TODO")

def check_tropchem_monthly():
    '''
        see if monthly averaged output matches month of averaged daily averages
    '''
    # Read month of daily averages
    trop_mavg_fromdays=get_tropchem_data(date=datetime(2005,1,1),runtype='tropchem',monthavg=True)

    # read monthly average:
    trop_mavg=get_tropchem_data(date=datetime(2005,1,1),runtype='tropchem', fname='trac_avg_200501_month.nc')

    print("Keys in daily output")
    print(trop_mavg_fromdays.keys())
    print("Keys in monthly output")
    print(trop_mavg.keys())

    for key in ['E_isop_bio','hcho','isop','N_air']:
        print(key)
        print('from days, from monthly')
        mavg=trop_mavg[key] ;davg=trop_mavg_fromdays[key]
        print(davg.shape,mavg.shape)
        print(np.nanmean(davg),np.nanmean(mavg))

    # Also compare against UCX month average?

def compare_tc_ucx(d0=datetime(2007,1,1), dn=datetime(2007,2,28)):
    '''
        Check UCX vs Tropchem
        Plot surface and tropospheric HCHO, ISOP, O3, OH from satellite output files
    '''
    dstr=d0.strftime('%Y%m%d-')+dn.strftime('%Y%m%d')
    region=pp.__GLOBALREGION__

    ## Read the GEOS-Chem UCX files..
    satu=GC_class.GC_sat(d0,dn,run='UCX') # [days, lats, lons,72]
    # 'hcho', 'psurf', OH'
    tracu=GC_class.GC_tavg(d0,dn,run='UCX') # [lats,lons,72]
    # 'isop', 'hcho', 'psurf'

    ## read tropchem outputs
    satt=GC_class.GC_sat(d0,dn,run='tropchem') # [days, lats, lons, 47]
    #'isop', 'E_isop_bio', 'temp', 'surftemp', 'NO2', 'hcho', 'OH'
    tract=GC_class.GC_tavg(d0,dn,run='tropchem') # [days,lats,lons,47]
    # 'isop', 'NO2', 'hcho', 'psurf'
    lats=satu.lats
    lons=satu.lons

    satkeys=['hcho','isop','OH','O3']
    trackeys=['hcho','isop','OH', 'O3'] # for some reason O3 isn't removed by check in 12 lines
    # check we have the keys
    print(satt.attrs.keys())
    print(satu.attrs.keys())
    print(tract.attrs.keys())
    print(tracu.attrs.keys())
    for key in satkeys:
        if not ( hasattr(satt,key) and hasattr(satu,key) ):
            satkeys.remove(key)
            print('removing ucxkey',key)

    for key in trackeys:
        if not ( hasattr(tract,key) and hasattr(tracu,key) ):
            trackeys.remove(key)
            print('removing trackey',key)

    keys=[]
    keys.extend(satkeys)
    keys.extend(trackeys)


    # setup stuff for plotting: limits, units, etc.
    units={ 'hcho'      :r'molec cm$^{-3}$',
            'OH'        :r'molec cm$^{-3}$',
            'isop'      :r'molec cm$^{-3}$',
            'O3'        :r'molec cm$^{-3}$',}
    rlims={'hcho'       :(-20,20),
           'OH'         :(-50,50),
           'isop'       :(-70,70),
           'O3'         :(-20,20), }

    dlims={'hcho'       :(-1e13,1e13),
           'OH'         :(-2e10,2e10),
           'isop'       :(-1e15,1e15),
           'O3'         :(-1e18,1e18),}

    vlims={'hcho'       :(1e13, 1e15), # surface (looks good summer 07)
           'hcho_tc'    :(5e14, 5e16), # total column
           'OH'         :(1e9,  1e11),
           'OH_tc'      :(1e11, 1e14),
           'isop'       :(1e13, 5e15), # isoprene surface amounts molec/cm2
           'isop_tc'    :(1e14, 5e16),
           'O3'         :(1e15, 5e16),
           'O3_tc'      :(5e18, 1e19), }
    ticks ={'hcho_tc'   :[5e14, 1e15, 1e16, 5e16],
            'isop_tc'   :[5e14, 1e15, 1e16, 5e16],
            'O3'        :[1e15,1e16,5e16],
            'O3_tc'     :[5e18,1e19]}

    # satellite outputs to compare
    tot_satu=satu.get_total_columns(satkeys)
    tot_satt=satt.get_total_columns(satkeys)
    tot_tracu=tracu.get_total_columns(trackeys)
    tot_tract=tract.get_total_columns(trackeys)

    surf_satu=satu.units_to_molec_cm2(satkeys)
    surf_satt=satt.units_to_molec_cm2(satkeys)
    surf_tracu=tracu.units_to_molec_cm2(trackeys)
    surf_tract=tract.units_to_molec_cm2(trackeys)

    matplotlib.rcParams["text.usetex"]      = False     #
    matplotlib.rcParams["legend.numpoints"] = 1         # one point for marker legends
    matplotlib.rcParams["figure.figsize"]   = (16, 16)  #
    matplotlib.rcParams["font.size"]        = 22        # font sizes:
    matplotlib.rcParams["axes.titlesize"]   = 30        # title font size
    matplotlib.rcParams["axes.labelsize"]   = 20        #
    matplotlib.rcParams["xtick.labelsize"]  = 20        #
    matplotlib.rcParams["ytick.labelsize"]  = 20        #

    for key in satkeys:
        print(key)
        # compare surface and troposphere
        # First do total column

        dats   =  [np.nanmean(tot_satu[key],axis=0), np.nanmean(tot_satt[key],axis=0)] # average over time
        pname  = 'Figs/GC/UCX_vs_trp_middaytotcol_%s_%s.png'%(dstr,key)
        titles = ['UCX', 'tropchem']
        pp.compare_maps(dats,[lats,lats],[lons,lons],pname=pname,titles=titles,
                        vmin=vlims[key+'_tc'][0], vmax=vlims[key+'_tc'][1], # total column has own scale
                        ticks=ticks.get(key+'_tc',None),
                        rmin=rlims[key][0], rmax=rlims[key][1],
                        amin=dlims[key][0]*10, amax=dlims[key][1]*10, # total column diffs are 10 times higher (summed over vert)
                        linear=False, region=region,
                        suptitle='%s total column (middays %s) [%s]'%(key,dstr,'molec cm$^{-2}$'))

        # then do surface molec/cm2
        dats   = [np.nanmean(surf_satu[key][:,:,:,0],axis=0), np.nanmean(surf_satt[key][:,:,:,0],axis=0)]
        pname  = 'Figs/GC/UCX_vs_trp_middaysurf_%s_%s.png'%(dstr,key)
        titles = ['UCX', 'tropchem']
        pp.compare_maps(dats, [lats, lats], [lons, lons], pname=pname, titles=titles,
                        vmin=vlims[key][0], vmax=vlims[key][1],
                        ticks=ticks.get(key,None),
                        rmin=rlims[key][0], rmax=rlims[key][1],
                        amin=dlims[key][0], amax=dlims[key][1],
                        linear=False, region=region,
                        suptitle='%s surface (middays %s) [%s]'%(key,dstr,units[key]))

    for key in trackeys:
        print(key)
        # compare surface and troposphere
        # First do total column
        d1 = tot_tracu[key]
        if len(d1.shape) > 2:
            d1=np.nanmean(d1,axis=0) # tracu may or may not have days
        dats   = [d1, np.nanmean(tot_tract[key],axis=0)] # average over time
        pname  = 'Figs/GC/UCX_vs_trp_avgtotcol_%s_%s.png'%(dstr,key)
        titles = ['UCX', 'tropchem']
        pp.compare_maps(dats,[lats,lats],[lons,lons],pname=pname,titles=titles,
                        vmin=vlims[key+'_tc'][0], vmax=vlims[key+'_tc'][1],
                        ticks=ticks.get(key+'_tc',None),
                        rmin=rlims[key][0], rmax=rlims[key][1],
                        amin=dlims[key][0]*10, amax=dlims[key][1]*10,
                        linear=False, region=region,
                        suptitle='%s Total column (full day avgs %s) [%s]'%(key,dstr,'molec cm$^{-2}$'))

        # then do surface molec/cm2
        d2 = surf_tracu[key]
        if len(d2.shape) > 3:
            d2=np.nanmean(d2,axis=0)
        dats   = [d2[:,:,0], np.nanmean(surf_tract[key][:,:,:,0],axis=0)]
        pname  = 'Figs/GC/UCX_vs_trp_avgsurf_%s_%s.png'%(dstr,key)
        titles = ['UCX', 'tropchem']
        pp.compare_maps(dats, [lats, lats], [lons, lons], pname=pname, titles=titles,
                        vmin=vlims[key][0], vmax=vlims[key][1],
                        ticks=ticks.get(key,None),
                        rmin=rlims[key][0], rmax=rlims[key][1],
                        amin=dlims[key][0], amax=dlims[key][1],
                        linear=False, region=region,
                        suptitle='%s surface amounts (full day avgs %s) [%s]'%(key,dstr,units[key]))

def AMF_comparison_tc_ucx(month=datetime(2005,1,1)):
    '''
    Look at monthly averaged AMF using UCX and using tropchem
    '''
    ## read tropchem and ucx run
    #
    d0 = month
    dN = util.last_day(month)
    ucx=GC_class.GC_sat(d0,dN,run='UCX') # [days, lats, lons,72]
    trop=GC_class.GC_sat(d0,dN,run='tropchem') # [days, lats, lons, 47]

    ## To calculate AMFs we need scattering weights from satellite swath files
    #

    for i, d in enumerate(util.list_days(d0, dN)):
        omhcho = reprocess.get_good_pixel_list(d)

        w=omhcho['omega']
        w_pmids=omhcho['omega_pmids']
        AMF_G=omhcho['AMF_G']
        lat=omhcho['lat']
        lon=omhcho['lon']

        print("Running amf calc for tropchem")
        for arr in [w, w_pmids, AMF_G, lat, lon]:
            print(np.shape(arr))
            print('      ', arr)
        trop_amf_z, trop_amf_s = trop.calculate_AMF(w[i], w_pmids[i], AMF_G[i], lat[i], lon[i], plotname=None, debug_levels=False)
        print(trop_amf_z,trop_amf_s)
        print("Running amf calc for UCX")
        ucx_amf_z, ucx_amf_s = ucx.calculate_AMF(w[i], w_pmids[i], AMF_G[i], lat[i], lon[i], plotname=None, debug_levels=False)
        print(ucx_amf_z,ucx_amf_s)
        break
    return

def OLD_compare_tc_ucx(date=datetime(2005,1,1),extra=False,fnames=None,suffix=None):
    '''
        Check UCX vs Tropchem
        set Extra to true to look at E_isop_biog and OH (uses specific output file 20050101
        set fnames=[tropchem.nc,ucx.nc] outputs to test specific files
    '''
    ymstr=date.strftime("%Y%m")
    region=pp.__GLOBALREGION__

    # Read the netcdf files (output specified for this test)
    # UCX FILE:
    #ucx=GC_output(date,UCX=True, fname='UCX_trac_avg_20050101.nc')
    if fnames is None:
        ucx=GC_output(date,UCX=True)
        trp=GC_output(date,UCX=False,monthavg=True)
    else:
        trp_fname,ucx_fname=fnames#
        trp=GC_output(date,UCX=False, monthavg=True, fname=trp_fname)
        ucx=GC_output(date,UCX=True, fname=ucx_fname)

    lats_all=ucx.lats
    lons_all=ucx.lons

    all_keys=['hcho','isop','E_isop_bio','OH','O3','NO2']
    keys=[]
    for key in all_keys:
        if hasattr(ucx,key) and hasattr(trp,key): # look at fields in both outputs
            keys.append(key)
            print('%s will be compared at surface '%(key))
    #keys=['hcho','isop']
    units={ 'hcho'      :'ppbv',        #r'molec cm$^{-2}$',
            'E_isop_bio':r'atom C cm$^{-2}$ s$^{-1}$',
            'OH'        :r'molec cm$^{-3}$',
            'isop'      :r'ppbv',
            'O3'        :r'ppbv',
            'NO2'       :r'ppbv',}
    cbarfmt={}; cbarxtickrot={}
    rlims={'hcho'       :(-20,20),
           'E_isop_bio' :(-50,50),
           'OH'         :(-50,50),
           'isop'       :(-70,70),
           'O3'         :(-50,50),
           'NO2'        :(-50,50),}
    dlims={'hcho'       :(-.5,.5),
           'E_isop_bio' :(-1e12,1e12),
           'OH'         :(-8e5,8e5),
           'isop'       :(-6,6),
           'O3'         :(-40,40),
           'NO2'        :(-100,100),}
    alims={'hcho'       :(None,None),
           'E_isop_bio' :(None,None),
           'OH'         :(1e6,4e6),
           'isop'       :(None,None),
           'O3'         :(None,None),
           'NO2'        :(None,None),}
    for key in keys:
        cbarfmt[key]=None; cbarxtickrot[key]=None
    cbarfmt['OH']="%.1e"; cbarxtickrot['OH']=30

    ucx_data=ucx.get_field(keys=keys,region=region)
    trp_data=trp.get_field(keys=keys,region=region)
    lats=ucx_data['lats'];lons=ucx_data['lons']

    print("Region: ",region)
    for key in keys:
        print(key)
        dats=[ucx_data[key], trp_data[key]]
        for di,dat in enumerate(dats):
            pre=['UCX  ','trop '][di]
            print(pre+"%s shape: %s"%(key,str(np.shape(dat))))
            print("    min/mean/max: %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
            # Just look at surface from here
            if len(dat.shape)==3:
                dat=dat[0]
                print("    at surface  : %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
                dats[di]=dat

        # whole tropospheric column
        #data['tc']  = tc.get_trop_columns(keys=keys)
        #data['ucx'] = ucx.get_trop_columns(keys=keys)

        # Plot values and differences for each key
        suffix=[suffix,''][suffix is None]
        pname='Figs/GC/UCX_vs_trp_glob_%s_%s%s.png'%(trp.dstr,key,suffix)
        u=dats[0];t=dats[1]
        amin,amax = alims[key]
        rmin,rmax = rlims[key]
        dmin,dmax = dlims[key]
        args={'region':region,'clabel':units[key], 'vmin':amin, 'vmax':amax,
              'linear':True, 'cmapname':'PuRd', 'cbarfmt':cbarfmt[key],
              'cbarxtickrot':cbarxtickrot[key]}

        f,axes=plt.subplots(2,2,figsize=(16,14))

        plt.sca(axes[0,0])
        pp.createmap(u,lats,lons, title="%s UCX"%key, **args)

        plt.sca(axes[0,1])
        pp.createmap(t,lats,lons, title="%s tropchem"%key, **args)

        plt.sca(axes[1,0])
        args['vmin']=dmin; args['vmax']=dmax
        args['cmapname']='coolwarm'
        pp.createmap(u-t,lats,lons,title="UCX - tropchem", **args)

        plt.sca(axes[1,1])
        args['vmin']=rmin; args['vmax']=rmax; args['clabel']='%'
        pp.createmap((u-t)*100.0/u, lats, lons, title="100*(UCX-tc)/UCX",
                     suptitle='%s %s %s'%('surface', key, trp.dstr), pname=pname, **args)



def Examine_Model_Slope(month=datetime(2005,1,1)):
    '''
        compares isop emission [atom_C/cm2/s] against hcho vert_column [molec_hcho/cm2]
        as done in Palmer et al. 2003
        Also plots sample of regressions over Australia
    '''

    # Retrieve data
    dates= util.list_days(month,month=True)
    GC=GC_class.GC_biogenic(month)
    region=pp.__AUSREGION__
    ymstr=month.strftime('%b, %Y')
    hcho_min=1e14
    hcho_max=3e16
    Eisop_min=1e11
    Eisop_max=1.2e13
    cmapname='gnuplot'
    # plot names
    pname=month.strftime('Figs/GC/E_isop_vs_hcho_%Y%m.png')


    # Get slope and stuff we want to plot
    model   = GC.model_slope(return_X_and_Y=True)
    lats    = model['lats']
    lons    = model['lons']
    hcho    = model['hcho']
    isop    = model['isop']
    # Y=slope*X+b with regression coeff r
    reg     = model['r']
    off     = model['b']
    slope   = model['slope']
    ocean   = util.oceanmask(lats,lons)
    hcho[:,ocean] = np.NaN
    isop[:,ocean] = np.NaN

    f,axes=plt.subplots(2,2,figsize=(20,14))

    # Now plot the slope and r^2 on the map of Aus:
    plt.sca(axes[0,0]) # first plot slope
    vmin=1e-7
    slope[slope < vmin] = np.NaN # Nan the zeros and negatives for nonlinear plot
    pp.createmap(slope, lats, lons, vmin=1e3, vmax=1e6,
                 aus=True, linear=False, cmapname=cmapname,
                 suptitle="HCHO trop column vs isoprene emissions %s"%ymstr,
                 clabel='slope', title=r'$\Omega_{HCHO}$ = slope x E$_{isop}$ + b')
    plt.sca(axes[1,0]) # then plot r2 and save figure
    bmap,cs,cb = pp.createmap(reg**2,lats,lons,vmin=0,vmax=1.0,
                              aus=True,linear=True, cmapname=cmapname,
                              clabel=r'r$^2$')

    # plot time series (spatially averaged)
    ts_isop=np.nanmean(isop,axis=(1,2))
    ts_hcho=np.nanmean(hcho,axis=(1,2))
    plt.sca(axes[0,1])
    pp.plot_time_series(dates,ts_isop,ylabel=r'E$_{isop}$ [atom C cm$^{-2}$ s$^{-1}$]',
        title='time series for %s'%ymstr, dfmt="%d", color='r',legend=False, label=r'E$_{isop}$')
    h1, l1 = axes[0,1].get_legend_handles_labels()
    twinx=axes[0,1].twinx()
    plt.sca(twinx)
    pp.plot_time_series(dates,ts_hcho,ylabel=r'$\Omega_{HCHO}$ [ molec cm$^{-2}$ ]',
        xlabel='time', dfmt="%d", color='m', legend=False, label=r'$\Omega_{HCHO}$')
    h2, l2 = twinx.get_legend_handles_labels()
    plt.legend(h1+h2, l1+l2, loc='best')


    plt.sca(axes[0,1])
    plt.autoscale(True)
    # plot a sample of ii_max scatter plots and their regressions
    ii=0; ii_max=9
    colours=[cm.rainbow(i) for i in np.linspace(0, 0.9, ii_max)]
    randlats= np.random.choice(range(len(lats)), size=30)
    randlons= np.random.choice(range(len(lons)), size=30)
    # loop over random lats and lons
    for xi,yi in zip(randlons,randlats):
        if ii==ii_max: break
        lat=lats[yi]; lon=lons[xi]
        X=isop[:,yi,xi]; Y=hcho[:,yi,xi]
        if np.isclose(np.mean(X),0.0) or np.isnan(np.mean(X)): continue
        xlims=np.array([Eisop_min,Eisop_max])
        # add dot to map
        plt.sca(axes[1,0])
        bmap.plot(lon,lat,latlon=True,markersize=10,marker='o',)

        # Plot scatter and regression
        plt.sca(axes[1,1])
        plt.scatter(X,Y,color=colours[ii])
        m,b,r = slope[yi,xi],off[yi,xi],reg[yi,xi]
        plt.plot(xlims, m*xlims+b,color=colours[ii],
            label='Y[%5.1fS,%5.1fE] = %.1eX + %.2e, r=%.2f'%(-1*lat,lon,m,b,r))

        ii=ii+1
    plt.xlim(xlims)
    plt.ylim([hcho_min,hcho_max])
    plt.xlabel(r'E$_{isop}$ [atom C cm$^{-2}$ s$^{-1}$]')
    plt.ylabel(r'$\Omega_{HCHO}$ [molec cm$^{-2}$]')
    plt.title('Sample of regressions')
    plt.legend(loc=0,fontsize=9) # show legend

    plt.savefig(pname)
    plt.close()
    print("SAVED: ",pname)



def E_isop_series(gc, aus=False, region=None):
    ''' Plot E_isop time series '''

    data=gc.E_isop
    # subset to region
    reg=gc._get_region(aus,region)
    lati,loni=pp.lat_lon_range(gc.lats,gc.lons,reg)
    data=data[lati,:]
    data=data[:,loni]

    # Average over space lats and lons
    data=np.mean(data, axis=(0,1))

    # plot name
    pname='Figs/GC/E_isop_series_%s%s.png'%(['','aus_'][aus], gc.dstr)

    f=plt.figure(figsize=(10,8))
    pp.plot_time_series(gc.dates, data, xlabel='time', dfmt="%m%d",
        ylabel=r'E$_{isop}$ [atom$_C$ cm$^{-2}$ s$^{-1}$]',
        pname=pname, legend=True, title='Emissions of Isoprene (%s)'%gc.dstr)


def E_isop_map(gc, aus=False, region=None):
    ''' basemap plot of E_isop
        region=[S,W,N,E] boundaries
    '''
    region=gc._get_region(aus, region)
    data=np.mean(gc.E_isop,axis=2) # average over time

    pname='Figs/GC/E_isop_%s%s.png'%(['','aus_'][aus], gc.dstr)
    pp.createmap(data,gc.lats,gc.lons, region=region, vmin=1e10,vmax=1e13,
        title='Emissions of isoprene', clabel=r'atom$_C$ cm$^{-2}$ s$^{-1}$',
        pname=pname)

def check_shapefactors(date=datetime(2005,1,1)):
    '''
        Check shapefactors for shapefactor nc files
    '''
    sf= GC_class.GC_sat(date) #gchcho(date)

    def check_column(key):
        blah=getattr(sf,key)
        print("%s: %s : %.2e"%(key, blah.shape, np.nanmean(blah)))
        while len(blah.shape) > 1:
            blah=blah[:,0]
        print(blah)
    #print("Shape: ", sf.Shape_s.shape, np.nanmean(sf.Shape_s))
    #print("N_HCHO: ", sf.N_HCHO.shape, np.nanmean(sf.N_HCHO))
    #print("pmids: ", sf.pmids.shape)
    for key in ['Shape_s','N_HCHO','zmids','boxH','sigmas','pmids','pedges']:
        check_column(key)

# If this script is run directly:
if __name__=='__main__':
    pp.InitMatplotlib()

    all2005=[datetime(2005,1,1),util.last_day(datetime(2005,12,1))]
    sum05=[datetime(2005,1,1),util.last_day(datetime(2005,2,1))]
    aut05=[datetime(2005,3,1),util.last_day(datetime(2005,5,1))]
    win05=[datetime(2005,6,1),util.last_day(datetime(2005,8,1))]
    spr05=[datetime(2005,9,1),util.last_day(datetime(2005,11,1))]
    dstrs=['Jan-Feb 2005','Autumn (MAM) 2005','Winter (JJA) 2005',
           'Sprint (SON) 2005']
    d0=datetime(2005,1,1)
    d1=datetime(2005,1,5)
    region=pp.__AUSREGION__
    label='AUS'

    ## Grab picture of background hcho levels
    # ran on 7/9/18
    #check_modelled_background()

    ## UCX VS TROPCHEM AMF
    #
    AMF_comparison_tc_ucx()

    ## tropchem vs UCX plots
    # Look at 2007 summer since I have OH for daily avg files from then.
    #compare_tc_ucx(datetime(2007,1,1),util.last_day(datetime(2007,2,1)))

    # Checking units:

    #Examine_Model_Slope() # unfinished 30/5/18
    #check_rsc_interp()   # last run 29/5/18

    #HCHO_vs_temp(d0=d0,d1=d1,
    #             region=SEA,regionlabel='SEA',
    #             regionplus=pp.__AUSREGION__)

    # Test the function with 3 days of data
    #GC_vs_OMNO2d(d0=d0, d1=d1,
    #             region=region, regionlabel=label,
    #             drop_low_anthro=True)

    #
    #for dates,dstr in zip([sum05,spr05,win05,aut05],dstrs):

     #   for soil in [True, False]:
     #       GCe_vs_OMNO2d(d0=dates[0], d1=dates[1],
     #            region=region, regionlabel=label,
     #            soil=False,dstr_lab=dstr)

        #GC_vs_OMNO2d(d0=dates[0], d1=dates[1],
        #             region=region, regionlabel=label)

    #    for region, label in zip(subs,labels):
    #        HCHO_vs_temp(d0=dates[0],d1=dates[1],
    #                     region=region,regionlabel=label)
    #        GC_vs_OMNO2d(d0=dates[0], d1=dates[1],
    #                     region=region, regionlabel=label,
    #                     drop_low_anthro=True)



    #GC_vs_OMNO2d(month=datetime(2005,1,1))
    #compare_to_campaigns_daily_cycle()
    #compare_to_campaigns()
    #check_shapefactors()
    #check_tropchem_monthly()
    #biogenic_vs_tavg()
    # Compare explicit dates:
    #    for cdate in [ datetime(2004,7,1), ]:
    #        yymm=cdate.strftime("%Y%m")
    #        fnames= [ "trac_avg_%s.nc"%yymm, "trac_avg_UCX_%s.nc"%yymm]
    #        compare_tc_ucx(cdate,fnames=fnames)

    #compare_tc_ucx(datetime(2005,1,1),
    #               fnames=['trac_avg_200501_month.nc','trac_avg_UCX_200501.nc'],
    #               suffix='_rerun')



    # scripts mapping stuff
    date=datetime(2005,1,1)
    #tc=GC_output(date,UCX=False)
    #E_isop_map(tc,aus=True)
    #E_isop_series(tc,aus=True)
    #isop_hcho_RMA(tc)
