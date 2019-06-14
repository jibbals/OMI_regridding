# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them
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
from scipy.constants import N_A as N_Avogadro

# local imports:
import utilities.plotting as pp
from utilities import GC_fio, fio, masks
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes import GC_class
from classes.omhchorp import omhchorp
from classes.campaign import campaign
import reprocess
import concurrent.futures

from utilities import GMAO

##################
#####GLOBALS######
##################

__AUSREGION__=GMAO.__AUSREGION__
__GLOBALREGION__=GMAO.__GLOBALREGION__
__subregions__ = GMAO.__subregions__
__subregions_colors__ = GMAO.__subregions_colors__
__subregions_labels__ = GMAO.__subregions_labels__

NA     = util.NA
SWA    = util.SWA
SEA    = util.SEA
subs   = [SWA,NA,SEA]
labels = ['SWA','NA','SEA']
colours = ['chartreuse','magenta','aqua']



################
###FUNCTIONS####
################

def check_amf_differences(month=datetime(2005,1,1)):
    '''
        Look at AMF_z and AMF_s for a month or two and check how different they are...
    '''
    # read omhchorp month
    d0,d1 = util.first_day(month), util.last_day(month)
    amf_list = ['AMF_GC',        # AMF calculated using by GEOS-Chem
                'AMF_GCz',       # secondary way of calculating AMF with GC
                'AMF_OMI',       # AMF from OMI swaths
                'AMF_PP', 
                ]
    
    omrp = omhchorp(d0,d1, keylist=amf_list)
    # subset to AUS
    lati,loni=util.lat_lon_range(omrp.lats,omrp.lons,pp.__AUSREGION__)
    agc=omrp.AMF_GC[:,lati,:]
    agc = agc[:,:,loni]
    agcz=omrp.AMF_GCz[:,lati,:]
    agcz=agcz[:,:,loni]
    aomi=omrp.AMF_OMI[:,lati,:]
    aomi=aomi[:,:,loni]
    app=omrp.AMF_PP[:,lati,:]
    app=app[:,:,loni]
    
    plt.scatter(agc, agcz)
    plt.xlabel('$AMF_{\sigma}$')
    plt.ylabel('$AMF_z$')
    plt.title('AMF comparison over Australia for January 2005')
    pname='test_amfs.png'
    plt.savefig(pname)
    print('saved ',pname)
    plt.close()
    

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

def check_old_vs_new_emission_diags(month=datetime(2005,1,1)):
    # compare new_emissions hemco output
    d0=datetime(month.year,month.month,1)
    d1=util.last_day(d0)
    old, oldattrs = GC_fio.read_Hemco_diags_hourly(d0,d1, new_emissions=False)
    new, newattrs = GC_fio.read_Hemco_diags_hourly(d0,d1, new_emissions=True)

    lats=old['lat']
    lons=old['lon']

    old_isop = np.mean(old['ISOP_BIOG'],axis=0) # hourly -> 1 month avg
    new_isop = np.mean(new['ISOP_BIOG'],axis=0) # daily -> 1 month avg

    pname=month.strftime('Figs/Emiss/check_old_new_emission_diags_%Y%m.png')
    pp.compare_maps([old_isop,new_isop], [lats,lats],[lons,lons],linear=True,
                    rmin=-400,rmax=400, vmin=0,vmax=0.8e-9, titles=['old','new'],
                    region=pp.__GLOBALREGION__, pname=pname)



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
    omi=omhchorp(d0,d1,keylist=['latitude','longitude','VCC_OMI','gridentries'])
    omivcc=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    for ti in range(omi.n_times):
        omivcc[ti,:,:]=util.regrid(omi.VCC_OMI[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)

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
    # top left plot should have y axis on outside
    ax0b.yaxis.tick_right()
    ax0b.yaxis.set_label_position("right")        

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

def GCe_vs_OMNO2d(year=2005,
                  map_cmap='PuRd' ,reg_cmap='YlOrRd',dmap_cmap='RdBu_r'):
    '''
        for each season:
            Fig 1: Plot OMIno2, GCno2, and bias
            Fig 2: Plot Emissions of NO (anthro and soil) from GC
            Fig 3: Plot GC vs OMI, Eanth vs BIAS, Esoil vs BIAS
    '''

    #first december should be prior year
    seasons = [[datetime(year,1,1),datetime(year,2,28)],
               [datetime(year,3,1),datetime(year,5,31)],
               [datetime(year,6,1),datetime(year,8,31)],
               [datetime(year,9,1),datetime(year,11,30)]]
    seasonlabels=['Summer (JF)',
                  'Autumn (MAM)',
                  'Winter (JJA)',
                  'Spring (SON)']

    pname1='Figs/GC/NO_maps_%4d.png'%year
    pname2='Figs/GC/NO_emiss_maps_%4d.png'%year
    pname3='Figs/GC/NO_regressions_%4d.png'%year


    region=pp.__AUSREGION__


    # Read omno2d
    OM_tropno2_all, OM_dates_all, OM_lats, OM_lons = fio.read_omno2d_year(year=year)


    ## FIG 1: 4rows 3cols
    #
    f1,axes1 = plt.subplots(4,3,figsize=(15,17))
    # plot scale limits
    rmin,rmax   = -50, 50           # limits for percent relative difference in plots
    amin,amax   = -1e15, 1e15       # limits for absolute diffs
    vmin,vmax   =  1e14, 14e14      # limits for molec/cm2 plots
    linear      = True              # linear or logarithmic scale

    ## FIG 2: 4rows, 2 cols
    #
    f2, axes2 = plt.subplots(4,2,figsize=(13,17))
    anthlinear = False
    soillinear = True
    soilmin,soilmax = 1e10, 6e10     # limits for soil no emissions
    anthmin,anthmax = 1e7,  1.5e11    # limits for anthro emiss
    anthtitle = 'anthropogenic NO'
    soiltitle = 'soil NO'

    ## FIG 3: 4 rows, 3 cols
    #
    f3, axes3 = plt.subplots(4,3,figsize=(15,17))

    # dot size
    regression_dot_size=30
    # font sizes
    tf, sf, lf  = 26, 22, 20

    # loop over seasons in year
    for i, [[d0,d1], seasonlabel] in enumerate(zip(seasons,seasonlabels)):
        #print(i, d0, d1, seasonlabel)

        # pull out season subset
        di = util.date_index(d0,OM_dates_all,d1)
        OM_tropno2 = np.copy(OM_tropno2_all[di])
        print('read OMNO2d ', seasonlabel,' ', OM_tropno2.shape)

        # Read satellite output from GC troprun
        # Keys needed for satellite tropNO2
        keys=['IJ-AVG-$_NO2','BXHGHT-$_BXHEIGHT','TIME-SER_AIRDEN','TR-PAUSE_TPLEV',]
        GC=GC_class.GC_sat(d0,dayN=d1,keys=keys)
        GC_tropno2=GC.get_trop_columns(['NO2'])['NO2']
        GC_lats,GC_lons=GC.lats,GC.lons

        # average over time, ignore warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            [GC_tropno2, OM_tropno2] = [ np.nanmean(arr,axis=0) for arr in [GC_tropno2, OM_tropno2] ]
        OM_tropno2_low=util.regrid_to_lower(OM_tropno2,OM_lats,OM_lons,GC.lats,GC.lons)
        assert OM_tropno2_low.shape == GC_tropno2.shape, 'Reduced OMI Grid should match GC'

        # FIG 1: OMI    |    GC    |  OMI - GC
        #

        # First plot GC map
        plt.sca(axes1[i,0])
        m,cs1,cb = pp.createmap(GC_tropno2,GC_lats,GC_lons,
                 colorbar=False,
                 linear=linear, vmin=vmin, vmax=vmax,
                     cmapname=map_cmap, region=region)
        if i == 0:
            plt.title('GC $\Omega_{NO_2}$', fontsize=tf)
        plt.ylabel(seasonlabel, fontsize=sf)

        # Plot the omi map
        plt.sca(axes1[i,1])
        pp.createmap(OM_tropno2_low, GC_lats, GC_lons,
                     colorbar=False,
                     linear=linear, vmin=vmin, vmax=vmax,
                     cmapname=map_cmap, region=region)
        if i == 0:
            plt.title('OMI $\Omega_{NO_2}$',fontsize=tf)
        # Plot differences and corellation
        # first abs diff
        reldiff = False
        plt.sca(axes1[i,2])
        if reldiff:
            m, cs2, cb2 = pp.createmap(100*(GC_tropno2-OM_tropno2_low)/OM_tropno2_low, GC_lats, GC_lons,
                         colorbar=False, clabel='%',
                         linear=True, vmin=rmin, vmax=rmax,
                         cmapname=dmap_cmap, region=region)
            if i == 0:
                plt.title('100(GC - OMI)/OMI', fontsize=tf)
            ticks=np.floor(np.linspace(rmin,rmax,5))
        else:
            m, cs2, cb2 = pp.createmap(GC_tropno2-OM_tropno2_low, GC_lats, GC_lons,
                         colorbar=False, clabel='molec/cm2',
                         linear=True, vmin=amin, vmax=amax,
                         cmapname=dmap_cmap, region=region)
            if i == 0:
                plt.title('GC - OMI', fontsize=tf)
            ticks=np.floor(np.linspace(amin,amax,5))

        # FIGURE 2: E_ANTHRO   |    E_SOIL

        # Read emissions from GC:
        GC_tavg=GC_class.GC_tavg(d0,d1,keys=['NO-SOIL_NO','ANTHSRCE_NO',], run='nochem')
        anthrono=GC_tavg.ANTHSRCE_NO
        soilno=GC_tavg.NO_soil
        GC_lats,GC_lons=GC_tavg.lats,GC_tavg.lons

        # remove near zero emissions squares
        for arr in [anthrono, soilno]:
            arr[arr < 1]=np.NaN

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            [anthrono, soilno] = [ np.nanmean(arr,axis=0) for arr in [anthrono, soilno] ]

        # plot the emissions:
        plt.sca(axes2[i,0])
        m,cs3,cb = pp.createmap(anthrono, GC_lats, GC_lons,
                     colorbar=False,
                     linear=anthlinear, vmin=anthmin, vmax=anthmax,
                     cmapname=reg_cmap, region=region)
        if i == 0:
            plt.title(anthtitle, fontsize=tf)
        plt.ylabel(seasonlabel,fontsize=sf)

        plt.sca(axes2[i,1])
        m,cs4,cb = pp.createmap(soilno, GC_lats, GC_lons,
                     colorbar=False,
                     linear=soillinear, vmin=soilmin, vmax=soilmax,
                     cmapname=reg_cmap, region=region)
        if i == 0:
            plt.title(soiltitle, fontsize=tf)


        # FIG 3: GC vs OMI, coloured by soil nox, then anthro and soil vs bias
        # now we must pull out the land data for corellations:
        subsets=util.lat_lon_subset(GC_lats,GC_lons,region,data=[GC_tropno2,OM_tropno2_low, anthrono, soilno])
        #lati,loni=subsets['lati'],subsets['loni']
        lats,lons=subsets['lats'],subsets['lons'] # GC_lats[lati],GC_lons[loni]
        [GC_tropno2,OM_tropno2_low, anthrono, soilno]=subsets['data']

        # Lets mask the oceans at this point:
        oceanmask=util.get_mask(GC_tropno2, lats = lats, lons = lons, maskocean=True)
        for arr in [GC_tropno2,OM_tropno2_low, anthrono, soilno]:
            arr[oceanmask] = np.NaN

        # bias between GC and OMI
        bias= (GC_tropno2 - OM_tropno2_low)
        colouring = (anthrono + soilno).flatten()
        clabel='emissions NO'
        # first coloured by anthro, then by soil
        plt.sca(axes3[i,0])
        cs5,cb = pp.plot_regression(OM_tropno2_low.flatten(),GC_tropno2.flatten(),
                           limsx=[vmin,vmax], limsy=[vmin,vmax],
                           logscale=False, legendfont=12, size=regression_dot_size,
                           showcbar=False, colours=colouring, clabel=clabel,
                           cmap=reg_cmap)
        if i==0:
            plt.title('$\Omega_{NO_2}$', fontsize=tf)
        plt.ylabel(seasonlabel+' \n GC', fontsize=lf)
        if i==3:
            plt.xlabel('OMI', fontsize=lf)

        # Then show regression with bias
        #plt.sca(axes3[i,1])
        #soilclabel = 'soil NO'
        #cs6, cb = pp.plot_regression(OM_tropno2_low.flatten(),GC_tropno2.flatten(),
        #                   limsx=[vmin,vmax], limsy=[vmin,vmax],
        #                   logscale=False, legendfont=12, size=regression_dot_size,
        #                   showcbar=False, colours=soilno.flatten(), clabel=soilclabel,
        #                   cmap=reg_cmap)
        #if i==0:
        #    plt.title('', fontsize=tf)
        #if i==3:
        #    plt.xlabel('OMI', fontsize=lf)

        #emiss vs GC
        plt.sca(axes3[i,1])
        pp.plot_regression(anthrono.flatten(), bias.flatten(),
                           logscale=False, diag=False,
                           legend=False, drawregression=False,
                           cmap=reg_cmap, showcbar=False,
                           colours=colouring,
                           size=regression_dot_size)
        if i == 0:
            plt.title('anthropogenic', fontsize=tf)
        if i == 3:
            plt.xlabel('emissions', fontsize=lf)


        plt.sca(axes3[i,2])
        pp.plot_regression(soilno.flatten(), bias.flatten(),
                           logscale=False, diag=False,
                           legend=False, legendfont=12, drawregression=False,
                           cmap=reg_cmap, showcbar=False,
                           colours=colouring,
                           size=regression_dot_size)
        if i == 0:
            plt.title('soil', fontsize=tf)
        if i == 3:
            plt.xlabel('emissions', fontsize=lf)

        plt.ylabel('GC - OMI',fontsize=lf)
        axes3[i,1].yaxis.tick_right()
        axes3[i,1].set_yticklabels(['']*10)
        axes3[i,2].yaxis.tick_right()
        axes3[i,2].yaxis.set_label_position("right")

    # FIG 1 Prettify
    # fix up plot spacing and colour bars
    plt.sca(axes1[0,0])
    plt.subplots_adjust(hspace=0,wspace=0) # remove space between maps
    # add colourbars at bottom
    f1.subplots_adjust(top=0.95)
    f1.subplots_adjust(right=0.9)
    f1.subplots_adjust(bottom=0.125)
    cbar_ax1 = f1.add_axes([0.15, 0.05, 0.4, 0.03])
    cb1=f1.colorbar(cs1,cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('$\Omega$ [NO cm$^{-2}$]', fontsize=lf)
    cbar_ax2 = f1.add_axes([0.675, 0.05, 0.2, 0.03])
    cb2=f1.colorbar(cs2,cax=cbar_ax2, orientation='horizontal')
    cb2.set_label('Difference [NO cm$^{-2}$]', fontsize=lf)
    cb2.set_ticks(ticks)

    plt.savefig(pname1)
    print('saved ',pname1)
    plt.close(f1)

    # FIG 2 PRETTIFY
    # fix up plot spacing and colour bars
    plt.sca(axes2[0,0])
    plt.subplots_adjust(hspace=0,wspace=0) # remove space between maps
    # add colourbars at bottom
    f2.subplots_adjust(top=0.95)
    f2.subplots_adjust(right=0.9)
    f2.subplots_adjust(bottom=0.125)
    cbar_ax1 = f2.add_axes([0.125, 0.05, 0.35, 0.03])
    cb3=f2.colorbar(cs3,cax=cbar_ax1, orientation='horizontal')
    cb3.set_label('Emission NO cm$^{-2}$ s$^{-1}$', fontsize=lf)
    cbar_ax2 = f2.add_axes([0.525, 0.05, 0.35, 0.03])
    cb4=f2.colorbar(cs4,cax=cbar_ax2, orientation='horizontal')
    cb4.set_label('Emission NO cm$^{-2}$ s$^{-1}$', fontsize=lf)
    #cb2.set_ticks(ticks)

    plt.savefig(pname2)
    print('saved ', pname2)
    plt.close(f2)

    # FIG 3 PRETTIFY
    plt.sca(axes3[0,0])
    # add colourbars at bottom

    f3.subplots_adjust(bottom=0.125)
    plt.subplots_adjust(hspace=.4, wspace=.1) # remove space between maps
    cbar_ax1 = f3.add_axes([0.3, 0.05, 0.4, 0.03])
    cb5=f2.colorbar(cs5,cax=cbar_ax1, orientation='horizontal')
    cb5.set_label(clabel, fontsize=lf)

    plt.savefig(pname3)
    print('Saved ',pname3)
    plt.close(f3)

# OLD VERSION OF TEST/Plots
#def GCe_vs_OMNO2d(d0=datetime(2005,1,1), d1=None,
#             region=pp.__AUSREGION__, regionlabel='AUS',
#             soil=True, dstr_lab=None,
#             map_cmap='PuRd' ,reg_cmap='YlOrRd',dmap_cmap='RdBu_r'):
#    '''
#        Plot regressions between soil NO vs bias from GC-OMI
#            also anthro nox can be plotted if soil==False
#    '''
#
#    # Set up time bounds: one month if d1 is missing
#    if d1 is None:
#        d1=util.last_day(d0)
#    dstr="%s-%s"%(d0.strftime("%Y%m%d"),d1.strftime("%Y%m%d"))
#    if dstr_lab is None:
#        dstr_lab=dstr
#
#    ## Read our data:
#
#    # Read omno2d
#    OMNO2d,OM_attrs=fio.read_omno2d(day0=d0,dayN=d1)
#    OM_tropno2=OMNO2d['tropno2']
#    OM_lats=OMNO2d['lats']
#    OM_lons=OMNO2d['lons']
#
#    # Read satellite output from GC troprun
#    # Keys needed for satellite tropNO2
#    keys=['IJ-AVG-$_NO2','BXHGHT-$_BXHEIGHT','TIME-SER_AIRDEN','TR-PAUSE_TPLEV',]
#    GC=GC_class.GC_sat(d0,dayN=d1,keys=keys)
#    GC_tropno2=GC.get_trop_columns(['NO2'])['NO2']
#    GC_lats,GC_lons=GC.lats,GC.lons
#
#    # Read emissions from GC:
#    GC_tavg=GC_class.GC_tavg(d0,d1,keys=['NO-SOIL_NO','ANTHSRCE_NO',], run='nochem')
#    anthrono=GC_tavg.ANTHSRCE_NO
#    soilno=GC_tavg.NO_soil
#
#    # remove near zero emissions squares
#    for arr in [anthrono, soilno]:
#        arr[arr < 1]=np.NaN
#
#    # average over time:
#    to_average = [anthrono, soilno ,GC_tropno2, OM_tropno2]
#
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore", category=RuntimeWarning)
#        [anthrono, soilno ,GC_tropno2, OM_tropno2] = [ np.nanmean(arr,axis=0) for arr in to_average ]
#    OM_tropno2_low=util.regrid_to_lower(OM_tropno2,OM_lats,OM_lons,GC.lats_e,GC.lons_e)
#    assert OM_tropno2_low.shape == GC_tropno2.shape, 'Reduced OMI Grid should match GC'
#    ## PLOTTING
#    ##
#
#    # plot scale limits
#    rmin,rmax=-50,50 # limits for percent relative difference in plots
#    amin,amax=-1e15,1e15 # limits for absolute diffs
#    vmin,vmax=1e14,14e14 # limits for molec/cm2 plots
#    linear=True # linear or logarithmic scale
#    elinear=[False,True][soil] # for emissions
#    soilmin,soilmax=1e10,6e10 # limits for soil no emissions
#    anthmin,anthmax=1e5,1.5e11 # limits for anthro emiss
#    emiss=[anthrono,soilno][soil]
#    emimin,emimax=[[anthmin,anthmax],[soilmin,soilmax]][soil]
#    emisstitle=['$E_{anthro NO}$','$E_{soil NO}$'][soil]
#    regression_dot_size=30
#
#    # set up axes for 3,3,2
#    fig = plt.figure(figsize=(15,17))
#    axes=[ plt.subplot(331), plt.subplot(332), plt.subplot(333),
#           plt.subplot(334), plt.subplot(335), plt.subplot(336),
#                   plt.subplot(325), plt.subplot(326) ]
#
#    # First plot GC map
#    plt.sca(axes[0])
#    m,cs,cb = pp.createmap(GC_tropno2,GC_lats,GC_lons,
#                 title='$\Omega_{GEOS-Chem NO2}$', colorbar=False,
#                 linear=linear, vmin=vmin,vmax=vmax,
#                 cmapname=map_cmap, region=region)
#
#    # Add colorbar to the right:
#    divider = make_axes_locatable(axes[0])
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    fig.colorbar(cs,cax=cax, orientation='vertical')
#
#    # Plot the omi map
#    plt.sca(axes[1])
#    pp.createmap(OM_tropno2_low, GC_lats, GC_lons,
#                 title='$\Omega_{OMI NO2}$', colorbar=False,
#                 linear=linear, vmin=vmin,vmax=vmax,
#                 cmapname=map_cmap, region=region)
#
#    # plot the emissions:
#    plt.sca(axes[2])
#    pp.createmap(emiss, GC_lats, GC_lons,
#                 title=emisstitle, colorbar=True,
#                 linear=elinear, vmin=emimin, vmax=emimax,
#                 cmapname=reg_cmap, region=region)
#
#    # Plot differences and corellation
#    # first abs diff
#    plt.sca(axes[3])
#    pp.createmap(GC_tropno2-OM_tropno2_low, GC_lats, GC_lons,
#                 title='GC - OMI', colorbar=True, clabel='molec/cm2',
#                 linear=True, vmin=amin, vmax=amax,
#                 cmapname=dmap_cmap, region=region)
#
#    # then rel diff
#    plt.sca(axes[4])
#    pp.createmap(100*(GC_tropno2-OM_tropno2_low)/OM_tropno2_low, GC_lats, GC_lons,
#                 title='100(GC - OMI)/OMI', colorbar=True, clabel='%',
#                 linear=True, vmin=rmin, vmax=rmax,
#                 cmapname=dmap_cmap, region=region)
#
#
#    # now we must pull out the land data for corellations:
#
#    subsets=util.lat_lon_subset(GC_lats,GC_lons,region,data=[GC_tropno2,OM_tropno2_low, emiss])
#    lati,loni=subsets['lati'],subsets['loni']
#    lats,lons=subsets['lats'],subsets['lons'] # GC_lats[lati],GC_lons[loni]
#    [GC_tropno2,OM_tropno2_low, emiss]=subsets['data']
#
#    # Lets mask the oceans at this point:
#    oceanmask=util.get_mask(GC_tropno2, lats=lats,lons=lons,maskocean=True)
#    for arr in [GC_tropno2,OM_tropno2_low, emiss]:
#        arr[oceanmask] = np.NaN
#
#    # bias between GC and OMI
#    bias= (GC_tropno2 - OM_tropno2_low)
#
#    # then corellations
#    plt.sca(axes[5])
#    colours = emiss
#    clabel = 'molec/cm2/s'
#    pp.plot_regression(OM_tropno2_low.flatten(),GC_tropno2.flatten(),
#                       limsx=[vmin,vmax], limsy=[vmin,vmax],
#                       logscale=False, legendfont=12, size=regression_dot_size,
#                       showcbar=True, colours=colours.flatten(), clabel=clabel,
#                       cmap=reg_cmap)
#    plt.title('GC vs OMI')
#    plt.ylabel('GC')
#    plt.xlabel('OMI')
#
#    #emiss vs GC
#    plt.sca(axes[6])
#    pp.plot_regression(emiss.flatten(), GC_tropno2.flatten(),
#                       logscale=False, legendfont=12, diag=False,
#                       cmap=reg_cmap, showcbar=False,
#                       colours=colours.flatten(),
#                       size=regression_dot_size)
#
#    plt.title('GEOS-Chem $E_{NO}$ vs $\Omega_{NO2}$')
#    plt.xlabel(['anthro','soil'][soil])
#    plt.ylabel('GC')
#
#    #emiss vs bias
#    plt.sca(axes[7])
#    pp.plot_regression(emiss.flatten(), bias.flatten(),
#                       logscale=False, legendfont=12, diag=False,
#                       cmap=reg_cmap, showcbar=False,
#                       colours=colours.flatten(),
#                       size=regression_dot_size)
#
#    plt.title('$E_{NO}$ vs (GC-OMI)')
#    plt.xlabel('emissions')
#    plt.ylabel('GC-OMI')
#
#    emisstype=['anthro','soil'][soil]
#    pname='Figs/GC/GC%s_vs_OMNO2_%s_%s.png'%(emisstype,regionlabel,dstr)
#    plt.suptitle('GEOS-Chem vs OMINO2d %s'%dstr_lab, fontsize=32)
#    plt.savefig(pname)
#    print('Saved ',pname)
#    plt.close()


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

def hcho_lifetime(year=2005):
    '''
    Use tau = HCHO / Loss    to look at hcho lifetime over the year
    '''
    d0=datetime(year,1,1)
    dtest=datetime(year,1,31)
    dN=datetime(year,12,31)

    # read hcho and losses from trac_avg
    # ch20 in ppbv, air density in molec/cm3,  Loss HCHO mol/cm3/s
    keys=['IJ-AVG-$_CH2O','BXHGHT-$_N(AIR)', 'PORL-L=$_LHCHO']
    run = GC_class.GC_tavg(d0,dtest, keys=keys) # [time, lat, lon, lev]
    hcho = run.hcho[:,:,:,0] # [time, lat, lon, lev=47] @ ppbv
    N_air = run.N_air[:,:,:,0] # [time, lat, lon, lev=47] @ molec/cm3
    Lhcho = run.Lhcho[:,:,:,0] # [time, lat, lon, lev=38] @ mol/cm3/s  == molec/cm3/s !!!!


    #N_Avogadro = molec/mol
    # [ppbv * 1e-9 * (molec air / cm3) / (molec/cm3/s)] = s
    tau = hcho * 1e-9 * N_air  /  Lhcho
    print (np.mean(hcho), np.mean(N_air), np.mean(Lhcho))
    # [time, lat, lon, lev]
    # just want surface I guess?
    print(tau.shape)
    print(np.mean(tau), np.min(tau), np.max(tau), '(in seconds)')
    # change to hours
    tau=tau/3600.
    print(np.mean(tau), np.min(tau), np.max(tau), '(in hours)')

    pp.createmap(np.nanmean(tau,axis=0),lats=run.lats, lons=run.lons, aus=True,
                 pname='tau_test.png',linear=True, title="HCHO lifetime in hours ")

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


def hcho_lifetime_from_slope(d0=datetime(2005,1,1), dN=datetime(2005,12,31)):
    '''
        Read midday slope:
            assume yields range from 0.2 to 0.4 and look at area averaged lifetimes
        
        return: multi-year avg time series - one for each subregion
    '''
    # read GEOS-Chem midday slopes
    pname='Figs/GC/HCHO_lifetime.png'
    data, attrs = masks.read_smearmask(d0,dN,keys=['slope','smearmasklit'])
    lats=data['lats']
    lons=data['lons']
    dates=data['dates']
    slope=data['slope']
    mask=data['smearmasklit']

    yield_min       = 0.2 # minimum expected yield
    yield_max       = 0.4 # ...
    Yield_const     = [yield_min, yield_max]
    tau             = []
    # remove ocean squares
    oceanmask       = util.oceanmask(lats,lons)
    oceanmask       = np.repeat(oceanmask[np.newaxis,:,:], len(dates), axis=0)
    for i in range(2):
        tau.append( slope/Yield_const[i] / 3600.) # seconds -> hours
        tau[i][mask>0]  = np.NaN
        tau[i][tau[i]<0]   = np.NaN
        tau[i][oceanmask]  = np.NaN
    
    slope[mask>0]   = np.NaN # mask applied
    slope[slope<=0]  = np.NaN # remove negatives...
    slope[slope>10000] = np.NaN # extra mask..

    Ylims           = [0,18]   # hours
    Yticks          = [3,7,11,15]

    # Want to look at timeseires and densities in these subregions:
    regions=__subregions__
    colors=__subregions_colors__
    labels=__subregions_labels__

    # k or tau[oceanmask]    = np.NaN

    # multi-year monthly averages
    myat    = [ util.multi_year_average_regional(taui,dates,lats,lons,grain='monthly',regions=regions) for taui in tau ]
    myaS    = util.multi_year_average_regional(slope,dates,lats,lons,grain='monthly',regions=regions)
    
    # pull out dataframe list (one for each region)
    dft     = [ myati['df'] for myati in myat ]
    dfS     = myaS['df']

    x=range(12)
    n=len(dft[0])
    f,axes = plt.subplots(n,1,figsize=[16,12], sharex=True,sharey=True)
    for i in range(n):
        ax=axes[i]
        plt.sca(ax)
        # two data frame lines to draw
        print("TEST")
        print( type(myat[0]['df']), type(myat[0]['df'][0]))
        meant       = [ dfti[i].mean().values.squeeze() for dfti in dft ]
        # upper and lower quartiles now based on lower,higher yields
        uqt         = dft[0][i].quantile(0.75).values.squeeze()
        lqt         = dft[1][i].quantile(0.25).values.squeeze()
        
        if i == 2:
            plt.ylabel('lifetime [h]', fontsize=22)

        plt.fill_between(x, lqt,uqt, color=colors[i], alpha=0.5, facecolor=colors[i], linewidth=0)
        for meantau in meant:
            plt.plot(x, meantau, color=colors[i], linestyle='-',linewidth=2)
        plt.ylim(Ylims)
        plt.yticks(Yticks)
        plt.title(labels[i], color=colors[i], fontsize=24)

        #if i==0:
        #    plt.legend(loc='best')
        #if i%2 == 1:
        #    axes[i].yaxis.set_label_position("right")
        #    axes[i].yaxis.tick_right()
    plt.sca(axes[-1])
    plt.ylim(Ylims)
    plt.yticks(Yticks)
    plt.xlim([-0.5,11.5])
    plt.xticks(x)
    plt.gca().set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    plt.xlabel('month')
    plt.suptitle('estimated HCHO lifetime',fontsize=30)
    f.subplots_adjust(hspace=.3)


    ## save figure
    plt.savefig(pname)
    print("Saved %s"%pname)
    plt.close()

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

    dlims={ 'hcho'      :(-1e13,1e13),
            'hcho_tc'   :(-1e14,1e14),
            'OH'        :(-2e10,2e10),
            'OH_tc'     :(-2e11,2e11),
            'isop'      :(-1e15,1e15),
            'isop_tc'   :(-1e16,1e16),
            'O3'        :(-5e15,5e15),
            'O3_tc'     :(-1e18,1e18),}

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
            'O3_tc'     :[5e18,6e18,7e18,8e18,9e18,1e19]}

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
        print('making plot for ', key)
        # compare surface and troposphere
        # First do total column

        dats   =  [np.nanmean(tot_satu[key],axis=0), np.nanmean(tot_satt[key],axis=0)] # average over time
        pname  = 'Figs/GC/UCX_vs_trp_middaytotcol_%s_%s.png'%(dstr,key)
        titles = ['UCX', 'tropchem']
        pp.compare_maps(dats,[lats,lats],[lons,lons],pname=pname,titles=titles,
                        vmin=vlims[key+'_tc'][0], vmax=vlims[key+'_tc'][1], # total column has own scale
                        ticks=ticks.get(key+'_tc',None),
                        rmin=rlims[key][0], rmax=rlims[key][1],
                        amin=dlims[key+'_tc'][0], amax=dlims[key+'_tc'][1],
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
                        suptitle='%s surface (middays %s) [%s]'%(str.upper(key),dstr,units[key]))

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
                        amin=dlims[key+'_tc'][0], amax=dlims[key+'_tc'][1],
                        linear=False, region=region,
                        suptitle='%s Total column (daily mean %s) [%s]'%(str.upper(key),dstr,'molec cm$^{-2}$'))

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
                        suptitle='%s surface amounts (daily mean %s) [%s]'%(str.upper(key),dstr,units[key]))

def AMF_comparison_tc_ucx(month=datetime(2005,1,1),max_procs=4):
    '''
    Look at monthly averaged AMF using UCX and using tropchem
    '''
    ## read tropchem and ucx run
    #
    d0 = month
    dN = util.last_day(month)
    dates = util.list_days(datetime(2005,1,1), datetime(2005,1,2))

    #ucx=GC_class.GC_sat(d0,dN,run='UCX') # [days, lats, lons,72]
    #trop=GC_class.GC_sat(d0,dN,run='tropchem') # [days, lats, lons, 47]
    # read one day at a time....
    start=timeit.default_timer()
    ## To calculate AMFs we need scattering weights from satellite swath files
    #
    lats,lons,amfu,amft = [],[],[],[]
    print("TESTING: NEED TO DO WHOLE MONTH")
    for i, d in enumerate(dates):
        omhcho  = reprocess.get_good_pixel_list(d,getExtras=True)
        ucx     = GC_class.GC_sat(d, run='UCX')
        trop    = GC_class.GC_sat(d, run='tropchem')
        w       = omhcho['omega'][:,0::50] # [47, 677...]
        w_pmids = omhcho['omega_pmids'][:,0::50] #[47,677...]
        AMF_G   = omhcho['AMF_G'][0::50] # [677...]
        lat     = omhcho['lat'][0::50]   # [677...]
        lon     = omhcho['lon'][0::50]   # [677...]
    

        print("Running amf calc for tropchem")
        omegain,pmidsin,amfgin,latin,lonin=[],[],[],[],[]
        for j in range(len(AMF_G)): # just do one pixel from each 50

            omegain.append(w[:,j])
            pmidsin.append(w_pmids[:,j])
            amfgin.append(AMF_G[j])
            latin.append(lat[j])
            lonin.append(lon[j])
            lats.extend(lat)
            lons.extend(lon)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            #trop_amf_z, trop_amf_s = trop.calculate_AMF(...)
            #ucx_amf_z, ucx_amf_s = ucx.calculate_AMF(w[:,j], w_pmids[:,j], AMF_G[j], lat[j], lon[j], plotname=None, debug_levels=False)
            tropreturns=executor.map(trop.calculate_AMF,omegain,pmidsin,amfgin,latin,lonin)
            ucxreturns=executor.map(ucx.calculate_AMF,omegain,pmidsin,amfgin,latin,lonin)
            amft.extend([a for a,b in tropreturns])
            amfu.extend([a for a,b in ucxreturns])


    end=timeit.default_timer()
    print("TIMEIT: took %6.2f minutes to run amf comparison"%(end-start/60.0))

    # lets save since it takes so long to run...
    fio.save_to_hdf5(month.strftime('Data/amfs_trop_ucx_%Y%m.h5'),
                     arraydict={'amfu':amfu,'amft':amft,'lats':lats,'lons':lons,},
                     attrdicts={'amfu':{'desc':'list of AMFs from UCX model output'},
                                'amft':{'desc':'list of AMFs from tropchem model output'},})

    #save_to_hdf5(outfilename, arraydict, fillvalue=np.NaN, attrdicts={}, fattrs={},verbose=False)

    plt.figure(figsize=(14,10))
    pp.density(np.array(amfu),color='m',label="UCX")
    pp.density(np.array(amft),color='k',label="tropchem")
    plt.title("AMF distributions")
    plt.savefig(month.strftime("AMF_dists_trop_vs_ucx_%Y%m.png"))
    plt.close()



    return




#def OLD_compare_tc_ucx(date=datetime(2005,1,1),extra=False,fnames=None,suffix=None):
#    '''
#        Check UCX vs Tropchem
#        set Extra to true to look at E_isop_biog and OH (uses specific output file 20050101
#        set fnames=[tropchem.nc,ucx.nc] outputs to test specific files
#    '''
#    ymstr=date.strftime("%Y%m")
#    region=pp.__GLOBALREGION__
#
#    # Read the netcdf files (output specified for this test)
#    # UCX FILE:
#    #ucx=GC_output(date,UCX=True, fname='UCX_trac_avg_20050101.nc')
#    if fnames is None:
#        ucx=GC_output(date,UCX=True)
#        trp=GC_output(date,UCX=False,monthavg=True)
#    else:
#        trp_fname,ucx_fname=fnames#
#        trp=GC_output(date,UCX=False, monthavg=True, fname=trp_fname)
#        ucx=GC_output(date,UCX=True, fname=ucx_fname)
#
#    lats_all=ucx.lats
#    lons_all=ucx.lons
#
#    all_keys=['hcho','isop','E_isop_bio','OH','O3','NO2']
#    keys=[]
#    for key in all_keys:
#        if hasattr(ucx,key) and hasattr(trp,key): # look at fields in both outputs
#            keys.append(key)
#            print('%s will be compared at surface '%(key))
#    #keys=['hcho','isop']
#    units={ 'hcho'      :'ppbv',        #r'molec cm$^{-2}$',
#            'E_isop_bio':r'atom C cm$^{-2}$ s$^{-1}$',
#            'OH'        :r'molec cm$^{-3}$',
#            'isop'      :r'ppbv',
#            'O3'        :r'ppbv',
#            'NO2'       :r'ppbv',}
#    cbarfmt={}; cbarxtickrot={}
#    rlims={'hcho'       :(-20,20),
#           'E_isop_bio' :(-50,50),
#           'OH'         :(-50,50),
#           'isop'       :(-70,70),
#           'O3'         :(-50,50),
#           'NO2'        :(-50,50),}
#    dlims={'hcho'       :(-.5,.5),
#           'E_isop_bio' :(-1e12,1e12),
#           'OH'         :(-8e5,8e5),
#           'isop'       :(-6,6),
#           'O3'         :(-40,40),
#           'NO2'        :(-100,100),}
#    alims={'hcho'       :(None,None),
#           'E_isop_bio' :(None,None),
#           'OH'         :(1e6,4e6),
#           'isop'       :(None,None),
#           'O3'         :(None,None),
#           'NO2'        :(None,None),}
#    for key in keys:
#        cbarfmt[key]=None; cbarxtickrot[key]=None
#    cbarfmt['OH']="%.1e"; cbarxtickrot['OH']=30
#
#    ucx_data=ucx.get_field(keys=keys,region=region)
#    trp_data=trp.get_field(keys=keys,region=region)
#    lats=ucx_data['lats'];lons=ucx_data['lons']
#
#    print("Region: ",region)
#    for key in keys:
#        print(key)
#        dats=[ucx_data[key], trp_data[key]]
#        for di,dat in enumerate(dats):
#            pre=['UCX  ','trop '][di]
#            print(pre+"%s shape: %s"%(key,str(np.shape(dat))))
#            print("    min/mean/max: %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
#            # Just look at surface from here
#            if len(dat.shape)==3:
#                dat=dat[0]
#                print("    at surface  : %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
#                dats[di]=dat
#
#        # whole tropospheric column
#        #data['tc']  = tc.get_trop_columns(keys=keys)
#        #data['ucx'] = ucx.get_trop_columns(keys=keys)
#
#        # Plot values and differences for each key
#        suffix=[suffix,''][suffix is None]
#        pname='Figs/GC/UCX_vs_trp_glob_%s_%s%s.png'%(trp.dstr,key,suffix)
#        u=dats[0];t=dats[1]
#        amin,amax = alims[key]
#        rmin,rmax = rlims[key]
#        dmin,dmax = dlims[key]
#        args={'region':region,'clabel':units[key], 'vmin':amin, 'vmax':amax,
#              'linear':True, 'cmapname':'PuRd', 'cbarfmt':cbarfmt[key],
#              'cbarxtickrot':cbarxtickrot[key]}
#
#        f,axes=plt.subplots(2,2,figsize=(16,14))
#
#        plt.sca(axes[0,0])
#        pp.createmap(u,lats,lons, title="%s UCX"%key, **args)
#
#        plt.sca(axes[0,1])
#        pp.createmap(t,lats,lons, title="%s tropchem"%key, **args)
#
#        plt.sca(axes[1,0])
#        args['vmin']=dmin; args['vmax']=dmax
#        args['cmapname']='coolwarm'
#        pp.createmap(u-t,lats,lons,title="UCX - tropchem", **args)
#
#        plt.sca(axes[1,1])
#        args['vmin']=rmin; args['vmax']=rmax; args['clabel']='%'
#        pp.createmap((u-t)*100.0/u, lats, lons, title="100*(UCX-tc)/UCX",
#                     suptitle='%s %s %s'%('surface', key, trp.dstr), pname=pname, **args)





def model_slope_series(d0=datetime(2005,1,1),dN=datetime(2012,12,31), latlon=pp.__cities__['Syd'], loclabel='Syd'):
    '''
        look at slope before/after SF and using MYA for each month
    '''
    # set up plot limits
    slopemin=500
    slopemax=9100
    nmin, nmax = 0, 31
    rmin, rmax = -0.1,1.1
    rfilt, nfilt = 0.4, 10
    smearmin = masks.__smearminlit__
    smearmax = masks.__smearmaxlit__

    # plot names
    dstr=d0.strftime('%Y%m%d')+'-'+dN.strftime('%Y%m%d')
    pname = 'Figs/GC/slope_series_%s_%s.png'%(loclabel, dstr)

    # first read biogenic model month by month and save the slope, and X and Y for wollongong
    slopedata, slopeattrs = fio.read_slopes(d0,dN)
    allmonths = util.list_months(d0,dN)
    allyears  = util.list_years(d0,dN)

    lat,lon   = latlon
    lats,lons = slopedata['lats'],slopedata['lons']
    lati,loni = util.lat_lon_index(lat,lon,lats,lons)
    # update lat,lon to match square centre
    lat, lon  = lats[lati], lons[loni]

    # Y=slope*X+b with regression coeff r
    r     = slopedata['r'][:,lati,loni]
    r_sf  = slopedata['r_sf'][:,lati,loni]
    ci    = slopedata['ci'][:,lati,loni]
    ci_sf = slopedata['ci_sf'][:,lati,loni]
    s     = slopedata['slope'][:,lati,loni]
    s_sf  = slopedata['slope_sf'][:,lati,loni]
    n     = slopedata['n'][:,lati,loni]
    n_sf  = slopedata['n_sf'][:,lati,loni]

    # multiyear averaged:
    n_mya, n_sf_mya   = slopedata['n_mya'][:,lati,loni],slopedata['n_sf_mya'][:,lati,loni]
    ci_mya, ci_sf_mya = slopedata['ci_mya'][:,lati,loni],slopedata['ci_sf_mya'][:,lati,loni]
    s_mya, s_sf_mya   = slopedata['slope_mya'][:,lati,loni],slopedata['slope_sf_mya'][:,lati,loni]
    r_mya, r_sf_mya   = slopedata['r_mya'][:,lati,loni],slopedata['r_sf_mya'][:,lati,loni]

    print('  normal   ,    sf    ,     mya,       sf_mya    ')
    for arrs in [[s,s_sf, s_mya, s_sf_mya],[r,r_sf,r_mya, r_sf_mya], [ci,ci_sf, ci_mya, ci_sf_mya], [n,n_sf, n_mya, n_sf_mya]]:
        for i in range(min(np.shape(arrs[0])[0],5)): # up to 5 thingies
            print(arrs[0][i],arrs[1][i],arrs[2][i], arrs[3][i])
        #print(arrs[0]-arrs[1])


    plt.subplots(figsize=(18,18))
    ax11=plt.subplot(4,2,1)
    # first plot slope as time series, +- confidence interval
    plt.plot_date(allmonths,s, 'b-', linewidth=3)
    plt.fill_between(allmonths, ci[:,0], ci[:,1], alpha=.6)
    plt.ylim(slopemin,slopemax)
    plt.ylabel('slope [s]')
    plt.title('unfiltered',fontsize=24)


    # also do sf version
    ax12=plt.subplot(4,2,2, sharex=ax11, sharey=ax11)
    plt.plot_date(allmonths,s_sf, '-', linewidth=3)
    plt.fill_between(allmonths, ci_sf[:,0], ci_sf[:,1], alpha=.6)
    plt.ylim(slopemin,slopemax)
    plt.plot([allmonths[0], allmonths[-1]], [smearmin,smearmin], 'b--')
    plt.plot([allmonths[0], allmonths[-1]], [smearmax,smearmax], 'b--')
    ax12.yaxis.set_label_position("right")
    ax12.yaxis.tick_right()
    plt.title('smear filtered',fontsize=24)
    # Now set the ticks and labels
    for ax in [ax11,ax12]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # then plot r and count
    ax21=plt.subplot(4,2,3, sharex=ax11)
    plt.plot_date(allmonths,r, '-', color='r',linewidth=2)
    plt.ylabel('r',color='r')
    plt.ylim(rmin,rmax)
    plt.sca(plt.twinx())
    plt.plot_date(allmonths,n, '-', color='m')
    plt.ylim(nmin,nmax)
    plt.setp(plt.gca().get_yticklabels(),visible=False)

    # also for sf
    ax22=plt.subplot(4,2,4, sharex=ax21, sharey=ax21)
    plt.plot_date(allmonths,r_sf, '-', color='r',linewidth=2)
    plt.plot([allmonths[0], allmonths[-1]], [rfilt,rfilt], 'r--')
    plt.setp(ax22.get_yticklabels(),visible=False)
    plt.ylim(rmin,rmax)
    plt.sca(plt.twinx())
    plt.plot_date(allmonths,n_sf, '-', color='m')
    plt.plot([allmonths[0], allmonths[-1]], [nfilt,nfilt], 'm--')
    plt.ylabel('n',color='m')
    plt.ylim(nmin,nmax)

    # now we do xlabels
    for ax in [ax21,ax22]:
        ax.set_xticks(allyears)
        ax.set_xticklabels([ i.strftime("%Y") for i in allyears], rotation=20)

    # plot mya version of slopes...
    ax31 = plt.subplot(4,2,5)
    plt.plot(range(12),s_mya, linewidth=3,color='b')
    plt.fill_between(range(12), ci_mya[:,0], ci_mya[:,1], alpha=.6)
    plt.ylim(slopemin,slopemax)

    # plot mya sf
    ax32 = plt.subplot(4,2,6, sharex=ax31, sharey=ax31)
    plt.plot(range(12),s_sf_mya, linewidth=3, color='b')
    plt.fill_between(range(12), ci_sf_mya[:,0], ci_sf_mya[:,1], alpha=.6)
    plt.plot([0, 11], [smearmin,smearmin], 'b--')
    plt.plot([0, 11], [smearmax,smearmax], 'b--')
    plt.ylabel('slope [s]')
    # Now set the ticks and labels
    ax32.yaxis.set_label_position("right")
    ax32.yaxis.tick_right()
    for ax in [ax31,ax32]:
        plt.setp(ax.get_xticklabels(), visible=False)


    # then plot r and count
    ax41 = plt.subplot(4,2,7,sharex=ax32)
    plt.plot(range(12),r_mya, 'r-',linewidth=2)
    plt.ylim(rmin,rmax)
    plt.ylabel('r',color='r')
    plt.sca(plt.twinx())
    plt.plot_date(range(12), n_mya, 'm-')
    plt.ylim(100,300)
    plt.setp(plt.gca().get_yticklabels(),visible=False)

    # also for sf
    ax42 = plt.subplot(4, 2, 8, sharex=ax41)
    plt.plot(range(12),r_sf_mya, 'r-',linewidth=2)
    plt.plot([0, 11], [rfilt,rfilt], 'r--')
    plt.ylim(rmin,rmax)
    plt.setp(plt.gca().get_yticklabels(),visible=False)
    plt.sca(plt.twinx())
    plt.plot_date(range(12), n_sf_mya, 'm-')
    plt.ylim(100,300)
    plt.ylabel('n',color='m')
    # Now set the ticks and labels
    for ax in [ax41,ax42]:
        ax.set_xticks(range(12))
        ax.set_xticklabels([ 'j','f','m','a','m','j','j','a','s','o','n','d' ])
        ax.set_xlim(-0.5,11.5)

    # finally save figure
    latlonstr="%.3f,%.3f"%(lat, lon)
    plt.suptitle('Model Slope %s'%latlonstr,fontsize=28)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(pname)
    print('saved ',pname)
    plt.close()


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

    ## Look at HCHO lifetime over Australia
    #
    #hcho_lifetime(2005)
    #hcho_lifetime_from_slope()

    ## new vs old diagnostics
    #check_old_vs_new_emission_diags()

    ## LOOK AT NO vs GEOS-Chem emissions
    #for year in np.arange(2005,2010):
    #    GCe_vs_OMNO2d(year=year)
    
    ## EXAMINE AMFs
    #check_amf_differences()
    
    ## UCX VS TROPCHEM AMF
    #
    #AMF_comparison_tc_ucx()

    ## tropchem vs UCX plots
    # Look at 2007 summer since I have OH for daily avg files from then.
    #compare_tc_ucx(datetime(2007,1,1),util.last_day(datetime(2007,2,1)))

    # Checking units:

    ## Check slope
    #for latlon, label in zip([pp.__cities__['Syd'], pp.__cities__['Mel'], [-16,135]],['Syd','Mel','Nowhere']):
    #    model_slope_series(latlon=latlon, loclabel=label)

    
    #check_rsc_interp()   # last run 29/5/18

    #HCHO_vs_temp(d0=d0,d1=d1,
    #             region=SEA,regionlabel='SEA',
    #             regionplus=pp.__AUSREGION__)

    # Test the function with 3 days of data
    #GC_vs_OMNO2d(d0=d0, d1=d1,
    #             region=region, regionlabel=label,
    #             drop_low_anthro=True)

    #
    for dates,dstr in zip([sum05,spr05,win05,aut05],dstrs):

        #GC_vs_OMNO2d(d0=dates[0], d1=dates[1],
        #             region=region, regionlabel=label)

        for region, label in zip(subs,labels):
            HCHO_vs_temp(d0=dates[0],d1=dates[1],
                         region=region,regionlabel=label)
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
