#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests and analysis involving filtering: smoke, fire, NOx, and smearing


History:
    Created on Thu Apr 26 10:11:43 2018

@author: jesse
"""
#########################################
############# Modules ###################
#########################################

# don't actually display any plots, just create them
import matplotlib
matplotlib.use('Agg')

# Local stuff
from utilities import fio
from utilities import plotting as pp
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes.omhchorp import omhchorp
from classes.GC_class import GC_tavg, GC_sat, Hemco_diag
from classes.E_new import E_new
from utilities.plotting import __AUSREGION__
from Inversion import smearing, __Thresh_Smearing__ # smearing filter creation

# General stuff
import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy
import random

from datetime import datetime#, timedelta
from os.path import isfile

# Plotting libraries
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm # for lognormal colour bar
#import matplotlib.patches as mpatches
import seaborn # kdeplots



#########################################
############# Globals ###################
#########################################

__Thresh_NO2_d__ = fio.__Thresh_NO2_d__ # 1e15 # daily threshhold
__Thresh_NO2_y__ = fio.__Thresh_NO2_y__ # 1.5e15 # yearly avg threshhold
__Thresh_AAOD__  = fio.__Thresh_AAOD__  # 0.03 # AAOD smoke threshhold
__Thresh_fires__ = fio.__Thresh_fires__ # 1 # active fire pixel count threshhold

cities=pp.__cities__

# Text file for text output
__no2_txt_file__ = 'no2_output.txt'

# Want to look at timeseires and densities in these subregions:
__subzones__ = pp.__subzones_AUS__
__colors__ = pp.__subzones_colours__



#########################################
############# Functions #################
#########################################

def summary_pixels_filtered():
    '''
        List by year how many usable pixels are filtered, also show yearly cycle of available and filtered
    '''
    d0=datetime(2005,1,1)

    pixx=[]
    for year in util.list_years(d0,datetime(2013,1,1)):
        if not isfile('Data/Isop/E_new/emissions_%s.h5'%year.strftime("%Y%m")):
            break
        Enew=E_new(datetime(year.year,1,1),datetime(year.year,12,31),dkeys=['pixels_u','firefilter','anthrofilter'])

        pix  = Enew.pixels_u # unfiltered pixel counts
        pix[Enew.oceanmask3d]=0
        pixa=np.copy(pix)
        pixb=np.copy(pix)
        pixc=np.copy(pix)
        pixa[Enew.firefilter]=0
        pixb[Enew.anthrofilter]=0
        pixc[Enew.firefilter]=0
        pixc[Enew.anthrofilter]=0
        tot=np.sum(pix)
        tota=np.sum(pixa)
        totb=np.sum(pixb)
        totc=np.sum(pixc)
        pixx.append((year.year, tot, tot-tota, 100*(tot-tota)/tot, tot-totb, 100*(tot-totb)/tot , tot-totc, 100*(tot-totc)/tot))


    print("Pixels removed by filtering:")
    print("year   ,   pixels    ,  fire               ,    anthro            ,    both "  )
    for pixs in pixx:
        print("%4d   &  %5.1e    &  %5.1e(%4.1f\\%%)     &    %5.1e(%4.1f\\%%)    &    %5.1e(%4.1f\\%%) \\\\"%pixs )

def check_smoke_filtering(d0=datetime(2005,1,1), dn=datetime(2005,12,31)):
    '''
    '''
    Enew = E_new(d0,dn,dkeys=['pixels_u','pixels','firefilter','anthrofilter',])
    lats = Enew.lats;  lons=Enew.lons
    firesum = np.nansum(Enew.firefilter,axis=0).astype(np.float)
    anthsum = np.nansum(Enew.anthrofilter,axis=0).astype(np.float)
    pixsum  = np.nansum(Enew.pixels_u,axis=0).astype(np.float)
    pix  = Enew.pixels_u # unfiltered pixel counts
    pix[Enew.oceanmask3d]=0
    pixa=np.copy(pix)
    pixb=np.copy(pix)
    pixc=np.copy(pix)
    pixa[Enew.firefilter]=0
    pixb[Enew.anthrofilter]=0
    pixc[Enew.firefilter]=0
    pixc[Enew.anthrofilter]=0

    # make smoke mask over same time period
    smoke,sdates,slats,slons = fio.make_smoke_mask(d0,dn,region=pp.__AUSREGION__)
    smokesum = np.nansum(smoke,axis=0).astype(np.float)
    pixd=np.copy(pix)
    pixd[smoke] = 0
    assert np.all(slats == lats)

    plt.figure(figsize=[16,16])
    plt.subplot(2,2,1)
    pp.createmap(smokesum,lats,lons,title='days filtered (smoke)',
                 aus=True,linear=True, set_under='grey',vmin=1)
    plt.subplot(2,2,2)
    pp.createmap(firesum,lats,lons,title='days filtered (fire)',
                 aus=True,linear=True, set_under='grey',vmin=1)
    plt.subplot(2,1,2)
    # time series for pixel counts
    for arr,colour,label in zip([pix,pixa,pixd],['k','m','c'],['unfiltered','fire','smoke']):
        pp.plot_time_series(Enew.dates,np.nansum(arr,axis=(1,2)),color=colour,label=label)
    plt.xlabel('day')
    plt.ylabel('pixel count')
    plt.legend(loc='best')
    timestr='%s-%s'%(d0.strftime('%Y%m%d'),dn.strftime('%Y%m%d'))
    plt.suptitle('Fire and smoke filters applied on %s'%timestr,fontsize=30)
    pname='Figs/Filters/SmokeFilter_%s.png'%timestr
    plt.savefig(pname)
    print('Saved ',pname)

    ####
    ## Also create a plot looking at each of fire/anthro filters and portion of data removed
    for arr,title,pixf in zip([firesum,smokesum],['Pyrogenic','Smoke'],[pixa,pixb]):
        pname='Figs/Filters/%s_%s.png'%(title,timestr)
        plt.figure(figsize=[10,16])
        plt.subplot(2,1,1)
        pp.createmap(100*arr/pixsum,lats,lons,title=title,
                     linear=True, vmin=1,vmax=100,
                     aus=True, set_under='grey')
        plt.subplot(2,1,2)
        pixot=np.nanmean(100*(1-pixf/pix),axis=(1,2))
        pp.plot_time_series(Enew.dates, pixot,title='Portion removed')
        plt.xlabel('time')
        plt.ylabel('%')
        plt.suptitle('%s filter: %s'%(title,timestr),fontsize=30)

        plt.savefig(pname)
        print('Saved ',pname)

def show_mask_filtering(d0=datetime(2005,1,1), dn=datetime(2006,1,1)):
    '''
        masked squares count for fire, masked squares count for anthro
            time series of pixel counts with and without filtering
    '''

    Enew = E_new(d0,dn,dkeys=['pixels_u','pixels','firefilter','anthrofilter',])
    lats = Enew.lats;  lons=Enew.lons
    firesum = np.nansum(Enew.firefilter,axis=0).astype(np.float)
    anthsum = np.nansum(Enew.anthrofilter,axis=0).astype(np.float)
    pixsum  = np.nansum(Enew.pixels_u,axis=0).astype(np.float)
    pix  = Enew.pixels_u # unfiltered pixel counts
    pix[Enew.oceanmask3d]=0
    pixa=np.copy(pix)
    pixb=np.copy(pix)
    pixc=np.copy(pix)
    pixa[Enew.firefilter]=0
    pixb[Enew.anthrofilter]=0
    pixc[Enew.firefilter]=0
    pixc[Enew.anthrofilter]=0


    plt.figure(figsize=[16,16])
    plt.subplot(2,2,1)
    pp.createmap(firesum,lats,lons,title='days filtered (fire)',
                 aus=True,linear=True, set_under='grey',vmin=1)
    plt.subplot(2,2,2)
    pp.createmap(anthsum,lats,lons,title='days filtered (anth)',
                 aus=True,linear=True, set_under='grey',vmin=1)
    plt.subplot(2,1,2)
    # time series for pixel counts
    for arr,colour,label in zip([pix,pixa,pixb,pixc],['k','m','c','r'],['unfiltered','fire','anth','both']):
        pp.plot_time_series(Enew.dates,np.nansum(arr,axis=(1,2)),color=colour,label=label)
    plt.xlabel('day')
    plt.ylabel('pixel count')
    plt.legend(loc='best')
    timestr='%s-%s'%(d0.strftime('%Y%m%d'),dn.strftime('%Y%m%d'))
    plt.suptitle('Anthro and Fire filters applied on %s'%timestr,fontsize=30)
    pname='Figs/Filters/PixelsFiltered_%s.png'%timestr
    plt.savefig(pname)
    print('Saved ',pname)

    ####
    ## Also create a plot looking at each of fire/anthro filters and portion of data removed
    for arr,title,pixf in zip([firesum,anthsum],['Pyrogenic','Anthropogenic'],[pixa,pixb]):
        pname='Figs/Filters/%s_%s.png'%(title,timestr)
        plt.figure(figsize=[10,16])
        plt.subplot(2,1,1)
        pp.createmap(100*arr/pixsum,lats,lons,title=title,
                     linear=True, vmin=1,vmax=100,
                     aus=True, set_under='grey')
        plt.subplot(2,1,2)
        pixot=np.nanmean(100*(1-pixf/pix),axis=(1,2))
        pp.plot_time_series(Enew.dates, pixot,title='Portion removed')
        plt.xlabel('time')
        plt.ylabel('%')
        plt.suptitle('%s filter: %s'%(title,timestr),fontsize=30)

        plt.savefig(pname)
        print('Saved ',pname)




def HCHO_vs_temp_locational(d0=datetime(2005,1,1),d1=datetime(2005,2,28),
                            locations=None,suffix=''):
    '''
    Look at Modelled Temperature vs satellite HCHO, low tmp with high HCHO could suggest fire
    Hopefully fire filter removes some of these points and improves regression

    Plot comparison of temperature over region over time
        Regression in subset of region, with and without fire filtered

    '''
    if locations is None:
        locations = {'Syd':[-33.87,151.21], # Sydney
                     'Can':[-35.28,149.13], # Canberra
                     'W1': [-33.87+2,151.21-2.5], # Sydney left and up one square
                     'W2': [-33.87,151.21-2.5], # Sydney left one square
                     'W3': [-33.87-2,151.21-2.5], # Sydney left and down one square
                     }
    nlocs=len(locations)

    # read fire filter, VCC_GC, and VCC_GC_lr
    Enew=E_new(d0,d1,dkeys=['firefilter','VCC_GC','pixels_u','pixels'])
    ntimes=len(Enew.dates)
    # read modelled hcho, and temperature at midday
    gc=GC_sat(d0,d1,keys=['IJ-AVG-$_CH2O','DAO-FLDS_TS'],run='tropchem')


    ymd=d0.strftime('%Y%m%d') + '-' + d1.strftime('%Y%m%d')
    pname='Figs/Filters/HCHO_vs_temp_%s%s.png'%(ymd,suffix)

    # fire filter to lower resolution
    firefilter = Enew.firefilter

    VCC_GC_u = np.copy(Enew.VCC_GC)
    VCC_GC   = np.copy(Enew.VCC_GC)
    VCC_GC[firefilter] = np.NaN

    firefilter_lr = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)],dtype=np.bool)
    VCC_GC_lr_u = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)])
    VCC_GC_lr   = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)])
    for ti in range(ntimes):
        firefilter_lr[ti,:,:] = util.regrid(Enew.firefilter[ti,:,:],
                                            Enew.lats,Enew.lons,
                                            Enew.lats_lr,Enew.lons_lr,
                                            groupfunc=np.sum) > 0
        VCC_GC_lr_u[ti,:,:]   = util.regrid_to_lower(VCC_GC_u[ti,:,:],
                                                     Enew.lats,Enew.lons,
                                                     Enew.lats_lr,Enew.lons_lr,
                                                     pixels=Enew.pixels_u[ti,:,:])
        VCC_GC_lr[ti,:,:]     = util.regrid_to_lower(VCC_GC[ti,:,:],
                                                     Enew.lats,Enew.lons,
                                                     Enew.lats_lr,Enew.lons_lr,
                                                     pixels=Enew.pixels[ti,:,:])
    # Temperature avg:
    surftemps=gc.surftemp[:,:,:,0] # surface temp in Kelvin

    # Figure will be small map? and corellations in following subplots
    fig, ax = plt.subplots(nlocs, 2, figsize=(20,18), sharex='all', sharey='all')

    for i, (name,[lat,lon]) in enumerate(locations.items()):

        # Grab HCHO and temperature at lat,lon
        #
        elati,eloni = util.lat_lon_index(lat,lon,Enew.lats_lr,Enew.lons_lr)
        hcho_u = VCC_GC_lr_u[:,elati,eloni]
        hcho = VCC_GC_lr[:,elati,eloni]
        # Mark which parts will be affected by fire filter
        fires= np.array(firefilter_lr[:,elati,eloni],dtype=np.bool)
        glati,gloni = util.lat_lon_index(lat,lon,gc.lats,gc.lons)
        temp = surftemps[:,glati,gloni]

        # scatter between hcho and temp
        #
        plt.sca(ax[i,0])
        plt.scatter(temp, hcho_u, color='k')
        # add x over fire filtered ones
        plt.scatter(temp[fires],hcho_u[fires], marker='x', color='r')
        # add regression (exponential)
        pp.add_regression(temp, hcho_u, addlabel=True,
                          exponential=True, color='k', linewidth=1)
        plt.legend(loc='best')
        # add regression after filter?
        #pp.add_regression(temp[~fullmask], hcho[~fullmask], addlabel=True,
        #                  exponential=True, color='r', linewidth=1)
        plt.title('%s (%.1f,%.1f)'%(name,lat,lon))
        if i==nlocs//2:
            plt.ylabel('VCC$_{GC}$ [molec/cm2]')

        # scatter between hcho and temp with fires removed
        #
        plt.sca(ax[i,1])
        plt.scatter(temp, hcho, color='k')
        # add x over fire filtered ones
        plt.scatter(temp[fires],hcho[fires], marker='x', color='teal')
        # add regression (exponential)
        pp.add_regression(temp, hcho, addlabel=True,
                          exponential=True, color='k', linewidth=1)
        plt.legend(loc='best')
        # add regression after filter?
        #pp.add_regression(temp[~fullmask], hcho[~fullmask], addlabel=True,
        #                  exponential=True, color='r', linewidth=1)
        plt.title('%s (fires masked)'%(name))

    plt.suptitle('Scatter and regressions %s'%ymd,fontsize=36)
    for ax in [ax[i,0],ax[i,1]]:
        plt.sca(ax)
        plt.xlabel('Kelvin')
    #plt.legend(loc='best')
    # set fontsizes for plot
    fs=10
    for attr in ['ytick','xtick','axes']:
        plt.rc(attr, labelsize=fs)
    plt.rc('font',size=fs)

    #cax, _ = matplotlib.colorbar.make_axes(ax)
    #matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)

    # Extra figure using CPC temperatures
    pname='Figs/Filters/HCHO_vs_CPCtemp_%s%s.png'%(ymd,suffix)

    # Read the CPC temperatures
    tmax, _d, lats_cpc, lons_cpc = fio.read_CPC_temp(d0,d1,regrid=True)
    tmax=tmax+273.15 # to kelvin

    # Figure will be small map? and corellations in following subplots
    fig, ax = plt.subplots(nlocs, 2, figsize=(20,18), sharex='all', sharey='all')

    # same as before but not using lowres
    for i, (name,[lat,lon]) in enumerate(locations.items()):

        # Grab HCHO and temperature at lat,lon
        #
        elati,eloni = util.lat_lon_index(lat,lon,Enew.lats,Enew.lons)
        hcho_u = VCC_GC_u[:,elati,eloni]
        hcho = VCC_GC[:,elati,eloni]
        # Mark which parts will be affected by fire filter
        fires= np.array(firefilter[:,elati,eloni],dtype=np.bool)
        clati,cloni = util.lat_lon_index(lat,lon,lats_cpc,lons_cpc)
        temp = tmax[:,clati,cloni]

        # scatter between hcho and temp
        #
        plt.sca(ax[i,0])
        plt.scatter(temp, hcho_u, color='k')
        # add x over fire filtered ones
        plt.scatter(temp[fires],hcho_u[fires], marker='x', color='r')
        # add regression (exponential)
        pp.add_regression(temp, hcho_u, addlabel=True,
                          exponential=True, color='k', linewidth=1)
        plt.legend(loc='best')
        # add regression after filter?
        #pp.add_regression(temp[~fullmask], hcho[~fullmask], addlabel=True,
        #                  exponential=True, color='r', linewidth=1)
        plt.title('%s (%.1f,%.1f)'%(name,lat,lon))
        if i==nlocs//2:
            plt.ylabel('VCC$_{GC}$ [molec/cm2]')

        # scatter between hcho and temp with fires removed
        #
        plt.sca(ax[i,1])
        plt.scatter(temp, hcho, color='k')
        # add x over fire filtered ones
        plt.scatter(temp[fires],hcho[fires], marker='x', color='teal')
        # add regression (exponential)
        pp.add_regression(temp, hcho, addlabel=True,
                          exponential=True, color='k', linewidth=1)
        plt.legend(loc='best')
        # add regression after filter?
        #pp.add_regression(temp[~fullmask], hcho[~fullmask], addlabel=True,
        #                  exponential=True, color='r', linewidth=1)
        plt.title('%s (fires masked)'%(name))

    plt.suptitle('Scatter and regressions (CPC temperature) %s'%ymd,fontsize=36)
    for ax in [ax[i,0],ax[i,1]]:
        plt.sca(ax)
        plt.xlabel('Kelvin')
    #plt.legend(loc='best')
    # set fontsizes for plot
    fs=10
    for attr in ['ytick','xtick','axes']:
        plt.rc(attr, labelsize=fs)
    plt.rc('font',size=fs)

    #cax, _ = matplotlib.colorbar.make_axes(ax)
    #matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)




def HCHO_vs_temp_vs_fire(d0=datetime(2005,1,1),d1=datetime(2005,1,31), subset=2,
                         detrend=False,
                         regionplus = pp.__AUSREGION__):
    '''
    Look at Modelled Temperature vs satellite HCHO, low tmp with high HCHO could suggest fire
    Hopefully fire filter removes some of these points and improves regression

    Plot comparison of temperature over region over time
        Regression in subset of region, with and without fire filtered

    '''
    # subregions:
    region=pp.__AUSREGION__
    NA     = util.NA
    SWA    = util.SWA
    SEA    = util.SEA
    subs   = [SWA,NA,SEA]
    labels = ['SWA','NA','SEA']
    colours = ['chartreuse','magenta','aqua']
    if subset is not None:
        regionlabel=labels[subset]
        region = subs[subset]

    # region + view area
    if regionplus is None:
        regionplus=np.array(region)+np.array([-10,-15,10,15])

    if d1 is None:
        d1=util.last_day(d0)

    # ymd string and plot name
    ymd=d0.strftime('%Y%m%d') + '-' + d1.strftime('%Y%m%d')
    pname='Figs/Filters/Fire_vs_HCHO_vs_temp_%s_%s.png'%(regionlabel,ymd)

    # read modelled hcho, and temperature at midday
    gc=GC_sat(d0,d1,keys=['IJ-AVG-$_CH2O','DAO-FLDS_TS'],run='tropchem')
    n_times=gc.ntimes

    # read satellite HCHO (VCC original) and fire counts, fire mask, and smoke mask
    omi=omhchorp(d0,d1,keylist=['VCC_OMI','fires','firemask','smokemask','gridentries'])

    # Regrid omi columns to lower GC resolution
    omivcc=np.zeros([n_times,gc.nlats,gc.nlons])
    omifires=np.zeros([n_times,gc.nlats,gc.nlons])
    omifiremask=np.zeros([n_times,gc.nlats,gc.nlons])
    omismokemask=np.zeros([n_times,gc.nlats,gc.nlons])
    for ti in range(n_times):
        omivcc[ti,:,:]=util.regrid(omi.VCC_OMI[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omifires[ti,:,:]=util.regrid(omi.fires[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omifiremask[ti,:,:]=util.regrid(omi.firemask[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omismokemask[ti,:,:]=util.regrid(omi.smokemask[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)

    # Temperature avg:
    temp=gc.surftemp[:,:,:,0] # surface temp in Kelvin
    surfmeantemp=np.mean(temp,axis=0)

    # Figure will be map and corellation in subregion, two rows
    plt.figure(figsize=[16,12])

    # First plot temp map of region
    #
    lati,loni=util.lat_lon_range(gc.lats,gc.lons,regionplus)
    smt=surfmeantemp[lati,:]
    smt=smt[:,loni]
    tmin,tmax=np.min(smt),np.max(smt)
    hmin,hmax=1e15, 2e16
    ax0=plt.subplot(221)
    m, cs, cb=pp.createmap(surfmeantemp, gc.lats, gc.lons,
                         region=regionplus, cmapname='rainbow',
                         cbarorient='bottom',
                         vmin=tmin,vmax=tmax,
                         GC_shift=True, linear=True,
                         title='Temperature '+ymd, clabel='Kelvin')

    # Add rectangle around where we are correlating
    pp.add_rectangle(m,region,linewidth=2)

    # Also plot average HCHO
    plt.subplot(222)
    pp.createmap(np.nanmean(omi.VCC_OMI,axis=0),omi.lats,omi.lons,
                 region=regionplus, cmapname='rainbow',
                 cbarorient='bottom',
                 vmin=hmin,vmax=hmax,
                 GC_shift=True, linear=False,
                 title='VCC$_{OMI}$', clabel='molec/cm2')

    # Get fire and hcho subsetted to region
    subsets=util.lat_lon_subset(gc.lats,gc.lons,region,data=[temp,omivcc,omifires,omifiremask,omismokemask],has_time_dim=True)
    lati,loni=subsets['lati'],subsets['loni']
    lats,lons=subsets['lats'],subsets['lons']
    temp,hcho,fires,firemask,smokemask = subsets['data']

    # remove mean (detrend spatially before correlation)
    if detrend:
        for arr in [temp,hcho,fires,firemask]:
            # Average over time dim, then repeat over time dim
            arrmean=np.repeat(np.nanmean(arr,axis=0)[np.newaxis,:,:],n_times,axis=0)
            arr[:,:,:] = arr-arrmean

    # Get ocean and fire/smoke masks
    oceanmask=util.oceanmask(lats,lons)
    oceanmask= np.repeat(oceanmask[np.newaxis, :, :], n_times, axis=0) # repeat along time dimension
    fullmask=oceanmask + (firemask>0) + (smokemask>0)

    # Colour the scatter plot by fire count
    norm = plt.Normalize(vmin=0., vmax=1.)
    cmap = plt.cm.get_cmap('rainbow')
    # colour by fire mask
    colors = [ cmap(norm([0,1][value])) for value in fullmask[~oceanmask] ]

    # Next will be scatter between hcho and temp, coloured by fire counts
    #
    ax=plt.subplot(223)
    plt.scatter(temp[~oceanmask],hcho[~oceanmask], color=colors,)
    plt.title('Scatter (coloured by firecounts)')
    plt.ylabel('VCC$_{OMI}$ [molec/cm2]')
    plt.xlabel('Kelvin')
    pp.add_regression(temp[~oceanmask], hcho[~oceanmask], addlabel=True,
                      exponential=True, color='k', linewidth=1)
    pp.add_regression(temp[~fullmask], hcho[~fullmask], addlabel=True,
                      exponential=True, color='r', linewidth=1)
    plt.legend(loc='best')
    # set fontsizes for plot
    fs=10
    for attr in ['ytick','xtick','axes']:
        plt.rc(attr, labelsize=fs)
    plt.rc('font',size=fs)

    #cax, _ = matplotlib.colorbar.make_axes(ax)
    #matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    # Final plot with regression on sample of gridsquares
    landsample=list(np.argwhere(~oceanmask[0])) # list of indices lat,lon of land squares
    jj=0
    NP=3
    for i,j in random.sample(landsample,NP):
        jj=jj+1
        lat=lats[i]; lon=lons[j]
        X,Y = temp[:,i,j],hcho[:,i,j]
        mx,my = m(lons[j], lats[i])

        # Add dot to map
        plt.sca(ax0)
        m.plot(mx, my, 'd', markersize=8, color='white')

        # Colour scatter by whether it's masked or not
        M=fullmask[:,i,j]
        if len(X[~M]) < 1:
            print('all of ',lat,lon,'masked')
            continue
        #colors = [ cmap(norm([0,1][value])) for value in M ]
        colors = [ ['k','m'][value] for value in M] # purple if masked

        # Add scatter and regressions
        plt.subplot(2,NP,NP+jj)
        plt.scatter(X,Y, color=colors)
        pp.add_regression(X, Y, addlabel=True,
                          exponential=True, color='k',linewidth=1)
        pp.add_regression(X[~M], Y[~M], addlabel=True,
                          exponential=True, color='r',linewidth=1)

        plt.title('lat,lon=%.1f,%.1f'%(lat,lon), fontsize=10)
        plt.ylabel('VCC$_{OMI}$ [molec/cm2]')
        plt.xlabel('Kelvin')
        plt.legend(loc='best',fontsize=8)
    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)

def check_filters_work_with_missing_day():
    '''
    '''
    mday=datetime(2005,2,7) # missing day from satellite data sets
    priorday=datetime(2005,2,6)

    fire,d,lat,lon = fio.make_fire_mask(mday, prior_days_masked=2)
    fire2,d,lat,lon = fio.make_fire_mask(priorday, prior_days_masked=1)

    assert np.all(fire==fire2), "fire mask should not be changed by missing day"


    no2,d,lat,lon=fio.make_anthro_mask(mday) #Handles missing day OK
    yaa,lat,lon=fio.yearly_anthro_avg(mday)
    avgmask=yaa>fio.__Thresh_NO2_y__
    assert np.sum(no2)==np.sum(avgmask), "should be no extra days filtered for anthro on missing day..."

    smoke,d,lat,lon=fio.make_smoke_mask(mday)
    assert np.sum(smoke) == 0, "should be no days filtered for smoke on missing day..."

def test_mask_effects(d0=datetime(2006,1,1),dn=datetime(2006,1,31),
                      outputs=['VCC_OMI','VCC_GC','VCC_PP']):
    '''
        Show and count how many pixels are removed by each filter per day/month
        Show affect on HCHO columns over Aus from each filter per month

        Four plots, one for each mask and one with all masks
            output, output masked
            Entries, Entries
            table of differences for AUS land squares

        Can decide output from VCCs or E_VCCs (remember to use _u for unfiltered ones)

    '''
    #output = 'VCC_OMI' # This is the array we will test our filters on...
    for output in outputs:

        #d0=datetime(month.year,month.month,1)
        #dn=util.last_day(d0)
        dstr="%s-%s"%(d0.strftime("%Y%m%d"),dn.strftime("%Y%m%d"))
        suptitle="%s with and without filtering over %s"%(output,dstr)
        pnames="Figs/Filters/filtered_%%s_%%s_%s.png"%dstr
        titles="%s"
        masktitles="masked by %s"

        # Read E_new to get both masks and HCHO (and E_new if desired)
        Enew=E_new(d0,dn)
        dates=Enew.dates
        lats,lons=Enew.lats,Enew.lons

        masks=[mask.astype(np.bool) for mask in [Enew.firefilter, Enew.anthrofilter]]
        mask_names=['fire','anthro']

        vmin = 1e14; vmax=4e16
        arr = getattr(Enew,output)
        units= Enew.attributes[output]['units']
        if units is None:
            units='???'
        print(units)
        for mask,mask_name in zip(masks,mask_names):
            title=titles%output
            pname=pnames%(output,mask_name)
            masktitle=masktitles%mask_name
            masked = np.copy(arr)
            masked[mask]=np.NaN

            pp.subzones(arr,dates,lats,lons, comparison=masked,
                        vmin=vmin,vmax=vmax,
                        title=title,suptitle=suptitle,comparisontitle=masktitle,
                        clabel=units)
            #TODO: Add axes and table of lost pixel counts...


            plt.savefig(pname)
            print('saved ',pname)
            plt.close()

def test_fires_removed(day=datetime(2005,1,25)):
    '''
    Check that fire affected pixels are actually removed
    '''
    # Read one or 8 day average:
    #
    omrp= omhchorp(day)
    pre     = omrp.VCC
    count   = omrp.gridentries
    lats,lons=omrp.latitude,omrp.longitude
    pre_n   = np.nansum(count)
    ymdstr=day.strftime(" %Y%m%d")
    # apply fire masks
    #
    fire8           = omrp.fire_mask_8 == 1
    fire16          = omrp.fire_mask_16 == 1
    post8           = dcopy(pre)
    post8[fire8]    = np.NaN
    post16          = dcopy(pre)
    post16[fire16]  = np.NaN
    post8_n         = np.nansum(count[~fire8])
    post16_n        = np.nansum(count[~fire16])

    # print the sums
    print("%1e entries, %1e after 8day fire removal, %1e after 16 day fire removal"%(pre_n,post8_n,post16_n))

    # compare and beware
    #
    f = plt.figure(num=0,figsize=(16,6))
    # Plot pre, post8, post16
    # Fires Counts?

    vmin,vmax=1e14,1e17

    titles= [ 'VCC'+s for s in ['',' - 8 days of fires', ' - 16 days of fires'] ]
    for i,arr in enumerate([pre,post8,post16]):
        plt.subplot(131+i)
        m,cs = pp.ausmap(arr,lats,lons,vmin=vmin,vmax=vmax,colorbar=False)
        plt.title(titles[i])
    pname='Figs/fire_exclusion_%s.png'%['8d','1d'][oneday]
    plt.suptitle("Effects of Fire masks"+ymdstr,fontsize=28)
    plt.tight_layout()
    #plt.subplots_adjust(top=0.92)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def test_filters_consistent():
    Enew=E_new(datetime(2005,1,1),datetime(2005,1,31))
    pix_u=Enew.pixels_u
    pix=np.copy(pix_u)
    pix[Enew.firefilter]=0
    pix[Enew.anthrofilter]=0
    assert np.all(pix == Enew.pixels) , 'filters not consistent!!!'


def no2_map(data, lats, lons, vmin, vmax,
            subzones=__subzones__, colors=__colors__):
    '''
        Plot australia, with subzones and stuff added
    '''

    cmapname='plasma'
    region=pp.__AUSREGION__
    if subzones is not None:
        region=subzones[0]
    bmap,cs,cb = pp.createmap(data, lats, lons, region=region,
                              vmin=vmin, vmax=vmax, clabel='molec/cm2',
                              cmapname=cmapname)
    # Add cities to map
    for city,latlon in cities.items():
        pp.add_point(bmap,latlon[0],latlon[1],markersize=12,
                     color='floralwhite', label=city, fontsize=12,
                     xlabeloffset=-50000,ylabeloffset=30000)

    # Add squares to map:
    if subzones is not None:
        for i,subzone in enumerate(subzones[1:]):
            pp.add_rectangle(bmap,subzone,color=colors[i+1],linewidth=2)

    return bmap,cs,cb

def no2_timeseries(no2_orig,dates,lats,lons,
                   subzones=__subzones__,colors=__colors__,
                   ylims=[2e14,4e15], print_values=False):
    '''
        time series for each subzone in no2_tests
    '''

    doys=[d.timetuple().tm_yday for d in dates]

    # loop over subzones
    for i,subzone in enumerate(subzones):
        # Subset our data to subzone
        no2=np.copy(no2_orig)
        lati,loni=util.lat_lon_range(lats,lons,subzone)
        no2 = no2[:,lati,:]
        no2 = no2[:,:,loni]

        # Mask ocean
        oceanmask=util.get_mask(no2[0],lats[lati],lons[loni],masknan=False,maskocean=True)
        print("Removing %d ocean squares"%(365*np.sum(oceanmask)))
        no2[:,oceanmask] = np.NaN

        # Also remove negatives
        negmask=no2 < 0
        print("Removing %d negative squares"%(np.sum(negmask)))
        no2[negmask]=np.NaN

        # get mean and percentiles of interest for plot
        #std = np.nanstd(no2,axis=(1,2))
        upper = np.nanpercentile(no2,75,axis=(1,2))
        lower = np.nanpercentile(no2,25,axis=(1,2))
        mean = np.nanmean(no2,axis=(1,2))
        totmean = np.nanmean(no2)

        lw=[1,4][i==0] # linewidth

        # plot timeseries
        plt.plot(doys, mean, color=colors[i],linewidth=lw)
        # Add IQR shading for first plot
        if i==0:
            plt.fill_between(doys, lower, upper, color=colors[i],alpha=0.2)

        # show yearly mean
        plt.plot([370,395],[totmean,totmean], color=colors[i],linewidth=lw)

        # change to log y scale?
        plt.ylim(ylims)
        plt.yscale('log')
        plt.ylabel('molec/cm2')
        yticks=[2e14,5e14,1e15,4e15]
        ytickstr=['%.0e'%tick for tick in yticks]
        plt.yticks(yticks,ytickstr)
        plt.xlabel('Day of year')

        # Print some stats if desired
        if print_values:
            with open(__no2_txt_file__,'a') as outf: # append to file
                outf.write("Stats for %d, %s, %s\n"%(i, str(subzone), colors[i]))
                outf.write("  yearly mean:  %.3e\n"%totmean)
                outf.write("          std:  %.3e\n"%np.nanstd(no2))
                outf.write("      entries:  %d\n"%np.sum(~np.isnan(no2)))
                outf.write("  gridsquares:  %d\n\n"%np.prod(np.shape(no2)))
            print("Wrote stats to ",__no2_txt_file__)

def no2_densities(year, no2_orig, lats, lons,
                  threshd=__Thresh_NO2_d__,
                  subzones=__subzones__, colors=__colors__):
    '''
        Look at densities of no2 pixels from omno2d
    '''
    plotname=year.strftime('Figs/OMNO2_densities_%Y.png')
    no2=np.copy(no2_orig)

    # Get mean for whole year
    no2_mean = np.nanmean(no2,axis=0)

    # plot map with regions:
    plt.figure(figsize=[16,14])

    title = 'Mean OMNO2d %d'%year.year
    vmin = 1e14
    vmax = 1e15
    plt.subplot(2,2,1)
    bmap,cs,cb = no2_map(no2_mean,lats,lons,vmin,vmax,subzones,colors)
    plt.title(title)

    # One density plot for each region in subzones
    for i,subzone in enumerate(subzones):
        # Subset our data to subzone
        lati,loni=util.lat_lon_range(lats,lons,subzone)
        no2 = np.copy(no2_orig)
        no2 = no2[:,lati,:]
        no2 = no2[:,:,loni]

        # Mask ocean
        oceanmask=util.get_mask(no2[0],lats[lati],lons[loni],masknan=False,maskocean=True)
        print("Removing %d ocean pixels"%(365*np.sum(oceanmask)))
        no2[:,oceanmask] = np.NaN

        # Also remove negatives?
        negmask=no2 < 0
        print("Removing %d negative pixels"%(np.sum(negmask)))
        no2[negmask]=np.NaN

        # all Australia density map
        bw=5e13 # bin widths
        if i == 0:
            plt.subplot(2, 2, 2)
            pp.density(no2,bw=bw,color=colors[i], linewidth=2)
        else:
            plt.subplot(2,4,i+4)
            pp.density(no2,bw=bw, color=colors[i], linewidth=2) # should work with 3d
        plt.xlim([0,5e15])
        plt.plot([threshd,threshd],[0,1], '--k')

    plt.suptitle("OMNO2d NO2 columns %d"%year.year, fontsize=24)
    plt.savefig(plotname)
    print("saved ",plotname)
    plt.close()

def typical_no2(no2_orig,dates,lats,lons,
                threshy=__Thresh_NO2_y__,
                subzones=__subzones__,colors=__colors__,):
    '''
        Plot of NO2 from OMNO2d product over Australia, including time series
    '''
    year=dates[0]
    region=subzones[0]
    plotname=year.strftime('Figs/OMNO2_timeseries_%Y.png')

    # Tropospheric cloud screened (<30%) no2 columns (molec/cm2)
    no2=np.copy(no2_orig)
    no2_mean, no2_std = np.nanmean(no2,axis=0), np.nanstd(no2,axis=0)

    # plot stuff:
    plt.figure(figsize=[16,16])
    # MEAN | STDev
    titles = ['Mean %d'%year.year, 'Standard deviation %d'%year.year]
    vmins  = [1e14, None]
    vmaxes = [5e15, None]

    axes=[]
    bmaps=[]
    for i,arr in enumerate([no2_mean,no2_std]):
        axes.append(plt.subplot(2,2,i+1))
        vmin,vmax=vmins[i],vmaxes[i]
        bmap,cs,cb = no2_map(arr, lats, lons,vmin,vmax,subzones,colors)
        plt.title(titles[i])

        bmaps.append(bmap) # save the bmap for later

    # Hatch for yearly threshhold
    pp.hatchmap(bmaps[0],no2_mean,lats,lons,threshy,region=region)


    # Bottom row
    axes.append(plt.subplot(2,1,2))

    # For each subset here, plot the timeseries
    no2_timeseries(no2_orig,dates,lats,lons,subzones,colors)

    plt.title('Mean time series (ocean masked) over %d'%year.year)

    plt.suptitle("OMNO2d NO2 columns", fontsize=24)
    plt.savefig(plotname)
    print("saved ",plotname)
    plt.close()

def no2_thresh(no2_orig,dates,lats,lons,
               threshd=__Thresh_NO2_d__, threshy=__Thresh_NO2_y__,
               subzones=__subzones__,colors=__colors__):
    '''
        Look at affect of applying threshhold
        no2_orig should be [t,lats,lons] for a particular year
    '''
    year=dates[0].year
    region=subzones[0]
    pname='Figs/OMNO2_threshaffect_%d.png'%year
    fig, axes = plt.subplots(2,2,figsize=[16,16])


    # Subset to region:
    subset=util.lat_lon_subset(lats,lons,region,data=[np.copy(no2_orig)],has_time_dim=True)
    no2=subset['data'][0]
    lats,lons=subset['lats'],subset['lons']


    # Mask ocean
    oceanmask=util.get_mask(no2[0],lats,lons,masknan=False,maskocean=True)
    #print(no2.shape, lats.shape, lons.shape)
    no2[:,oceanmask] = np.NaN

    # Also remove negatives
    negmask=no2 < 0
    #no2[negmask]=np.NaN

    # Filtered copy:
    no2_f=np.copy(no2)
    no2_mean=np.nanmean(no2,axis=0)

    # Filter for yearly threshhold, first count how many we will remove
    n_filtered_y = 0
    n_days=len(no2_f[:,0,0])
    n_filtered_y = n_days*np.nansum(no2_mean>threshy)
    # if day is nan in square removed by threshy then don't count it as removed
    n_filtered_y = n_filtered_y - np.sum(np.isnan(no2_f[:,no2_mean>threshy]))
    # apply filter
    no2_f[:,no2_mean>threshy]=np.NaN

    # Write filtering stats to file
    with open(__no2_txt_file__,'a') as outf: # append to file
        ngoods=np.nansum(no2>0)
        n_filtered_d=np.nansum(no2_f>threshd)
        outf.write("negative gridsquaredays (made NaN)      : %d \n"%np.sum(negmask))
        outf.write("non-NaN, non-ocean gridsquaredays       : %d \n"%ngoods)
        outf.write("Year threshhold (%.1e) removes       : %d (%.2f%%) \n"%
          (threshy, n_filtered_y, n_filtered_y*100/float(ngoods)))
        outf.write("Then day threshhold (%.1e) removes   : %d (%.2f%%) \n"%
          (threshd, n_filtered_d, n_filtered_d*100/float(ngoods)))

    # Apply daily filter
    no2_f[no2_f>threshd] = np.NaN

    # plot stuff:
    titles = ['Mean %d'%year, 'Threshholds applied']
    vmin = 1e14
    vmax = 2e15
    for i,arr in enumerate([no2,no2_f]):

        # Plot map with and without threshhold filter
        mean=np.nanmean(arr,axis=0)
        plt.sca(axes[0,i])
        # only show subzones in first plot
        bmap,_cs,_cb = no2_map(mean, lats, lons, vmin, vmax, [subzones,None][i==1], colors)
        plt.title(titles[i])

        # Hatch for yearly threshhold
        pp.hatchmap(bmap, mean, lats, lons, threshy, region=region)

        # Also time series with and without filter
        plt.sca(axes[1,i])
        no2_timeseries(arr,dates,lats,lons,subzones,colors,print_values=True)

    # Add threshholds to last timeseries
    plt.plot([0,395],[threshd,threshd], '--k',linewidth=1)
    plt.plot([0,395],[threshy,threshy], ':k',linewidth=1)

    plt.suptitle("OMNO2d threshhold affects %d"%year, fontsize=24)
    plt.savefig(pname)
    print('saved ',pname)
    plt.close()

def typical_aaod_month(month=datetime(2005,11,1)):
    ''' '''
    ymstr=month.strftime("%Y%m")
    pname2='Figs/AAOD_month_%s.png'%ymstr
    region=__AUSREGION__
    #vmin=1e-3
    #vmax=1e-1
    vmin,vmax=1e-7,5e-2
    cmapname='pink_r'

    # also show a month of aaod during nov 2005 ( high transport month )
    plt.figure()
    plt.subplot(211)

    # read the aaod and average over the month
    aaod,dates,lats,lons=fio.read_smoke(month,util.last_day(month))
    aaod=np.nanmean(aaod,axis=0)

    # create map
    pp.createmap(aaod,lats,lons,region=region,cmapname=cmapname,
                 vmin=vmin,vmax=vmax,set_bad='blue')

    # also show density map
    plt.subplot(212)
    pp.density(aaod,lats,lons,region=region)

    plt.savefig(pname2)
    print("Saved ",pname2)
    plt.close()

def typical_aaods():
    '''
    Check typical aaod over Australia during specific events
    row a) normal
    row b) Black saturday: 20090207-20090314
    row c) transported smoke: 20051103,08,17
    row d) dust storm : 20090922-24
    '''

    # read particular days of aaod
    dates = [ datetime(2007,8,30), datetime(2009,2,19),
              datetime(2005,11,8), datetime(2009,9,23) ]

    # plot stuff
    plt.figure(figsize=(16,16))
    pname='Figs/typical_AAODs.png'
    region=__AUSREGION__
    vmin=1e-4
    vmax=1e-1
    cmapname='pink_r'
    titles=['normal','black saturday','transported plume','dust storm']
    zooms=[None,[-40,140,-25,153],[-42,130,-20,155],[-40,135,-20,162]]
    TerraModis=['Figs/TerraModis_Clear_20070830.png',
                'Figs/TerraModis_BlackSaturday_20090219.png',
                'Figs/TerraModis_TransportedSmoke_20050811.png',
                'Figs/TerraModis_DustStorm_20090923.png']
    linear=False
    thresh=__Thresh_AAOD__

    for i,day in enumerate(dates):
        zoom=region
        plt.subplot(4,4,1+i*4)
        ymd=day.strftime('%Y %b %d')
        title = titles[i] +' '+ ymd
        aaod, lats, lons = fio.read_AAOD(day)
        m, cs, cb = pp.createmap(aaod, lats, lons, title=title, region=zoom,
                                 vmin=vmin, vmax=vmax, linear=linear,
                                 cmapname=cmapname, set_bad='blue')

        # Add hatch over threshhold values (if they exists)
        #(m, data, lats, lons, thresh, region=None):
        #pp.hatchmap(m,aaod,lats,lons,thresh,region=zoom, hatch='x',color='blue')

        if zooms[i] is not None:
            zoom=zooms[i]
            plt.subplot(4,4,2+i*4)
            m,cs,cb= pp.createmap(aaod, lats, lons ,region=zoom,
                                  vmin=vmin, vmax=vmax, linear=linear,
                                  cmapname=cmapname)
            # Add hatch to minimap also
            #pp.hatchmap(m,aaod,lats,lons,thresh,region=zoom, hatch='x',color='blue')

        plt.subplot(4,4,3+i*4)
        aaod, lats, lons = pp.density(aaod,lats,lons,region=zoom, vertical=True)
        plt.plot([0,50],[thresh,thresh]) # add line for thresh
        plt.title('density')
        plt.ylabel('AAOD')
        plt.gca().yaxis.set_label_position("right")
        #plt.xlim([-0.02,0.1])
        print('Mean AAOD=%.3f'%np.nanmean(aaod))
        print("using %d gridsquares"%np.sum(~np.isnan(aaod)))

        if TerraModis[i] is not None:
            plt.subplot(4,4,4+i*4)
            pp.plot_img(TerraModis[i])
            plt.title(ymd)

    plt.tight_layout()

    plt.savefig(pname)
    plt.close()
    print("Saved ",pname)

def smoke_vs_fire(d0=datetime(2005,1,1),dN=datetime(2005,1,31),region=__AUSREGION__):
    '''
        Compare fire counts to smoke aaod in the omhchorp files
    '''
    d0str=d0.strftime('%Y%m%d')
    if dN is None:
        dN = d0

    dNstr=dN.strftime('%Y%m%d')
    #n_times=(dN-d0).days + 1

    # Read the products from modis and omi, interpolated to 1x1 degrees
    fires, _dates, _modlats, _modlons = fio.read_fires(d0, dN, latres=1,lonres=1)
    aaod, _dates, _omilats, _omilons = fio.read_smoke(d0,dN,latres=1,lonres=1)

    assert all(_modlats==_omilats), 'interpolation is not working'
    lats=_modlats
    lons=_modlons
    # data fires and aaod = [times, lats, lons]
    # data.fires.shape; data.AAOD.shape

    f,axes=plt.subplots(2,2,figsize=(16,16)) # 2 rows 2 columns

    titles=['Fires','AAOD$_{500nm}$']
    linear=[True,False]
    fires=fires.astype(np.float)
    fires[fires<0] = np.NaN
    aaod[aaod<0]   = np.NaN # should do nothing

    # Average over time
    fires=np.nanmean(fires,axis=0)
    aaod=np.nanmean(aaod,axis=0)

    for i,arr in enumerate([fires,aaod]):

        # plot into right axis
        plt.sca(axes[0,i])
        pp.createmap(arr,lats,lons,title=titles[i],
                     linear=linear[i],region=region,
                     colorbar=True,cmapname='Reds',)
                     #vmin=vmins[i],vmax=vmaxes[i])
    plt.suptitle('Fires vs AAOD %s-%s'%(d0str,dNstr))

    # third subplot: regression
    plt.sca(axes[1,0])
    X=fires
    Y=aaod

    subset=util.lat_lon_subset(lats,lons,region,[X,Y])
    X,Y=subset['data'][0],subset['data'][1]
    lats,lons=subset['lats'],subset['lons']
    pp.plot_regression(X,Y,logscale=False,legend=False)
    plt.xlabel('Fires')
    plt.ylabel("AAOD")
    plt.title('Correlation')

    # Fourth plot: density of AAOD,Fires:
    plt.subplot(426)
    #plt.sca(axes[1,1])

    seaborn.set_style('whitegrid')
    seaborn.kdeplot(Y.flatten())# linestyle='-')
    plt.title('aaod density')
    plt.subplot(428)
    seaborn.set_style('whitegrid')
    seaborn.kdeplot(X.flatten())# linestyle='-')
    plt.title('fires density')

    pname='Figs/Smoke_vs_Fire_%s-%s.png'%(d0str,dNstr)
    plt.savefig(pname)
    print("Saved figure ",pname)

def smearing_definition(year=datetime(2005,1,1), old=False, threshmask=False):
    '''
        test smearing creation process.
    '''
    summer= util.list_months(datetime(year.year,1,1), util.last_day(datetime(year.year,2,1)))
    if year.year > 2005:
        summer = util.list_months(datetime(year.year-1,12,1), util.last_day(datetime(year.year,2,1)))
    winter= util.list_months(datetime(year.year,6,1), util.last_day(datetime(year.year,8,1)))

    summer_smear_midday, days,lats,lons=smearing(summer[0],midday=True)
    summer_smear_dayavg, days,lats,lons=smearing(summer[0],midday=False)
    winter_smear_midday, days,lats,lons=smearing(winter[0],midday=True)
    winter_smear_dayavg, days,lats,lons=smearing(winter[0],midday=True)

    for month in summer[1:]:
        # dayavg smearing
        dayavg, days,lats,lons = smearing(month,midday=False)
        summer_smear_dayavg = np.append(summer_smear_dayavg,np.array(dayavg),axis=0)
        # midday smearing
        midday, days,lats,lons = smearing(month,midday=True)
        summer_smear_midday = np.append(summer_smear_midday,np.array(midday),axis=0)
        # old version
        #todo
    for month in winter[1:]:
        # dayavg smearing
        dayavg, days,lats,lons = smearing(month,midday=False)
        winter_smear_dayavg = np.append(winter_smear_dayavg,np.array(dayavg),axis=0)
        # midday smearing
        midday, days,lats,lons = smearing(month,midday=True)
        winter_smear_midday = np.append(winter_smear_midday,np.array(midday),axis=0)
        # old version
        #todo

    smear_units='molec$_{HCHO}$*s/atom$_C$'
    # fix infinity problem (basemap plots may be shitty)
    for arr in [summer_smear_dayavg,summer_smear_midday,winter_smear_dayavg,winter_smear_midday]:
        arr[np.isinf(arr)]=np.NaN

    plt.figure(figsize=(13,13))

    ticks1=np.arange(1000, 10001, 3000)  # custom ticks for daily smearing
    ticks2=np.arange(1000, 5001,  1000)  # custom ticks for midday smearing
    ii=0
    for arr, title, ticks in zip([summer_smear_dayavg,summer_smear_midday,winter_smear_dayavg,winter_smear_midday],
                                     ['dayavg smearing (summer)','midday smearing (summer)','dayavg smearing (winter)','midday smearing (winter)'],
                                     [ticks1,ticks2,ticks1,ticks2]):
        ii=ii+1
        plt.subplot(2,2,ii)
        ## dayavg smearing in summer
        flatarr = np.nanmean(arr,axis=0)
        bmap,cs,cb = pp.createmap(flatarr, lats, lons, aus=True, title=title,
                                  linear=True, vmin=ticks[0], vmax=ticks[-1],
                                  ticks=ticks, clabel=smear_units)

        if threshmask:
            # Add diamond over strict threshold
            # pink where at least one gridday filtered
            maxarr=np.nanmax(arr,axis=0)
            pp.add_marker_to_map(bmap, maxarr>__Thresh_Smearing__,
                                 lats, lons, marker='d',
                                 landonly=False, markersize=6, color='pink')
            # red where always over threshhold
            minarr=np.nanmin(arr,axis=0)
            pp.add_marker_to_map(bmap, minarr>__Thresh_Smearing__,
                                 lats, lons, marker='x',
                                 landonly=False, markersize=8, color='darkred')


    # title and save figure
    plt.suptitle("smearing definition comparisons",fontsize=27)
    pname=year.strftime('Figs/Filters/smearing_definitions_%Y.png')
    plt.savefig(pname)
    print('SAVED ',pname)
    plt.close()

def smearing_threshold(year=datetime(2005,1,1), strict=4000, loose=6000):
    '''
    '''
    # Read smearing from E_new
    d0=datetime(year.year,1,1)
    dn=datetime(year.year,12,31)
    enew=E_new(d0,dn,dkeys=['smearing','smearfilter'])
    smear=enew.smearing
    if smear.shape[1] == len(enew.lats):
        lats=enew.lats
        lons=enew.lons
    else:
        lats=enew.lats_lr
        lons=enew.lons_lr
    mstr=[d.strftime('%b') for d in util.list_months(d0,dn)]


    # Plot map of smearing for each month.
    plt.figure(figsize=(15,15))
    for i in range(12):
        plt.subplot(4,3,i+1)
        smeari=smear[i]
        bmap,cs,cb=pp.createmap(smeari,lats,lons,region=pp.__AUSREGION__,
                                linear=True,vmin=1000,vmax=10000)

        plt.title(mstr[i])

        # Add diamond over strict threshold
        pp.add_marker_to_map(bmap, smeari>strict, lats, lons, marker='d',
                             landonly=False, markersize=10, color='r')

        # Add dot over loose threshold
        pp.add_marker_to_map(bmap, smeari>loose, lats, lons, marker='o',
                             landonly=False, markersize=8, color='k')

    plt.tight_layout()
    plt.suptitle('Smearing by month')
    pname='Figs/Filters/smearing_maps_2005.png'
    plt.savefig(pname)
    plt.close()
    print("SAVED FIGURE ", pname)

    # Plot distribution of smearing for each month...
    plt.figure(figsize=(15,15))
    for i in range(12):
        plt.subplot(4,3,i+1)
        #plt.subplot(1,2,i+1)
        smeari=smear[i]
        smeari[smeari>10000] = np.NaN#10000
        pp.distplot(smeari,lats,lons,region=pp.__AUSREGION__,bins=np.arange(1000,10600,500))
        plt.title(mstr[i])
        plt.xlim([500,10500])
        # format yaxis
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter('{0:.0e}'.format))
        #plt.ylim([0,4e-4])
    plt.tight_layout()
    plt.suptitle('Smearing distribution by month')
    pname='Figs/Filters/smearing_distr_2005.png'
    plt.savefig(pname)
    plt.close()
    print("SAVED FIGURE ", pname)



def smearing_vs_slope(month=datetime(2005,1,1)):
    '''
        compare map of smearing to map of model slope
    '''
    d0=datetime(month.year,1,1)
    dn=util.last_day(month)
    enew=E_new(d0,dn,dkeys=['smearing','smearfilter','ModelSlope'])
    smear=enew.smearing[0]
    slope=enew.ModelSlope[0]
    lats=enew.lats_lr
    lons=enew.lons_lr
    pp.compare_maps([smear,slope],[lats,lats],[lons,lons],
                    vmin=1000, vmax=10000,
                    titles=['Smearing','Slope'],
                    linear=True,
                    pname=d0.strftime("Figs/Filters/smear_vs_slope_%Y%m.png"))


def smearing_regridding(date=datetime(2005,1,1)):
    '''
        S=change in HCHO column / change in E_isop
        What's going on when we interpolate smearing to high resolution??
    '''
    smear,slats,slons=Inversion.smearing(date)
    omilats,omilons, omilate,omilone = util.lat_lon_grid()
    smear2 = util.regrid_to_higher(smear,slats,slons,omilats,omilons,interp='nearest')

    f,axs=plt.subplots(2,2,figsize=(20,22))

    plt.sca(axs[0,0])
    m,cs,cb=pp.createmap(smear, slats, slons, linear=True,
                 vmin=1000, vmax=10000,
                 region=pp.__AUSREGION__,
                 colorbar=False,
                 title='smearing')
    plt.sca(axs[0,1])
    m2,cs,cb=pp.createmap(smear2,omilats,omilons, linear=True,
                 vmin=1000, vmax=10000,
                 region=pp.__AUSREGION__,
                 colorbar=False,
                 title='Smearing interpolated')
    plt.sca(axs[1,0])
    m,cs,cb=pp.createmap(smear, slats, slons, linear=True,
                 vmin=1000, vmax=10000,
                 region=pp.__AUSREGION__,
                 cbarfmt='%.0e',
                 clabel='S',title='smearing')
    pp.add_grid_to_map(m)
    plt.sca(axs[1,1])
    m2,cs,cb=pp.createmap(smear2,omilats,omilons, linear=True,
                 vmin=1000, vmax=10000,
                 region=pp.__AUSREGION__,
                 cbarfmt='%.0e',
                 clabel='S', title='Smearing interpolated')
    pp.add_grid_to_map(m2)

    plt.suptitle('smearing 200501', fontsize=35)
    plt.tight_layout()
    plt.savefig('Figs/Filters/smearing_test.png')
    plt.close()


def check_no2_filter(year):
    '''
    '''
    # First read and whole year of NO2
    dat, attrs = fio.read_omno2d(datetime(year.year,1,1), util.last_day(datetime(year.year,12,1)))
    # Tropospheric cloud screened (<30%) no2 columns (molec/cm2)
    no2_orig=dat['tropno2']
    lats=dat['lats']
    lons=dat['lons']
    dates=dat['dates']

    with open(__no2_txt_file__,'a') as outf: # append to file
        outf.write("Year = %d\n"%year.year)

    # Now run all the NO2 analysis functions
    #
    no2_densities(year, no2_orig, lats, lons)
    typical_no2(no2_orig, dates, lats, lons)
    no2_thresh(no2_orig, dates, lats, lons)
