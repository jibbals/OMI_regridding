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
from classes.GC_class import GC_tavg, GC_sat
from classes.E_new import E_new
from utilities.plotting import __AUSREGION__

# General stuff
import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy

from datetime import datetime#, timedelta

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

def HCHO_vs_temp_vs_fire(d0=datetime(2005,1,1),d1=None, subset=2,
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

    # read satellite HCHO (VCC original) and fire counts, fire mask, and smoke mask
    omi=omhchorp(d0,d1,keylist=['VCC_OMI','fires','firemask','smokemask','gridentries'])

    # Regrid omi columns to lower GC resolution
    omivcc=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    omifires=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    omifiremask=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    omismokemask=np.zeros([gc.ntimes,gc.nlats,gc.nlons])
    for ti in range(omi.n_times):
        omivcc[ti,:,:]=util.regrid(omi.VCC_OMI[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omifires[ti,:,:]=util.regrid(omi.fires[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omifiremask[ti,:,:]=util.regrid(omi.firemask[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)
        omismokemask[ti,:,:]=util.regrid(omi.smokemask[ti,:,:],omi.lats,omi.lons,gc.lats,gc.lons)

    # Temperature avg:
    temp=gc.surftemp[:,:,:,0] # surface temp in Kelvin
    surfmeantemp=np.mean(temp,axis=0)

    # Figure will be map and corellation in subregion, two rows
    plt.figure(figsize=[12,16])

    # First plot temp map of region
    #
    lati,loni=util.lat_lon_range(gc.lats,gc.lons,regionplus)
    smt=surfmeantemp[lati,:]
    smt=smt[:,loni]
    tmin,tmax=np.min(smt),np.max(smt)

    plt.subplot(211)
    m, cs, cb=pp.createmap(surfmeantemp, gc.lats, gc.lons,
                         region=regionplus, cmapname='rainbow',
                         cbarorient='right',
                         vmin=tmin,vmax=tmax,
                         GC_shift=True, linear=True,
                         title='Temperature '+ymd, clabel='Kelvin')
    # Add rectangle around where we are correlating
    pp.add_rectangle(m,region,linewidth=2)


    # Get fire and hcho subsetted to region
    subsets=util.lat_lon_subset(gc.lats,gc.lons,region,data=[temp,omivcc,omifires,omifiremask,omismokemask],has_time_dim=True)
    lati,loni=subsets['lati'],subsets['loni']
    lats,lons=subsets['lats'],subsets['lons']
    temp,hcho,fires,firemask,smokemask = subsets['data']

    # Get ocean and fire/smoke masks
    oceanmask=util.oceanmask(lats,lons)
    oceanmask= np.repeat(oceanmask[np.newaxis, :, :], gc.ntimes, axis=0) # repeat along time dimension
    fullmask=oceanmask + (firemask>0) + (smokemask>0)

    # Colour the scatter plot by fire count
    norm = plt.Normalize(vmin=0., vmax=np.nanmax(fires[~oceanmask]))
    cmap = plt.cm.get_cmap('rainbow')
    colors = [ cmap(norm(value)) for value in fires[~oceanmask] ] # colour by fire count

    # Next will be scatter between hcho and temp, coloured by fire counts
    #
    ax=plt.subplot(212)
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


    cax, _ = matplotlib.colorbar.make_axes(ax)
    matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    #    for ax in ax1,ax2:
    #        # reset lower bounds
    #        ylims=ax.get_ylim()
    #        plt.ylim([max([-2,ylims[0]]), ylims[1]])
    #        xlims=ax.get_xlim()
    #        plt.xlim([max([xlims[0],270]),xlims[1]])

    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)



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

def Filter_affects_on_outputs(d0=datetime(2005,1,1),dn=None,region=pp.__AUSREGION__):
    '''
        Look at affects of filters to both VCC and E_new
    '''
    if dn is None:
        dn=util.last_day(d0)
    dstr="%s-%s"%(d0.strftime("%Y%m%d"),dn.strftime("%Y%m%d"))
    suptitles="%%s with and without filtering over %s"%dstr
    pnames="Figs/Emiss/filtered_%%s_%%s_%s.png"%dstr
    titles="%s"
    masktitles="masked by %s"

    Enew=E_new(d0,dn)
    dates=Enew.dates
    lats,lons=Enew.lats,Enew.lons
    masks=[mask.astype(np.bool) for mask in [Enew.firefilter,Enew.anthrofilter,Enew.firefilter+Enew.anthrofilter]]
    mask_names=['fire','anthro','fire+anthro']
    arr_names=['VCC_GC','VCC_OMI','E_VCC_GC','E_VCC_OMI']
    arr_vmins=[ 1e14   ,  1e14   ,  1e9     ,   1e9     ]
    arr_vmaxs=[ 1e16   ,  1e16   ,  5e12    ,   5e12    ]
    for arr_name,vmin,vmax in zip(arr_names,arr_vmins,arr_vmaxs):
        arr = getattr(Enew,arr_name)
        units= Enew.attributes[arr_name]['unit'] # TODO change to units when implemented
        for mask,mask_name in zip(masks,mask_names):
            title=titles%arr_name
            suptitle=suptitles%arr_name
            pname=pnames%(arr_name,mask_name)
            masktitle=masktitles%mask_name
            pp.subzones(arr,dates,lats,lons, mask=mask,
                        vmin=vmin,vmax=vmax,
                        title=title,suptitle=suptitle,masktitle=masktitle,
                        pname=pname,clabel=units)


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


def smearing_calculation(date=datetime(2005,1,1)):
    '''
        S=change in HCHO column / change in E_isop
    '''
    region=__AUSREGION__
    # READ normal and halfisop run outputs:
    full=GC_tavg(date)
    half=None


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