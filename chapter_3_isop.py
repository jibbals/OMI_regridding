#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:20:05 2019

    All the plots which are placed into chapter 3 will be uniformed here
    
@author: jesse
"""

# plotting libraries
import matplotlib
matplotlib.use('Agg')  # don't show plots interactively
import matplotlib.pyplot as plt
plt.ioff() # plot_date fix

from datetime import datetime, timedelta
import numpy as np



# local modules
from utilities import GMAO,GC_fio,fio, masks
from utilities import utilities as util
from utilities import plotting as pp
from utilities.JesseRegression import RMA, OLS

import Inversion
import tests
from tests import utilities_tests, test_new_emissions
import reprocess
import new_emissions
import Analyse_E_isop

from classes.E_new import E_new # E_new class
from classes import GC_class, campaign
from classes.omhchorp import omhchorp



import xbpch
import xarray
import pandas as pd
import seaborn as sns

import warnings
import timeit


###############
### Globals ###
###############
__VERBOSE__=True

## LABELS
# total column HCHO from GEOS-Chem
__Ogc__ = "$\Omega_{GC}$"
__Ogca__ = "$\Omega_{GC}^{\alpha}$"
__Ogc__units__ = 'molec cm$^{-2}$'


# total column HCHO from OMI (recalculated using PP code)
__Oomi__= "$\Omega_{OMI}$"
__Oomi__units__ = __Ogc__units__
# a priori
__apri__ = "$E_{GC}$"
__apri__units__ = "atom C cm$^{-2}$ s$^{-1}$"
__apri__label__ = r'%s [ %s ]'%(__apri__, __apri__units__)
# a postiori
__apost__ = "$E_{OMI}$"

# Plot size and other things...
def fullpageFigure(*kvpair):
    """set some Matplotlib stuff."""
    matplotlib.rcParams["text.usetex"]      = False     #
    matplotlib.rcParams["legend.numpoints"] = 1         # one point for marker legends
    matplotlib.rcParams["legend.fontsize"]  = 10        # legend font size
    matplotlib.rcParams["figure.figsize"]   = (12, 14)  #
    matplotlib.rcParams["font.size"]        = 18        # font sizes:
    matplotlib.rcParams["axes.titlesize"]   = 18        # title font size
    matplotlib.rcParams["axes.labelsize"]   = 13        #
    matplotlib.rcParams["xtick.labelsize"]  = 13        #
    matplotlib.rcParams["ytick.labelsize"]  = 13        #
    matplotlib.rcParams['image.cmap'] = 'plasma' #'PuRd' #'inferno_r'       # Colormap default
    # set extra key values if wanted
    for k,v in kvpair:
        matplotlib.rcParams[k] = v

labels=util.__subregions_labels__
colors=util.__subregions_colors__
regions=util.__subregions__
n_regions=len(regions)

###################
### SAVE SATELLITE OUTPUT FOR QUICK ANALYSIS
###################
def save_overpass_timeseries():
    # Read and store regional time series into dataframe and he5
    d0=datetime(2005,1,1)
    d1=datetime(2012,12,31)

    regions=util.__subregions__
    labels=util.__subregions_labels__
    
    
    outname = 'Data/GC_Output/overpass_timeseries_regional.csv'
        
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
               'IJ-AVG-$_NO',       # NO in ppb?
               'IJ-AVG-$_O3',       # O3 in ppb
               ] #+ GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys+GC_class.__gc_tropcolumn_keys__, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys+GC_class.__gc_tropcolumn_keys__, run='tropchem')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    
    
    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates
    
    outdict={}
    
    plusdata=['NOx','HCHO_TotCol']
    for origkey in satkeys+plusdata:
        if origkey in satkeys:
            key=GC_class._GC_names_to_nice[origkey]
            # Grab surface array
            new_surf = getattr(new_sat,key)[:,:,:,0] # ppb or ppbC
            trop_surf = getattr(tropchem_sat,key)[:,:,:,0]
        elif origkey == 'NOx':
            key='NOx'
            new_surf = new_sat.NO[:,:,:,0] + new_sat.NO2[:,:,:,0]
            trop_surf = tropchem_sat.NO[:,:,:,0] + tropchem_sat.NO2[:,:,:,0]
        elif origkey == 'HCHO_TotCol':
            key='HCHO_TotCol'
            new_surf = new_sat.get_total_columns()['hcho']
            trop_surf = tropchem_sat.get_total_columns()['hcho']
        
        units = 'ppbv'
        if origkey == 'IJ-AVG-$_ISOP':
            units = 'ppbC'
        
        # Make sure ocean squares are zero'd 
        oceanmask=util.oceanmask(lats,lons)
        oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],len(dates),axis=0)
        new_surf[oceanmask3d] = np.NaN
        trop_surf[oceanmask3d] = np.NaN
        
        # pull out subregions, keeping lats and lons
        new_regional, lats_regional, lons_regional = util.pull_out_subregions(new_surf,lats,lons,subregions=regions)
        trop_regional, lats_regional, lons_regional = util.pull_out_subregions(trop_surf,lats,lons,subregions=regions)
    
        # average spatial dims into time series
        # also store std, Q0, Q1, Q2, Q3, Q4
    
        
        for i,label in enumerate(labels):
            outdict['%s_%s_post_mean'%(key,label)] = np.nanmean(new_regional[i], axis=(1,2))
            outdict['%s_%s_post_std'%(key,label)] = np.nanstd(new_regional[i], axis=(1,2))
            outdict['%s_%s_post_Q0'%(key,label)] = np.nanpercentile(new_regional[i],0, axis=(1,2))
            outdict['%s_%s_post_Q1'%(key,label)] = np.nanpercentile(new_regional[i],25, axis=(1,2))
            outdict['%s_%s_post_Q2'%(key,label)] = np.nanpercentile(new_regional[i],50, axis=(1,2))
            outdict['%s_%s_post_Q3'%(key,label)] = np.nanpercentile(new_regional[i],75, axis=(1,2))
            outdict['%s_%s_post_Q4'%(key,label)] = np.nanpercentile(new_regional[i],100, axis=(1,2))
            
            outdict['%s_%s_pri_mean'%(key,label)] = np.nanmean(trop_regional[i], axis=(1,2))
            outdict['%s_%s_pri_std'%(key,label)] = np.nanstd(trop_regional[i], axis=(1,2))
            outdict['%s_%s_pri_Q0'%(key,label)] = np.nanpercentile(trop_regional[i],0, axis=(1,2))
            outdict['%s_%s_pri_Q1'%(key,label)] = np.nanpercentile(trop_regional[i],25, axis=(1,2))
            outdict['%s_%s_pri_Q2'%(key,label)] = np.nanpercentile(trop_regional[i],50, axis=(1,2))
            outdict['%s_%s_pri_Q3'%(key,label)] = np.nanpercentile(trop_regional[i],75, axis=(1,2))
            outdict['%s_%s_pri_Q4'%(key,label)] = np.nanpercentile(trop_regional[i],100, axis=(1,2))
            print(key, label, ' read into TS lists ')
            
    # Now read the satellite product into a similar format
    dkeys=['E_PP_lr','E_MEGAN', 'VCC_OMI_u','VCC_PP_u','pixels_lr','pixels_PP_lr','pixels_PP_u','pixels_u','pixels','pixels_PP']
    
    enew=E_new(d0,d1,dkeys=dkeys)
    lats=enew.lats_lr
    lons=enew.lons_lr
    
    assert np.all(dates==enew.dates), "Dates don't match"
    
    # Make sure ocean squares are zero'd 
    oceanmask=util.oceanmask(lats,lons)
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],len(dates),axis=0)
    
    for key in dkeys:
        
        # Grab regional subsets
        enewdata=getattr(enew,key).astype(np.float64)
        enewdata[oceanmask3d] = np.NaN
        regional,lats_r,lons_r = util.pull_out_subregions(enewdata,lats,lons,subregions=regions)
        
        
        for i,label in enumerate(labels):
            outdict['%s_%s_mean'%(key,label)] = np.nanmean(regional[i], axis=(1,2))
            outdict['%s_%s_std'%(key,label)] = np.nanstd(regional[i], axis=(1,2))
            if 'pixel' in key:
                outdict['%s_%s_sum'%(key,label)] = np.nansum(regional[i],axis=(1,2))
            else:
                outdict['%s_%s_Q0'%(key,label)] = np.nanpercentile(regional[i],0, axis=(1,2))
                outdict['%s_%s_Q1'%(key,label)] = np.nanpercentile(regional[i],25, axis=(1,2))
                outdict['%s_%s_Q2'%(key,label)] = np.nanpercentile(regional[i],50, axis=(1,2))
                outdict['%s_%s_Q3'%(key,label)] = np.nanpercentile(regional[i],75, axis=(1,2))
                outdict['%s_%s_Q4'%(key,label)] = np.nanpercentile(regional[i],100, axis=(1,2))
                
            print (key,label, ' read into TS lists')
            
    # SAVE THEM WITH DESCRIPTIVE TITLE NAMES INTO DATAFRAME....
    myDF = pd.DataFrame(outdict,index=dates)
    myDF.index.name = 'dates'
    myDF.to_csv(outname)

def read_overpass_timeseries():
    '''
        # EXAMPLE:
        #myDF = chapter_3_isop.read_overpass_timeseries()
        #
        #print(myDF)
        #
        #styles=['bs-','ro-','b^--','r^--']
        #linewidths=[3,3,2,2]
        #fig, ax = plt.subplots()
        #cols=['HCHO_TotCol_Aus_pri_mean','HCHO_TotCol_Aus_post_mean','HCHO_TotCol_Aus_post_Q2','HCHO_TotCol_Aus_pri_Q2']
        #for col, style, lw in zip(cols, styles, linewidths):
        #    myDF[col].plot(style=style, lw=lw, ax=ax)
    '''
    return pd.read_csv('Data/GC_Output/overpass_timeseries_regional.csv', index_col=0)
    


##########
### Plot functions
##########





###
# METHODS
###

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
    bg_ref, bglats_ref,bglons_ref = util.remote_pacific_background(hcho1,lats,lons,average_lons=True) # background with just lats
    bg_ref = np.repeat(bg_ref[:,np.newaxis], len(lons), axis=1) # copy ref background over all the longitudes
    
    # monthly output already for no-isoprene run
    hcho2 = gc_noisop.O_hcho
    #bg2, bglats,bglons = util.remote_pacific_background(hcho2,lats,lons)
    # difference between no isoprene and reference sector background from normal run
    diff = hcho2 - bg_ref

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
    pp.createmap(hcho1,lats,lons,region=ausregion, vmin=vmin,vmax=vmax, 
                 clabel=clabel, title='tropchem')
    plt.subplot(2,2,2)
    pp.createmap(bg1,bglats,bglons,region=bgregion, vmin=vmin,vmax=vmax, 
                 clabel=clabel, title='tropchem over Pacific ocean')
    plt.subplot(2,2,3)
    pp.createmap(hcho2,lats,lons,region=ausregion, vmin=vmin,vmax=vmax, 
                 clabel=clabel, title='no isoprene emitted')
    plt.subplot(2,2,4)
    pp.createmap(diff,lats,lons,region=ausregion, vmin=vmin,vmax=vmax, 
                 clabel=clabel, title='no isoprene - reference sector',
                 pname='Figs/GC/GC_background_hcho_%s.png'%month.strftime('%Y%m'))

# TODO update titles to just \Omega_{GC}, update legends to a priori or E_{GC}] and \Omega_{GC}, update y axes also
def Examine_Model_Slope(month=datetime(2005,1,1),use_smear_filter=False):
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
    Eisop_max=1.4e13
    xlims=np.array([Eisop_min,Eisop_max])
    slopemin=1e3
    slopemax=1e5
    cmapname='gnuplot'
    
    # plot names
    yyyymm=month.strftime('%Y%m')
    pname = 'Figs/GC/E_isop_vs_hcho%s_%s.png'%(['','_sf'][use_smear_filter], yyyymm)

    # Get slope and stuff we want to plot
    model   = GC.model_slope(return_X_and_Y=True)
    lats    = model['lats']
    lons    = model['lons']
    sfstr   = ['','sf'][use_smear_filter]
    hcho    = model['hcho'+sfstr]
    isop    = model['isop'+sfstr]
    # Y=slope*X+b with regression coeff r
    reg     = model['r'+sfstr]
    off     = model['b'+sfstr]
    slope   = model['slope'+sfstr]
    ocean   = util.oceanmask(lats,lons)
    hcho[:,ocean] = np.NaN
    isop[:,ocean] = np.NaN

    fullpageFigure() # set up full page figure size and fonts and etc...
    f,axes=plt.subplots(2,2)

    # Now plot the slope and r^2 on the map of Aus:
    plt.sca(axes[0,0]) # first plot slope
    vmin=1e-7
    slope[slope < vmin] = np.NaN # Nan the zeros and negatives for nonlinear plot
    pp.createmap(slope, lats, lons, vmin=slopemin, vmax=slopemax,
                 aus=True, linear=False, cmapname=cmapname,
                 suptitle="HCHO trop column vs isoprene emissions %s"%ymstr,
                 clabel='s', title="Slope")
    plt.sca(axes[1,0]) # then plot r2 and save figure
    bmap,cs,cb = pp.createmap(reg**2,lats,lons,vmin=0,vmax=1.0,
                              aus=True,linear=True, cmapname=cmapname,
                              clabel='', title='r$^2$')

    # plot time series (spatially averaged)
    ts_isop=np.nanmean(isop,axis=(1,2))
    ts_hcho=np.nanmean(hcho,axis=(1,2))
    plt.sca(axes[0,1])
    pp.plot_time_series(dates,ts_isop,ylabel=__apri__label__,
        title='Australian midday mean', dfmt="%d", color='r',legend=False, label=__apri__)
    h1, l1 = axes[0,1].get_legend_handles_labels()
    twinx=axes[0,1].twinx()
    plt.sca(twinx)
    pp.plot_time_series(dates,ts_hcho,ylabel=r'HCHO [ molec cm$^{-2}$ ]',
        xlabel='day', dfmt="%d", color='m', legend=False, label='HCHO')
    h2, l2 = twinx.get_legend_handles_labels()
    plt.legend(h1+h2, l1+l2, loc='best')

    plt.sca(axes[0,1])
    plt.autoscale(True)
    # plot a sample of ii_max scatter plots and their regressions
    ii=0; ii_max=9
    colours=[matplotlib.cm.Set1(i) for i in np.linspace(0, 0.9, ii_max)]
    randlati= np.random.choice(range(len(lats)), size=30)
    randloni= np.random.choice(range(len(lons)), size=30)
    # UPDATE: using sydney, west of sydney, and 2 west 1 north of sydney as sample
    # also use mid queensland
    samplelats = [-34, -34, -32, -22]
    samplelons = [ 151, 148.5, 146, 145]
    randlati, randloni=[],[]
    for i in range(len(samplelats)):
        randlattmp, randlontmp  = util.lat_lon_index(samplelats[i],samplelons[i],lats,lons)
        randlati.append(randlattmp)
        randloni.append(randlontmp)

    # loop over random lats and lons
    for xi,yi in zip(randloni,randlati):
        if ii==ii_max: break
        lat=lats[yi]; lon=lons[xi]
        X=isop[:,yi,xi]; Y=hcho[:,yi,xi]
        if np.isclose(np.nanmean(X),0.0) or np.isnan(np.nanmean(X)): continue

        # add dot to map
        plt.sca(axes[1,0])
        bmap.plot(lon,lat,latlon=True,markersize=10,marker='o',color=colours[ii])

        # Plot scatter and regression
        plt.sca(axes[1,1])
        plt.scatter(X,Y,color=colours[ii])
        m,b,r = slope[yi,xi],off[yi,xi],reg[yi,xi]
        plt.plot(xlims, m*xlims+b,color=colours[ii],
                 label='Y = %.1eX + %.2e, r=%.2f'%(m,b,r))
            #label='Y[%5.1fS,%5.1fE] = %.1eX + %.2e, r=%.2f'%(-1*lat,lon,m,b,r))

        ii=ii+1
    plt.xlim(xlims)
    plt.ylim([hcho_min,hcho_max])
    plt.xlabel(__apri__label__)
    plt.ylabel(r'HCHO [ molec cm$^{-2}$ ]')
    plt.title('Sample of regressions')
    plt.legend(loc=0) # show legend

    plt.savefig(pname)
    plt.close()
    print("SAVED: ",pname)


### 
# RESULTS
###

def time_series(d0=datetime(2005,1,1), d1=datetime(2012,12,31)):
    '''
        Time series before and after scaling 
        Method takes 5 mins but lots of RAM to read satellite years...
    '''
    
    pnames = 'Figs/new_emiss/time_series_%s.png'
    
    #read satellite overpass outputs
    DF = read_overpass_timeseries()
    
    dates=[datetime.strptime(dstr, '%Y-%m-%d') for dstr in DF.index]
    
    keys = ['HCHO_TotCol', 'isop', 'hcho', 'O3', 'NOx']
    units = [__Ogc__units__, 'ppbC', 'ppbv', 'ppbv','ppbv']
    titles= [__Ogc__, 'surface isoprene', 'surface HCHO', 'surface ozone', 'surface NO$_x$']
    suptitles = [ 'Seasonally averaged %s [%s]'%(lab,unit) for lab,unit in zip(titles,units) ]
    
    # plot series seasonally averaged
    # ONE plot per key
    for key, suptitle in zip(keys,suptitles):
        
        
        # Priori and posteriori overpass output
        # KEY_REGION_PRI/POST_METRIC
        col=[ '%s_%%s_%s_mean'%(key,pripost) for pripost in ['pri','post'] ]
        
        # time series
        new_regional_ts = [ DF[col[1]%reg] for reg in labels ]
        trop_regional_ts = [ DF[col[0]%reg] for reg in labels ]

        # Seasonal averages
        new_seasonal = [ util.resample(new_regional_ts[i],dates,"Q-NOV") for i in range(n_regions) ]
        trop_seasonal = [ util.resample(trop_regional_ts[i],dates,"Q-NOV") for i in range(n_regions) ]
        
        # dates are at right hand side of bin by default...
        dates_seasonal = new_seasonal[0].mean().index.to_pydatetime()
        dates_seasonal = [ date_s - timedelta(days=45) for date_s in dates_seasonal ]
        
        f,axes = plt.subplots(n_regions,1,figsize=[16,12], sharex=True,sharey=True)
        for i in range(n_regions):
            plt.sca(axes[i])
            newmean      = new_seasonal[i].mean().values.squeeze()
            tropmean     = trop_seasonal[i].mean().values.squeeze()

            plt.plot_date(dates_seasonal, tropmean, color=colors[i], label='Tropchem run',
                          fmt='-', linewidth=3)
            #plt.fill_between(x,lq,uq, color=colors[i], alpha=0.4)
            plt.plot_date(dates_seasonal, newmean, color=colors[i], label='Scaled run',
                          fmt='--',linewidth=3 )
            #plt.fill_between(x, lqmeg,uqmeg, color=colors[i], alpha=0.5, facecolor=colors[i], hatch='X', linewidth=0)
            plt.ylabel(labels[i], color=colors[i], fontsize=24)
            if i==0:
                plt.legend(loc='best')
            if i%2 == 1:
                axes[i].yaxis.set_label_position("right")
                axes[i].yaxis.tick_right()
        
        plt.xlabel('date', fontsize=24)
        plt.suptitle(suptitle,fontsize=30)
        f.subplots_adjust(hspace=0)


        ## save figure
        plt.savefig(pnames%key)
        print('SAVED ',pnames%key)
        plt.close()

def old_time_series(d0=datetime(2005,1,1), d1=datetime(2012,12,31)):
    '''
        Time series before and after scaling 
        Method takes 5 mins but lots of RAM to read satellite years...
    '''
    
    pnames = 'Figs/new_emiss/time_series_%s.png'
    
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
               'IJ-AVG-$_NO',       # NO in ppb?
               'IJ-AVG-$_O3',       # O3 in ppb
               ] #+ GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    
    
    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates
    months=util.list_months(d0,d1)
    
    # Same process for each key: read surface, split by region, plot series seasonally averaged
    for origkey in satkeys:
        key = GC_class._GC_names_to_nice[origkey]
        
        # Grab surface array
        new_surf = getattr(new_sat,key)[:,:,:,0] # ppb or ppbC
        trop_surf = getattr(tropchem_sat,key)[:,:,:,0]
        
        units = 'ppbv'
        if origkey == 'IJ-AVG-$_ISOP':
            units = 'ppbC'
        
        # pull out subregions, keeping lats and lons
        new_regional, lats_regional, lons_regional = util.pull_out_subregions(new_surf,lats,lons,subregions=regions)
        trop_regional, lats_regional, lons_regional = util.pull_out_subregions(trop_surf,lats,lons,subregions=regions)

        # average spatial dims into monthly time series
        new_regional_ts = [ np.nanmean(new_regional[i], axis=(1,2)) for i in range(n_regions) ]
        trop_regional_ts = [ np.nanmean(trop_regional[i], axis=(1,2)) for i in range(n_regions) ]

        # Seasonal averages
        new_seasonal = [ util.resample(new_regional_ts[i],dates,"Q-NOV") for i in range(n_regions) ]
        trop_seasonal = [ util.resample(trop_regional_ts[i],dates,"Q-NOV") for i in range(n_regions) ]
        dates_seasonal = new_seasonal[0].mean().index.to_pydatetime()
        # dates are at right hand side of bin by default...
        dates_seasonal = [ date_s - timedelta(days=45) for date_s in dates_seasonal ]
        
        f,axes = plt.subplots(n_regions,1,figsize=[16,12], sharex=True,sharey=True)
        for i in range(n_regions):
            plt.sca(axes[i])
            newmean      = new_seasonal[i].mean().values.squeeze()
            tropmean     = trop_seasonal[i].mean().values.squeeze()

            plt.plot_date(dates_seasonal, tropmean, color=colors[i], label='Tropchem run',
                          fmt='-', linewidth=3)
            #plt.fill_between(x,lq,uq, color=colors[i], alpha=0.4)
            plt.plot_date(dates_seasonal, newmean, color=colors[i], label='Scaled run',
                          fmt='--',linewidth=3 )
            #plt.fill_between(x, lqmeg,uqmeg, color=colors[i], alpha=0.5, facecolor=colors[i], hatch='X', linewidth=0)
            plt.ylabel(labels[i], color=colors[i], fontsize=24)
            if i==0:
                plt.legend(loc='best')
            if i%2 == 1:
                axes[i].yaxis.set_label_position("right")
                axes[i].yaxis.tick_right()
        
        plt.xlabel('date', fontsize=24)
        plt.suptitle('Seasonally averaged surface %s [%s]'%(key, units),fontsize=30)
        f.subplots_adjust(hspace=0)


        ## save figure
        plt.savefig(pnames%key)
        print('SAVED ',pnames%key)
        plt.close()

def trend_analysis(d0=datetime(2005,1,1),d1=datetime(2012,12,31)):
    '''
        Trends for surface ozone, hcho, isop, NO, NO2
        Method takes 5 mins but lots of RAM to read satellite years...
    '''
    
    pnames = 'Figs/new_emiss/trend_%s.png'
    
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
               'IJ-AVG-$_NO',       # NO in ppb?
               'IJ-AVG-$_O3',       # O3 in ppb
               ] #+ GC_class.__gc_tropcolumn_keys__
    print("TREND: READING new_emissions")
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    print("TREND: READING tropchem")
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('TREND: GEOS-Chem satellite outputs read')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    
    
    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates
    months=util.list_months(d0,d1)
    
    titles=['%s monthly anomaly', '%s monthly anomaly (scaled)']
    
    
    # Same process for each key: read surface, split by region, plot trends
    for origkey in satkeys:
        key = GC_class._GC_names_to_nice[origkey]
        
        # Grab surface array
        new_surf = getattr(new_sat,key)[:,:,:,0] # ppb or ppbC
        trop_surf = getattr(tropchem_sat,key)[:,:,:,0]
        
        # will be printing mean differences
        print('area:key,   new_emiss,   tropchem')
        
        # For each region plot deseasonalised mean and trend
        
        units = 'ppbv'
        if origkey == 'IJ-AVG-$_ISOP':
            units = 'ppbC'
        
        f, axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)
        
        for j, (arr,title) in enumerate(zip([trop_surf, new_surf],titles)):
            #mya_df = util.multi_year_average_regional(arr, dates, lats, lons, grain='monthly', regions=regions)
            #mya = [np.squeeze(np.array(mya_df['df'][i].mean())) for i in range(n_regions)]
            # pull out subregions, keeping lats and lons
            regional, lats_regional, lons_regional = util.pull_out_subregions(arr,lats,lons,subregions=regions)
            
            # average spatial dims into monthly time series
            regional_ts = [ np.nanmean(regional[i], axis=(1,2)) for i in range(n_regions) ]
            
            # get trend
            trends = [ util.trend(regional_ts[i], dates, remove_mya=True, resample_monthly=True, remove_outliers=True) for i in range(n_regions) ]
            
            #anomaly = [ trends[i]['anomaly'] for i in range(n_regions) ]
            #monthly = [ trends[i]['monthly'] for i in range(n_regions) ]
            
            #monthly = [ np.array(util.resample(np.nanmean(regional[i],axis=(1,2)),dates,bins='M').mean()).squeeze() for i in range(len(regions))]
            #anomaly = [ np.array([ old_monthly[k][i] - mya[k][i%12] for i in range(len(months)) ]) for k in range(len(regions)) ]
            
            plt.sca(axes[j])
            print(title%key)
            print('region, [ slope, regression coeff, p value for slope non zero]')
            for i in range(n_regions):
            #for monthly_anomaly, monthly_data, color, label in zip(anomaly, monthly, colors, labels):
                color=colors[i]; label=labels[i]
                
                trendi=trends[i]
                #monthly = trendi['monthly']
        
                # Calculate with all data (for interest)
                #m, b, r, cir, cijm = RMA(np.arange(len(months)), monthly_anomaly)
                #print("%s (has outliers) &  [ %.2e,  %.2e ]   & \\"%(label,cir[0][0], cir[0][1]))
                
                # Y = mX+b
                anomaly = trendi['anomaly']
                m=trendi['slope']
                b=trendi['intercept']
                r=trendi['r']
                p=trendi['p'] # two sided p value against slope being zero as H0
                outliers=trendi['outliers']
                
                print("%s &  [ m=%.2e,  r=%.3f, p=%.3f ]   & \\"%(label,m, r, p))
                
                # once more with outliers removed
                #std=np.nanstd(monthly_anomaly)
                #mean=np.nanmean(monthly_anomaly)
                #outliers = ( monthly_anomaly > mean + 3 * std ) + ( monthly_anomaly < mean - 3*std )
                #Xin = np.arange(len(months))[~outliers]
                #Yin = monthly_anomaly[~outliers]
                
                #m, b, r, cir, cijm = RMA(Xin, Yin)
                #print("%s (no outliers)  &  [ %.2e,  %.2e ]   & \\"%(label,cir[0][0], cir[0][1]))
                #print(m/np.nanmean(monthly_data) * 12)
                #if cir[0][0] * cir[0][1] > 0: # if slope doesn't surround zero then we plot line
                if p < 0.05:
                    pp.plot_time_series( [months[0], months[-1]], [b,b+m*(len(months)-1)], alpha=0.5, color=color ) # regression line
                    print("significant! ", key, label)
                pp.plot_time_series( months, anomaly, color=color, label=label, marker='.', markersize=6, linewidth=0) # regression line
                pp.plot_time_series( np.array(months)[outliers], anomaly[outliers], color=color, marker='x', markersize=8, linewidth=0)
            plt.title(title%key,fontsize=24)
            plt.ylabel(units)
            plt.xticks(util.list_years(d0,d1))
            plt.xlim(d0 - timedelta(days=31), d1+timedelta(days=31))
        
        plt.sca(axes[0])
        xywh=(.985, 0.15, 0.1, .7)
        plt.legend(bbox_to_anchor=xywh, loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=12, scatterpoints=1)
        
        plt.savefig(pnames%key)
        print('SAVED ',pnames%key)
        plt.close()


def seasonal_differences():
    ''' 
        Compare HCHO, O3, NO columns between runs over Australia
        # Grab all overpass data, resample to multiyear monthly avg, then look at seasonal compariosn
        First row:  Summer before, summer after, diff
        Second row: Winter before, winter after, diff
    '''
    d0 = datetime(2005,1,1)
    d1 = datetime(2006,12,31)
    print("CURRENTLY TESTING: NEED TO SET d1 TO 2012/12/31")
    #dstr = d0.strftime("%Y%m%d")
    pname1 = 'Figs/new_emiss/HCHO_total_columns_seasonal.png'
    pname2 = 'Figs/new_emiss/O3_surf_map_seasonal.png'
    pname3 = 'Figs/new_emiss/NOx_surf_map_seasonal.png'

    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbC
               'IJ-AVG-$_CH2O',     # hcho in ppbv
               'IJ-AVG-$_NO',       # NO in ppbv
               'IJ-AVG-$_NO2',     # NO2 in ppbv
               'IJ-AVG-$_O3', ] + GC_class.__gc_tropcolumn_keys__

    GCnew = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    GCtrop = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('GEOS-Chem satellite outputs read 2005')
    lats=GCnew.lats
    lons=GCnew.lons
    dates=GCnew.dates
    
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)

    # TOTAL column HCHO
    new_hcho  = GCnew.get_total_columns(keys=['hcho'])['hcho']
    trop_hcho  = GCtrop.get_total_columns(keys=['hcho'])['hcho']
    # surface O3 in ppbv
    new_o3 = GCnew.O3[:,:,:,0]
    trop_o3 = GCtrop.O3[:,:,:,0]
    # Surface NOx in ppbv
    print("NO units: ",GCnew.attrs['NO']['units'], " NO2 units: ", GCnew.attrs['NO2']['units'])
    new_NOx = GCnew.NO2[:,:,:,0] + GCnew.NO[:,:,:,0]

    trop_NOx = GCtrop.NO2[:,:,:,0] + GCtrop.NO[:,:,:,0]
    
    # MYA monthly averages:
    summer=np.array([0,1,11])
    winter=np.array([5,6,7])
    new_summers=[]
    new_winters=[]
    trop_summers=[]
    trop_winters=[]
    
    # HCHO,   O3,   NOX  simple comparisons:
    f=plt.figure()
    vmins = [1e15, 20, 0.01]
    vmaxs = [2e16, 50, 1]
    difflims = [[-3.5e15, 3.5e15], [-4,4], [-0.05,0.05]]
    units = ['molec cm$^{-2}$', 'ppbv', 'ppbv']
    linears= [False,True,False]
    stitles = ['Midday total column HCHO','Midday surface ozone','Midday surface NO$_x$']
    titles = ['Tropchem run','Scaled run', 'Absolute difference']
    pnames = [pname1,pname2,pname3]
    for i, new_arr, trop_arr in zip(range(3),[new_hcho, new_o3, new_NOx],[trop_hcho, trop_o3, trop_NOx]):
        new_mya = util.multi_year_average_spatial(new_arr, dates)
        trop_mya = util.multi_year_average_spatial(trop_arr, dates)
        new_summers.append(np.nanmean(new_mya['mean'][summer,:,:],axis=0))
        new_winters.append(np.nanmean(new_mya['mean'][winter,:,:],axis=0))
        trop_summers.append(np.nanmean(trop_mya['mean'][summer,:,:],axis=0))
        trop_winters.append(np.nanmean(trop_mya['mean'][winter,:,:],axis=0))
        
        # SUMMER PLOTS:
        vmin=vmins[i]; vmax=vmaxs[i]
        dmin=difflims[i][0]; dmax=difflims[i][1]
        plt.subplot(2,3,1)
        pp.createmap(new_summers[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, 
                     clabel=units[i], linear=linears[i], title=titles[0])
        plt.ylabel('Summer')
        plt.subplot(2,3,2)
        pp.createmap(trop_summers[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, 
                     clabel=units[i], linear=linears[i], title=titles[1])
        plt.subplot(2,3,3)
        pp.createmap(new_summers[i]-trop_summers[i],lats,lons,aus=True, vmin=dmin,vmax=dmax, 
                     clabel=units[i], linear=True, title=titles[2], 
                     cmapname='bwr')

        # WINTER PLOTS:
        plt.subplot(2,3,4)
        pp.createmap(new_winters[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, 
                     clabel=units[i], linear=linears[i], title=titles[0])
        plt.ylabel('Winter')
        plt.subplot(2,3,5)
        pp.createmap(trop_winters[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, 
                     clabel=units[i], linear=linears[i], title=titles[1])
        plt.subplot(2,3,6)
        pp.createmap(new_winters[i]-trop_winters[i],lats,lons,aus=True, vmin=dmin,vmax=dmax, 
                     clabel=units[i], linear=True, title=titles[2], 
                     cmapname='bwr')
        
        
        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        plt.suptitle(stitles[i])
        plt.savefig(pnames[i])
        plt.close(f)
        print('SAVING FIGURE ',pnames[i])
        

    
def regional_seasonal_comparison():
    ''' 
        Compare HCHO, O3, NO columns between runs over Australia
        # Grab all overpass data, resample to multiyear monthly avg, then look at seasonal compariosn
        First row:  Summer before, summer after, diff
        Second row: Winter before, winter after, diff
    '''
    pname = 'Figs/new_emiss/RegSeas_emissions.png'
    
    #read satellite overpass outputs
    DF = read_overpass_timeseries()
    dates=[datetime.strptime(dstr, '%Y-%m-%d') for dstr in DF.index]
    
    keys = ['E_MEGAN','E_PP_lr']
    klabels = [__apri__, __apost__]
    suptitle='Midday emissions [%s]'%__apri__units__
    #units = [__Ogc__units__, 'ppbC', 'ppbv', 'ppbv','ppbv']
    #titles= [__Ogc__, 'surface isoprene', 'surface HCHO', 'surface ozone', 'surface NO$_x$']
    #suptitles = [ 'Seasonally averaged %s [%s]'%(lab,unit) for lab,unit in zip(titles,units) ]
    
    
    # Priori and posteriori overpass output
    # Time series
    apris=   [ DF['E_MEGAN_%s_mean'%reg] for reg in labels ]
    aposts=  [ DF['E_PP_lr_%s_mean'%reg] for reg in labels ]
    
    # Seasonal averages
    apri_seasonal = [ util.resample(apris[i],dates,"Q-NOV") for i in range(n_regions) ]
    apost_seasonal = [ util.resample(aposts[i],dates,"Q-NOV") for i in range(n_regions) ]
    
    
    f,axes = plt.subplots(n_regions,1,figsize=[16,12], sharex=True,sharey=True)
    for i in range(n_regions):
        plt.sca(axes[i])
        apriseasons    = [ np.nanmean(apri_seasonal[i].mean().values.squeeze()[j::4]) for j in range(4) ]
        apostseasons   = [ np.nanmean(apost_seasonal[i].mean().values.squeeze()[j::4]) for j in range(4) ]

        X = np.arange(4)
        width=0.4
        plt.bar(X + 0.00, apriseasons, color = 'm', width = width, label=__apri__)
        plt.bar(X + width, apostseasons, color = 'cyan', width = width, label=__apost__)
        plt.xticks()
        plt.ylabel(labels[i], color=colors[i], fontsize=24)
        
        if i==0:
            plt.legend(loc='best', fontsize=18)
        if i%2 == 1:
            axes[i].yaxis.set_label_position("right")
            axes[i].yaxis.tick_right()
    
    plt.xticks(X+width, ['summer','autumn','winter','spring'])
    plt.xlabel('season', fontsize=24)
    plt.suptitle(suptitle,fontsize=30)
    f.subplots_adjust(hspace=0)


    ## save figure
    plt.savefig(pname)
    print('SAVED ',pname)
    plt.close()


################
### UNCERTAINTY
################

def uncertainty():
    '''
        Calculate uncertainty and plot it somehow
    '''
    
    # Read product with pixels and uncertainty
    # OMHCHORP: 
    #   Need: VC_OMI, col_uncertainty_OMI, entries
    # E_new: 
    #   Need: E_PP_lr, ModelSlope, 
    d0=datetime(2005,1,1)
    d1=datetime(2005,1,31)
    uncertkeys = ['VC_relative_uncertainty_lr','pixels_lr']
    # Read from E_new
    enew    = E_new(d0,d1, dkeys=uncertkeys)
    VCrunc  = enew.VC_relative_uncertainty_lr
    pix     = enew.pixels_lr
    lats    = enew.lats_lr
    lons    = enew.lons_lr
    
    # daily means
    dVCrunc = np.nanmean(VCrunc, axis=0)
    dpix    = np.nanmean(pix, axis=0)
    
    # monthly uncertainty: reweight using daily pixel counts 
    with np.errstate(divide='ignore'):
        mVCrunc = ( 1 / np.sqrt(np.nansum(pix,axis=0)) ) * ( np.nanmean(np.sqrt(pix) * VCrunc, axis=0) )
    
    dvmin=0.001
    dvmax=3
    dbins=np.logspace(-1,1,25)
    mvmax=1
    mbins=np.logspace(-2,0,25)
        
    # plot daily averages
    plt.subplot(3,2,1)
    pp.createmap(dVCrunc, lats, lons, aus=True, linear=True,
                 vmin=dvmin,vmax=dvmax,
                 title='Relative uncertainty')
    plt.subplot(3,2,2)
    pp.createmap(dpix, lats, lons, aus=True, linear=True,
                 title='OMI daily pixel count')
    # Distribution of daily
    plt.subplot(3,1,2)
    OM=util.oceanmask(lats,lons)
    dVCrunc[dVCrunc>dvmax] = 10
    plt.hist(dVCrunc[~OM], bins=dbins)
    plt.xscale('log')
    plt.title('histogram over land squares (mean daily)')
    
    # Distribution of monthly
    plt.subplot(3,1,3)
    mVCrunc[mVCrunc>mvmax] = mvmax
    plt.hist(mVCrunc[~OM], bins=mbins)
    plt.xscale('log')
    plt.title('histogram over land squares (monthly)')
    
    # add by quadrature to assumed error in AMF
    
    pname='test_uncert.png'
    plt.savefig(pname)
    plt.close(pname)

def pixel_counts_summary():
    
    d0 = datetime(2005,1,1)
    d1 = datetime(2012,12,31)
    
    #dstr = d0.strftime("%Y%m%d")
    pname1 = 'Figs/pixel_count_seasonally.png'
    pname2 = 'Figs/pixel_count_barchart.png'
    pname3 = 'Figs/pixel_count_barchart_unfiltered.png'
    
    #read satellite overpass outputs
    DF = read_overpass_timeseries()
    dates=[datetime.strptime(dstr, '%Y-%m-%d') for dstr in DF.index]
    
    key = 'pixels_PP_lr'
    keyf= 'pixels_PP'
    keyu= 'pixels_PP_u'
    suptitle='Mean pixel count per grid square per day'
    suptitle2='Mean pixel count per season per region'
    # Time series
    TS =   [ DF['%s_%s_mean'%(key,reg)] for reg in labels ]
    TSf = [ DF['%s_%s_sum'%(keyf,reg)] for reg in labels ]
    TSu = [ DF['%s_%s_sum'%(keyu,reg)] for reg in labels ]
    
    
    TS_seasonal = [ util.resample(TS[i],dates,"Q-NOV") for i in range(n_regions) ]
    TS_seasonalu= [ util.resample(TSu[i],dates,"Q-NOV") for i in range(n_regions) ]
    TS_seasonalf= [ util.resample(TSf[i],dates,"Q-NOV") for i in range(n_regions) ]
    
    # summary of used pixels
    f,axes = plt.subplots(n_regions,1,figsize=[16,12], sharex=True,sharey=True)
    X = np.arange(4)
    
    # summary that includes filtered and unfiltered pixel counts
    f2, axes2 = plt.subplots(n_regions,1,figsize=[16,12], sharex=True,sharey=False)
    width=0.4
    
    #
    for i in range(n_regions):
        
        seasonal = TS_seasonal[i].mean().values.squeeze()
        seasonalf= TS_seasonalf[i].sum().values.squeeze()
        seasonalu= TS_seasonalu[i].sum().values.squeeze()
        
        # Multiyear Seasonal summary
        #Also show temporal std. for mean pixel count
        myamean         = [ np.nanmean(seasonal[j::4]) for j in range(4) ]
        myastd          = [ np.nanstd(seasonal[j::4]) for j in range(4) ]
        myameanu         = [ np.nanmean(seasonalu[j::4]) for j in range(4) ]
        myastdu          = [ np.nanstd(seasonalu[j::4]) for j in range(4) ]
        myameanf         = [ np.nanmean(seasonalf[j::4]) for j in range(4) ]
        myastdf          = [ np.nanstd(seasonalf[j::4]) for j in range(4) ]
        

        plt.sca(axes[i])
        plt.bar(X + 0.00, myamean, color = 'cyan', yerr=myastd, ecolor='k', capsize=8, )# width = width, label='filtered')
        #plt.bar(X + width, apostseasons, color = 'cyan', width = width, label=__apost__)
        plt.xticks()
        plt.ylabel(labels[i], color=colors[i], fontsize=24)
        
        # Compare to unfiltered version
        plt.sca(axes2[i])
        plt.bar(X + 0.00, myameanu, color = 'm', yerr=myastdu, ecolor='k', capsize=8, width = width, label='unfiltered')
        plt.bar(X + width, myameanf, color = 'cyan', yerr=myastdf, ecolor='k', capsize=8, width = width, label='filtered')
        
        #if i==0:
        #    plt.legend(loc='best', fontsize=18)
        if i%2 == 1:
            for ax in [axes, axes2]:
                ax[i].yaxis.set_label_position("right")
                ax[i].yaxis.tick_right()
    
    
    
    plt.sca(axes[i])
    plt.xlabel('season', fontsize=24)    
    plt.xticks(X, ['summer','autumn','winter','spring'])
    plt.suptitle(suptitle,fontsize=30)
    f.subplots_adjust(hspace=0)
    ## save figure
    plt.savefig(pname2)
    print('SAVED ',pname2)
    plt.close()
    
    plt.sca(axes2[i])
    plt.xlabel('season', fontsize=24)    
    plt.xticks(X+width, ['summer','autumn','winter','spring'])
    plt.suptitle(suptitle2,fontsize=30)
    f2.subplots_adjust(hspace=0)
    ## save figure
    plt.savefig(pname3)
    print('SAVED ',pname3)
    plt.close()



if __name__ == "__main__":
    
    start=timeit.default_timer()
    
    # set up plotting parameters like font sizes etc..
    fullpageFigure()
    
    
    
    ## METHOD PLOTS
    
    #check_modelled_background() # finished? 24/2/19
    #Examine_Model_Slope() # finished ~ 20/2/19
    
    ## Results Plots
    
    # TODO: trend_analysis barplot summary
    #seasonal_differences()
    # TODO: time series compared to satellite HCHO
    # TODO: Seasonal regional multiyear comparison
    #regional_seasonal_comparison()
    #time_series()
    # TODO: 
    
    ## UNCERTAINTY
    #TODO: implement
    #uncertainty()
    # TODO: add sums to analysis TS
    # todo: discuss plot output from 
    pixel_counts_summary()

    ### Record and time STUJFFS
    
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for stuff"%((end-start)/60.0))
    

