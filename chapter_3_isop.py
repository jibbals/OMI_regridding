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
import utilities.utilities as util
import utilities.plotting as pp
from utilities.JesseRegression import RMA, OLS
from utilities import GMAO,GC_fio,fio, masks
import Inversion
import tests
from tests import utilities_tests, test_new_emissions
import reprocess
import new_emissions
import Analyse_E_isop

from classes.E_new import E_new # E_new class
from classes import GC_class, campaign
from classes.omhchorp import omhchorp

from utilities import masks

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
# total column HCHO from OMI (recalculated using PP code)
__Oomi__= "$\Omega_{OMI}$"
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


def trend_analysis(d0=datetime(2005,1,1),d1=datetime(2012,12,31)):
    '''
        Trends for surface ozone, hcho, isop, NO2?
        Method takes 5 mins but lots of RAM to read satellite years...
    '''
    
    pnames = 'Figs/new_emiss/trend_%s.png'
    
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
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
        
        units = 'ppb'
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
    VCrunc  = np.nanmean(enew.VC_relative_uncertainty_lr, axis=0)
    pix     = np.nanmean(enew.pixels_lr, axis=0)
    lats    = enew.lats_lr
    lons    = enew.lons_lr
    
    # plot daily averages
    plt.subplot(2,2,1)
    pp.createmap(VCrunc, lats, lons, aus=True, linear=True,
                 vmin=0.001, vmax=10,
                 title='Relative uncertainty')
    plt.subplot(2,2,2)
    pp.createmap(pix, lats, lons, aus=True, linear=True,
                 title='OMI daily pixel count')
    # plot monthly average
    plt.subplot(2,1,2)
    OM=util.oceanmask(lats,lons)
    plt.hist(VCrunc[~OM], bins=np.linspace(0,5,21))
    plt.title('histogram over land squares')
    
    # average/ sqrt(n)
    
    # add by quadrature to assumed error in AMF
    
    pname='test_uncert.png'
    plt.savefig(pname)
    plt.close(pname)


if __name__ == "__main__":
    
    start=timeit.default_timer()
    
    # set up plotting parameters like font sizes etc..
    fullpageFigure()
    
    ## METHOD PLOTS
    
    #check_modelled_background() # finished? 24/2/19
    #Examine_Model_Slope() # finished ~ 20/2/19
    
    ## Results Plots
    
    ## UNCERTAINTY
    uncertainty()


    ### Record and time STUJFFS
    
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for stuff"%((end-start)/60.0))
    

