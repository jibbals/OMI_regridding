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

## Summarise a data set over subregions
def summarise(arr, dates, lats, lons, label):
    ''' regional seasonal mean, median, std within subregions after taking monthly averages '''
    
    months = util.list_months(dates[0],dates[-1])
    # subset to AUS and remove ocean
    subs = util.lat_lon_subset(lats,lons,pp.__AUSREGION__,data=[arr], has_time_dim=True)
    darr = subs['data'][0]
    dlats, dlons = subs['lats'], subs['lons']
    om = util.oceanmask(dlats,dlons)
    om = np.repeat(om[np.newaxis,:,:], len(dates), axis=0)
    #print("shape of arrays in summarise")
    #print(np.shape(om),np.shape(darr),np.shape(lats),np.shape(dlats),np.shape(lons),np.shape(dlons))
    darr[om] = np.NaN
    
    # monthly averaged
    monthly = util.monthly_averaged(dates, darr, keep_spatial=True)
    darr = monthly['mean']
    
    # pull out summers
    summers=np.array([ i%12 in [0,1,11] for i in range(len(months)) ])
    autumns=np.array([ i%12 in [2,3,4] for i in range(len(months)) ])
    
    winters=np.array([ i%12 in [5,6,7] for i in range(len(months)) ])
    springs=np.array([ i%12 in [8,9,10] for i in range(len(months)) ])
    
    # sub regions
    darrsub, lats_regional, lons_regional = util.pull_out_subregions(darr,dlats,dlons,subregions=regions)
    
    reg_str = []
    
    print(label)
    print("region:    summer   , autumn   , winter   , spring")
    for i in range(n_regions):        
        darr_sum = darrsub[i][summers,:,:]
        darr_aut = darrsub[i][autumns,:,:]
        darr_win = darrsub[i][winters,:,:]
        darr_spr = darrsub[i][springs,:,:]
        
        
        inner_str = [labels[i]+':   ']
        for reglab, darray in zip(['SUMMER','AUTUMN','WINTER','SPRING'],[darr_sum, darr_aut, darr_win, darr_spr]):
            inner_str.append("%.1f(%.1f)"%(np.nanmean(darray),np.nanstd(darray)))
        print(inner_str)
        

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
    dkeys_lr = ['E_PP_lr','E_MEGAN','pixels_lr','pixels_PP_lr',]
    enew=E_new(d0,d1,dkeys=dkeys)
    lats,lons = enew.lats, enew.lons
    lats_lr=enew.lats_lr
    lons_lr=enew.lons_lr
    
    assert np.all(dates==enew.dates), "Dates don't match"
    
    # Make sure ocean squares are zero'd 
    oceanmask=util.oceanmask(lats,lons)
    oceanmask_lr=util.oceanmask(lats_lr,lons_lr)
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],len(dates),axis=0)
    oceanmask3d_lr=np.repeat(oceanmask_lr[np.newaxis,:,:],len(dates),axis=0)
    for key in dkeys:
        
        # Grab regional subsets
        enewdata=getattr(enew,key).astype(np.float64)
        if key in dkeys_lr:
            enewdata[oceanmask3d_lr] = np.NaN
            regional,lats_r,lons_r = util.pull_out_subregions(enewdata,lats_lr,lons_lr,subregions=regions)
            print(key,np.shape(enewdata),' LOW RES', np.shape(oceanmask3d_lr), len(lats_lr),len(lons_lr))
        else:
            enewdata[oceanmask3d] = np.NaN
            regional,lats_r,lons_r = util.pull_out_subregions(enewdata,lats,lons,subregions=regions)
            print(key,np.shape(enewdata),' HIGH RES', np.shape(oceanmask3d), len(lats),len(lons))
        
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
    
def PlotMultiyear(data, dates, lats,lons, weekly=False,
                  xlims=None, ylims=None, median=False, label=None):
    '''
        Split data into subregions, and take multiyear monthly or weekly means and IQR

    '''
    
    # multi-year monthly averages
    grain = ['monthly','weekly'][weekly]
    #print(data.shape, len(dates))
    mya    = util.multi_year_average_regional(data,dates,lats,lons,grain=grain,regions=regions)
    
    df     = mya['df']
    
    # monthly will be 12, weekly will be 52 or 53
    x=range(len(df[0].mean().values.squeeze()))
        
    #x=range([12,52][weekly])
    n=len(df)
    f,axes = plt.subplots(n,1, sharex=True,sharey=True)
    returns = []
    for i in range(n):
        plt.sca(axes[i])
        mean        = df[i].mean().values.squeeze()
        returns.append(mean)
        if median:
            mean    = df[i].median().values.squeeze()
            
        uq          = df[i].quantile(0.75).values.squeeze()
        lq          = df[i].quantile(0.25).values.squeeze()
        
        plt.fill_between(x,lq,uq, color=colors[i], alpha=0.4)
        plt.plot(x, mean, color=colors[i], label=label, linewidth=3)
        
        #plt.fill_between(x, lqmeg,uqmeg, color=colors[i], alpha=0.5, facecolor=colors[i], hatch='X', linewidth=0)
        #plt.plot(x, meanmeg, color=colors[i], linestyle='--', label='a priori',linewidth=3)
        plt.ylabel(labels[i], color=colors[i], fontsize=18)
        
        if i%2 == 1:
            axes[i].yaxis.set_label_position("right")
            axes[i].yaxis.tick_right()
    
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    
    monthletters=['J','F','M','A','M','J','J','A','S','O','N','D']
    #if weekly:
    plt.xticks([x,np.arange(2,52,4.5)][weekly])
    #else:
    #plt.xticks(x)
    plt.gca().set_xticklabels(monthletters)
    plt.xlabel('month', fontsize=20)
    #plt.suptitle('a priori vs a posteriori; mean and IQR\n [molec cm$^{-2}$ s$^{-1}$]',fontsize=30)
    f.subplots_adjust(hspace=0)

    ## save figure
    #plt.savefig(pname)
    #print("Saved %s"%pname)
    #plt.close()
    return returns, x, axes

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


def check_modelled_profile():
    '''
        Check profile before and after scaling
    '''
    pname_checkprof='Figs/check_GC_profile.png'
    
    LatWol, LonWol = pp.__cities__['Wol']
    
    # Read GC output
    #trop = GC_class.GC_sat(datetime(2007,8,1), datetime(2012,12,31), keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__)
    d0,d1=datetime(2005,1,1), datetime(2005,1,31)
    trop = GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__)
    tropa= GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__, run='new_emiss')
    # make sure pedges and pmids are created
    trop.add_pedges()
    
    # colours for trop and tropa
    c = 'r'
    ca= 'm'
    
    # grab wollongong square
    Woli, Wolj = util.lat_lon_index(LatWol,LonWol,trop.lats,trop.lons) # lat, lon indices
    GC_VC = trop.units_to_molec_cm2(keys=['hcho'])['hcho'][:,Woli,Wolj,:]
    GCa_VC = tropa.units_to_molec_cm2(keys=['hcho'])['hcho'][:,Woli,Wolj,:]
    
    GC_pmids=trop.pmids[:,Woli,Wolj,:]
    GC_zmids=trop.zmids[:,Woli,Wolj,:]
    
    # Total column also of interest:
    GC_TC = np.sum(GC_VC, axis=1)
    GCa_TC = np.sum(GCa_VC, axis=1)
    
    # check profile
    plt.close()
    plt.figure(figsize=[10,10])
    #ax0=plt.subplot(1,2,1)
    for i,prof in enumerate([trop.hcho[0:20,Woli,Wolj,:],tropa.hcho[0:20,Woli,Wolj,:]]):
        zmids = np.nanmean(GC_zmids[0:20,:],axis=0)/1000.0
        pmids = np.nanmean(GC_pmids[0:20,:],axis=0)
        
        mean = np.nanmean(prof,axis=0)
        lq = np.nanpercentile(prof, 25, axis=0)
        uq = np.nanpercentile(prof, 75, axis=0)
        plt.fill_betweenx(zmids, lq, uq, alpha=0.5, color=[c,ca][i])
        plt.plot(mean,zmids,label=['VMR','VMR$^{\\alpha}$'][i],linewidth=2,color=[c,ca][i])
    #plt.yscale('log')
    plt.ylim([0, 40])
    plt.ylabel('altitude [km]')
    plt.legend(fontsize=20)
    plt.xlabel('HCHO [ppbv]')
    plt.title("Wollongong midday HCHO profile Jan, 2005")
    plt.savefig(pname_checkprof)
    print("Saved ", pname_checkprof)
    
    # plot time series
    plt.close()

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

def compare_model_outputs():
    '''
        Print out seasonal regional surface values for things between tropchem and scaled run
    '''
    d0 = datetime(2005,1,1)
    d1 = datetime(2012,12,31)
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
               'IJ-AVG-$_NO',       # NO in ppb?
               'IJ-AVG-$_O3',       # O3 in ppb
               ] #+ GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    # dims for GEOS-Chem outputs
    dates=new_sat.dates
    lats=new_sat.lats
    lons=new_sat.lons
    
    
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    o3      = tropchem_sat.O3[:,:,:,0] # surface only
    o3a     = new_sat.O3[:,:,:,0]
    hcho    = tropchem_sat.hcho[:,:,:,0]
    hchoa   = new_sat.hcho[:,:,:,0]
    nox     = tropchem_sat.NO[:,:,:,0] + tropchem_sat.NO2[:,:,:,0]
    noxa    = new_sat.NO[:,:,:,0] + new_sat.NO2[:,:,:,0]
    
    for arr, label in zip([o3, o3a, hcho, hchoa, nox, noxa],['ozone','ozone_scaled', 'HCHO','HCHO_scaled', 'NOx','NOx_scaled']):
        print("summarise input shapes")
        print(np.shape(arr), np.shape(dates), np.shape(lats), np.shape(lons))
        summarise(arr, dates, lats, lons, label)

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
        if key=='NO$_x$':
            key='NOx'
        plt.savefig(pnames%key)
        print('SAVED ',pnames%key)
        plt.close()


def Seasonal_daycycle():
    '''
    '''
    d0=datetime(2005,1,1)
    dn=datetime(2012,12,31)
    # Read megan (3-hourly)
    MEGAN = GC_class.Hemco_diag(d0,dn)
    data=MEGAN.E_isop_bio # hours, lats, lons
    dates=np.array(MEGAN.dates)
    
    # Read top-down emissions (midday)
    Enew = E_new(d0,dn,dkeys=['E_PP_lr'])
    
    # average over some regions
    region_means=[]
    region_stds=[]
    topd_means=[]
    topd_stds=[]
    # UTC offsets for x axis
    # region_offsets=[11,10,10,11,12,11]
    offset=10 # utc + 10
    daylengths = util.daylengths()/60.0 # in hours

    for region in regions:
        # subset MEGAN data and serialise
        subset=util.lat_lon_subset(MEGAN.lats, MEGAN.lons, region, [data], has_time_dim=True)
        series=np.nanmean(subset['data'][0],axis=(1,2))
        
        # subset topd data
        topdsub=util.lat_lon_subset(Enew.lats_lr, Enew.lons_lr, region, [Enew.E_PP_lr], has_time_dim=True)
        topdser=np.nanmean(topdsub['data'][0],axis=(1,2))
        
        # group by month, and hour to get the multi-year monthly averaged diurnal cycle
        monthly_hours=util.multi_year_average(series,dates,grain='hourly')
        monthly_topd=util.multi_year_average(topdser, Enew.dates, grain='monthly')
        # save mean and std into [month, hour] array
        region_means.append(monthly_hours.mean().squeeze().values.reshape([12,24]))
        region_stds.append(monthly_hours.std().squeeze().values.reshape([12,24]))
        topd_means.append(monthly_topd.mean().squeeze().values)
        topd_stds.append(monthly_topd.std().squeeze().values)


    ## set up the plots
    # monthly day cycles : 4 rows 3 columns with shared axes
    f, axes = plt.subplots(4,3, sharex=True, sharey=True, figsize=(16,16))
    axes[3,1].set_xlabel('Hour (UTC+%d)'%offset)
    xlim=[6,22]
    axes[3,1].set_xlim(xlim)
    axes[1,0].set_ylabel('Emission (molec/cm2/s)')
    ylim=[1e11,1.1e13]
    axes[1,0].set_ylim(ylim)
    axes[1,0].set_yscale('log')
    titles=np.array([['Dec','Jan','Feb'],['Mar','Apr','May'],['Jun','Jul','Aug'],['Sep','Oct','Nov']])

    for r,region in enumerate(regions):
        means = region_means[r]
        stds  = region_stds[r]
        topdm = topd_means[r]
        topds = topd_stds[r]
        #offset= region_offsets[r]

        # plot the daily cycle and std range
        for i in range(4): # 4 rows
            for j in range(3): # 3 columns
                # shift forward by one month to get dec as first entry
                ii, jj = (i+int((j+1)%3==0))%4, (j+1)%3
                # grab month (map i,j onto (0-11)*24)
                #mi=i*3*24 + j*24
                mi=i*3+j #month index
                mip=(mi+1)%12
                dayhours = np.round(daylengths[mip])
                
                # grab mean and std from dataset for this month in this region
                mdata = means[mip,:].squeeze()
                mstd  = stds[mip,:].squeeze()
                
                mtopd = topdm[mip].squeeze()
                mtopds= topds[mip].squeeze()
                
                # make sin wave from topd midday mean
                sinbase=np.arange(0,dayhours+.1, 0.1)
                mtopd_sin = mtopd * np.sin(sinbase * np.pi / (dayhours) )
                sinbase = sinbase + 13.5 - dayhours/2.0
                # roll over x axis to get local time midday in the middle
                mhigh  = np.roll(mdata+mstd, offset)
                mlow   = np.roll(mdata-mstd, offset)
                mdata  = np.roll(mdata, offset)

                #plot into monthly panel, and remove ticks
                ax   = axes[ii,jj]
                plt.sca(ax)

                # remove ticks from right and top edges
                plt.tick_params(
                    axis='both',      # changes apply to the x-axis
                    which='both',     # both major and minor ticks are affected
                    right=False,      # ticks along the right edge are off
                    top=False,       # ticks along the top edge are off
                    left=jj==0,
                    bottom=ii==3)

                # first highlight the 1300-1400 time window with soft grey
                #plt.fill_betweenx(ylim,[13,13],[14,14], color='grey', alpha=0.2)
                plt.plot([13,13], ylim, color='grey',alpha=0.5)
                plt.plot([14,14], ylim, color='grey',alpha=0.5)
                

                if r == 0 :
                    plt.fill_between(np.arange(24), mhigh, mlow, color='k', alpha=0.25)
                plt.plot(np.arange(24), mdata, color=colors[r], linewidth=1+2*(r==0))
                plt.title(titles[ii,jj])
        
                # also plot topd from 1300-1400
                if r == 0:
                    
                    # Plot sin wave of correct monthly scale
                    #plt.plot([12,15], [mtopd, mtopd], color=colors[r], linewidth=2)
                    plt.plot(sinbase, mtopd_sin, '--', color=colors[r], linewidth=3)
                    
                    # Add std bars
                    plt.plot([13,14], [mtopd+mtopds, mtopd+mtopds], color=colors[r], linewidth=1)
                    plt.plot([13,14], [mtopd-mtopds, mtopd-mtopds], color=colors[r], linewidth=1)
                    plt.plot([13.5,13.5], [mtopd-mtopds, mtopd+mtopds], color=colors[r], linewidth=1)
                    


    # remove gaps between plots
    f.subplots_adjust(wspace=0, hspace=0.1)
    pname='Figs/Emiss/MEGAN_monthly_daycycle.png'
    plt.suptitle('Diurnal isoprene emissions')
    plt.savefig(pname)
    print('SAVED FIGURE ',pname)
    plt.close()

# Delta ozone vs Delta emissions
def ozone_sensitivity(area_averaged=False):
    '''
        Regional regression between delta E and delta surface ozone
            Also same plot but for HCHO
    '''
    pnames = 'Figs/new_emiss/delta_regression_%s.png'
    if area_averaged:
        pnames = 'Figs/new_emiss/delta_avg_regression_%s.png'
    
    d0,d1=datetime(2005,1,1),datetime(2012,12,31)
    ekeys=['E_MEGAN','E_PP_lr']
    enew=E_new(d0,d1,dkeys=ekeys)
    dates=enew.dates
    months=util.list_months(d0,d1)
    apri=enew.E_MEGAN
    apost=enew.E_PP_lr
    # [days, 18,19]
    deltaE=apri-apost
    
    elats=enew.lats_lr
    elons=enew.lons_lr
    # apply oceanmask
    om = enew.oceanmask3d_lr
    deltaE[om] = np.NaN
    # change to monthly averages
    deltaE = util.monthly_averaged(dates, deltaE, keep_spatial=True)['mean']
    # subset just to summer
    summers=np.array([ i%12 in [0,1,11] for i in range(len(months)) ])
    deltaE = deltaE[summers,:,:]
    dEi, lats_regional,lons_regional = util.pull_out_subregions(deltaE,elats,elons,subregions=regions)
    if area_averaged:
        dEi = [ np.nanmean(dE,axis=(1,2)) for dE in dEi ]
        
    #pixm=util.monthly_average(enew.dates, enew.pixels_PP_lr, keep_spatial=True)['sum']
    
    # need surf ozone before, and after
        
    satkeys = ['IJ-AVG-$_ISOP',     # isop in ppbc?
               'IJ-AVG-$_CH2O',     # hcho in ppb
               'IJ-AVG-$_NO2',      # NO2 in ppb
               'IJ-AVG-$_NO',       # NO in ppb?
               'IJ-AVG-$_O3',       # O3 in ppb
               ] #+ GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    
    
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    o3      = tropchem_sat.O3[:,:,:,0] # surface only
    o3a     = new_sat.O3[:,:,:,0]
    hcho    = tropchem_sat.hcho[:,:,:,0]
    hchoa   = new_sat.hcho[:,:,:,0]
    nox     = tropchem_sat.NO[:,:,:,0] + tropchem_sat.NO2[:,:,:,0]
    noxa    = new_sat.NO[:,:,:,0] + new_sat.NO2[:,:,:,0]
    
    do3     = o3-o3a
    dhcho   = hcho-hchoa
    dnox    = nox-noxa
    darrs   = [do3, dhcho, dnox]
    stitles = ['($E_{GC} - E_{OMI}$) vs. (ozone - ozone$^{\\alpha}$)',
               '($E_{GC} - E_{OMI}$) vs. (HCHO - HCHO$^{\\alpha}$)',
               '($E_{GC} - E_{OMI}$) vs. (NO$_x$ - NO$_x$$^{\\alpha}$)']
    keys    = ['ozone', 'HCHO', 'NOx']
    yticks = {'ozone':None ,'HCHO':np.arange(0,1.6,0.5), 'NOx':None} 
    for darr, stitle, key in zip(darrs,stitles, keys):
        
        # subset to AUS and remove ocean
        subs = util.lat_lon_subset(lats,lons,pp.__AUSREGION__,data=[darr], has_time_dim=True)
        darr = subs['data'][0]
        darr[om] = np.NaN
        
        # monthly averaged
        darr = util.monthly_averaged(dates, darr, keep_spatial=True)['mean']
        
        # pull out summers
        darr = darr[summers,:,:]
        
        # sub regions
        darri, lats_regional, lons_regional = util.pull_out_subregions(darr,elats,elons,subregions=regions)
        
        
        f,axes=plt.subplots(n_regions,1,figsize=[10,12], sharex=True, sharey=True)
        
        print(key)
        print(' region, [ yearly slope, regression coeff, p value for slope non zero]')
        
        for i in range(n_regions):
        #for monthly_anomaly, monthly_data, color, label in zip(anomaly, monthly, colors, labels):
            color=colors[i]; 
            label=labels[i]
            
            # regionoal deltas to compare
            DE = dEi[i] # emissions
            DA = darri[i] # array from satellite overpass outputs
            if area_averaged:
                DA = np.nanmean(darri[i],axis=(1,2))
            # correlation: Y = m X + b
            m, b, r, cir, cijm = RMA(DE.flatten(), DA.flatten())
            
            print("%s &  [ m=%.2e,  r=%.3f ]   & \\"%(label, m, r))
            print("   CI for m = [%.1e,  %.1e]"%(cir[0][0], cir[0][1]))
            #ppb = m * C/cm2/s
            # for ppb=1: C/cm2/s = 1/m
            decrease = -1.0/m
            print("   decreasing emissions by %.1e C/cm2/s leads to 1 ppb decrease in surface ppb"%decrease )
            plt.sca(axes[i])
            nans= (np.isnan(DE) + np.isnan(DA))
            X,Y = DE[~nans], DA[~nans]
            plt.scatter(X,Y,color=color)
            pp.add_regression(X,Y)
            plt.legend(loc='best')
            
            # add region label to opposite side
            plt.twinx()
            plt.yticks([], [])
            plt.ylabel(label,color=color)
            
        plt.sca(axes[-1])
        plt.ylabel('$\Delta$'+key)
        if not area_averaged:
            if yticks[key] is not None:
                plt.yticks(yticks[key])
        plt.xlabel('$\Delta$E '+__apri__units__)
        plt.suptitle(stitle)
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
    #print("TREND: READING new_emissions")
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    #print("TREND: READING tropchem")
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    #print('TREND: GEOS-Chem satellite outputs read')
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
        # Get NOx instead of NO or NO2
        if origkey == 'IJ-AVG-$_NO2':
            continue
        if origkey == 'IJ-AVG-$_NO':
            new_surf = new_sat.NO2[:,:,:,0] + new_sat.NO[:,:,:,0]
            trop_surf = tropchem_sat.NO2[:,:,:,0] + tropchem_sat.NO[:,:,:,0]
            key='NO$_x$'
        
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
            print('region, [ yearly slope, regression coeff, p value for slope non zero]')
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
                
                print("%s &  [ m=%.2e,  r=%.3f, p=%.3f ]   & \\"%(label,12.0*m, r, p))
                
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
        if key=='NO$_x$':
            key='NOx'
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
        Compare apri and apost in each region
        # Grab all overpass data, resample to 3 monthly avg, then look at seasonal compariosn
        #
        # First plot: Time_series_emissions.png
            Row1: a priori 3m averaged time series
            Row2: a posteriori
            Row3: Diffs
            coloured by regionn, Aus region shaded for IQR
        # Second plot: RegSeas_emissions.png
            seasonal mean apri and apost for each region, as a bar chart
        
    '''
    pname = 'Figs/new_emiss/RegSeas_emissions.png'
    pname1 = 'Figs/Emiss/E_zones_diffs.png'
    
    d0=datetime(2005,1,1)
    d1=datetime(2005,12,31)
    
    #read satellite overpass outputs
    #DF = read_overpass_timeseries()
    #dates=[datetime.strptime(dstr, '%Y-%m-%d') for dstr in DF.index]
    enew=E_new(d0,d1)
    
    
    # Priori and posteriori overpass output
    # Time series
    apris=   [ DF['E_MEGAN_%s_mean'%reg] for reg in labels ]
    aposts=  [ DF['E_PP_lr_%s_mean'%reg] for reg in labels ]
    
    # Seasonal averages
    apri_seasonal = [ util.resample(apris[i],dates,"Q-NOV") for i in range(n_regions) ]
    apost_seasonal = [ util.resample(aposts[i],dates,"Q-NOV") for i in range(n_regions) ]
    seasons=apri_seasonal[0].mean().index.to_pydatetime()
    seasons = [season - timedelta(days=45) for season in seasons]
    years = util.list_years(datetime(2005,1,1), datetime(2013,1,1))
    ##
    ## FIRST FIGURE:
    f,axes = plt.subplots(3,1,figsize=[16,12], sharex=True,sharey=False)
    for i in range(n_regions):
        
        apri_mean = apri_seasonal[i].mean()
        apost_mean = apost_seasonal[i].mean()
        lw=[1,2][i==0]
        
        # first subplot is apriori
        plt.sca(axes[0])
        pp.plot_time_series(seasons,apri_mean, color=colors[i], label=labels[i], linewidth=lw)
        #if i==0:
        #    plt.fill_between()
        
        plt.sca(axes[1])
        pp.plot_time_series(seasons,apost_mean, color=colors[i], linewidth=lw)
        
        plt.sca(axes[2])
        pp.plot_time_series(seasons,apri_mean - apost_mean, color=colors[i], linewidth=lw)
        
    plt.xlabel('date', fontsize=24)
    plt.xticks(years)
    plt.title('%s - %s'%(__apri__,__apost__))
    plt.suptitle('Midday emissions',fontsize=30)
    #f.subplots_adjust(hspace=0)
    
    plt.sca(axes[0])
    plt.title(__apri__)
    axes[0].legend(fontsize=18, loc='upper right', bbox_to_anchor=(1, 1.05), ncol=3,)
    
    
    plt.sca(axes[1])
    plt.title(__apost__)
    
    for ax in axes:
        plt.sca(ax)
        plt.ylabel(__apri__units__)
        # Add grid for scale comparison
        for line in [[0,0], [3e12,3e12], [6e12,6e12], [9e12,9e12]]:
            plt.plot([datetime(2004,1,1),datetime(2020,1,1)], line, '--', color='k', alpha=0.2)
            
        plt.xlim([seasons[0]-timedelta(days=40), seasons[-1]+timedelta(days=40)])
    ## save figure
    plt.savefig(pname1)
    print('SAVED ',pname1)
    plt.close()
    
    ##
    ## SECOND FIGURE:
    f, axes = plt.subplots(n_regions,1,figsize=[16,12], sharex=True, sharey=True)
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
    plt.suptitle('Midday emissions [%s]'%__apri__units__,fontsize=30)
    f.subplots_adjust(hspace=0)


    ## save figure
    plt.savefig(pname)
    print('SAVED ',pname)
    plt.close()

def campaign_vs_GC(midday=True):
    '''
        Compare campaign data to Enew and Egc in that one grid square
    '''
    mh='_midday'
    pnamea="Figs/GC_VS_CAMPAIGNS%s.png"%['',mh][midday]
    # Wollongong/sydney grid square
    LatWol, LonWol = pp.__cities__['Wol']
    
    
    # Read campaign data
    mumba = campaign.mumba()
    sps1  = campaign.sps(1)
    sps2  = campaign.sps(2)
    
    # GEOS chem apri and apost keys and colours
    gckeys = ['IJ-AVG-$_ISOP',
              'IJ-AVG-$_CH2O',
              'IJ-AVG-$_O3']
    cpri,cpost = ['m','cyan']
    
    ## Plot 3 rows x 3 columns, columns are campaigns, rows are species 
    ##
    f, axes=plt.subplots(3,3,sharex='col',sharey='row')
    
    stitles=['SPS1 vs GEOS-Chem','SPS2 vs GEOS-chem','MUMBA vs GEOS-Chem']
    for j, (cdata, pname, stitle) in enumerate(zip([sps1,sps2,mumba],pnames,stitles)):
        # SPS has different set of dates for ozone
        cdates=cdata.dates
        odates=cdates
        if j<2:
            odates=cdata.odates
        d0,d1=util.first_day(odates[0]),util.last_day(odates[-1])
        
        # pull out ozone,hcho,isoprene
        cozone = cdata.ozone
        chcho  = cdata.hcho
        cisop  = cdata.isop
        
        if midday:
            odates, cozone  = cdata.get_daily_hour(key='ozone')
            cdates, chcho   = cdata.get_daily_hour(key='hcho')
            _, cisop        = cdata.get_daily_hour(key='isop')
        
        # Pull out satellite overpass data for comparison
        trop = GC_class.GC_sat(d0,d1, keys=gckeys)
        tropa= GC_class.GC_sat(d0,d1, keys=gckeys, run='new_emissions')
        dates0 = trop.dates
        dates  = [d+timedelta(hours=13.5) for d in dates0] 
        
        # grab wollongong square
        Woli, Wolj  = util.lat_lon_index(LatWol,LonWol,trop.lats,trop.lons) # lat, lon indices
        ozone       = trop.O3[:,Woli,Wolj,0] # surface o3 ppb
        isop        = 0.2*trop.isop[:,Woli,Wolj,0] # surface isop ppbC *0.2 for ppb isoprene
        hcho        = trop.hcho[:,Woli,Wolj,0] # surface hcho ppb
        ozonea      = tropa.O3[:,Woli,Wolj,0] # surface o3 ppb
        isopa       = 0.2*tropa.isop[:,Woli,Wolj,0] # surface isop ppbC *0.2 for ppb isoprene
        hchoa       = tropa.hcho[:,Woli,Wolj,0] # surface hcho ppb
        
        # for each tracer plot time series comparison
        for i, (meas, gc, gca, spsdates, title) in enumerate(zip([cisop,chcho,cozone],[isop,hcho,ozone],[isopa,hchoa,ozonea],[cdates,cdates,odates],['Isoprene','HCHO','Ozone'])):
            plt.sca(axes[i,j])
            pp.plot_time_series(spsdates,meas,color='k',marker='+',label='measurement')
            pp.plot_time_series(dates,gc,color=cpri, marker='^', label='a priori')
            pp.plot_time_series(dates,gca,color=cpost,marker='x', label='a posteriori')
            
            # Hide the right and top spines
            axes[i,j].spines['right'].set_visible(False)
            axes[i,j].spines['top'].set_visible(False)
            
            if i==0:
                plt.legend(loc='best')
                plt.title(stitle)
            if j==0:
                plt.ylabel('%s [ppb]'%title)
                
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(pnamea)
    plt.close()
    print("SAVED: ",pnamea)

def FTIR_Comparison():
    '''
        Time series and seasonal mean and iqr of deconvolved HCHO total columns
        both a priori and posteriori plotted next to FTIR
    '''
    
    pname_TC_series='Figs/FTIR_TC_Comparison_series.png'
    pname_mya='Figs/FTIR_TC_Comparison_MYA.png'
    
    
    # FTIR Comparison
    LatWol, LonWol = pp.__cities__['Wol']
    
    # Read FTIR output
    ftir=campaign.Wgong()
    
    # Resample FTIR to just midday averages
    middatas=ftir.resample_middays()
    
    print("FTIR data shapes: ",np.shape(ftir.VMR),np.shape(ftir.dates))
    print("FTIR midday data shapes: ", np.shape(middatas['VMR']),np.shape(middatas['dates']), np.shape(ftir.dates))
    #plt.plot(middatas['DOF']) # DOFs range from 1.3 in summer to 1.7 in winters
    
    
    d0,d1=datetime(2007,7,1), datetime(2012,12,31)
    
    trop = GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__)
    tropa= GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__, run='new_emissions')
    # make sure pedges and pmids are created
    trop.add_pedges()
    dates=trop.dates
    
    # colours for trop and tropa
    c = 'r'
    ca= 'm'
    
    # grab wollongong square
    Woli, Wolj = util.lat_lon_index(LatWol,LonWol,trop.lats,trop.lons) # lat, lon indices
    GC_VMR  = trop.hcho[:,Woli,Wolj,:]
    GCa_VMR = tropa.hcho[:,Woli,Wolj,:]
    p       = trop.pmids[:,Woli,Wolj,:]
    #print("FTIR shapes before deconv:", np.shape(middatas['VMR']),np.shape(middatas['dates']), np.shape(ftir.dates))
    decon   = ftir.Deconvolve(GC_VMR, dates,p, checkname='check_interp.png')
    decona  = ftir.Deconvolve(GCa_VMR, dates,p,checkname='check_interp_a.png')
    
    # need delta pressure for TC conversion:
    #data={ k:decon[k] for k in ['new_TC', 'orig_TC', 'TC_ret']}
    data= {'apri':decon['new_TC'], 'apost':decona['new_TC'], 'FTIR':decon['TC_ret']}
    TC_df = pd.DataFrame(data, index=decon['dates'])
    
    # drop the NaNs and plot 
    TC_df.plot(linestyle='',marker='+')
    #plt.show()
    
    plt.savefig(pname_TC_series)
    plt.close()
    print("Saved ",pname_TC_series)
    
    
    # Annual cycle of FTIR, deconvolved a priori, and deconvolved a posteriori
    plt.close()
    TC_ftir = decon['TC_ret']
    TC_apri = decon['new_TC']
    TC_apost= decona['new_TC']
    tc_colors = ['k', 'r', 'teal']
    tc_labels = ['$\Omega_{FTIR}$','$\Omega_{GC}$', '$\Omega_{GC}^{\\alpha}$']
    for i, v in enumerate([TC_ftir,TC_apri,TC_apost]):
        mya = util.multi_year_average(v, decon['dates'])
        myav = np.squeeze(mya.mean().values)
        uq = np.squeeze(mya.quantile(.75).values)
        lq = np.squeeze(mya.quantile(.25).values)
        tcc = tc_colors[i]
        plt.fill_between(np.arange(12), lq, uq, color=tcc, alpha=0.35)
        plt.plot(np.arange(12),myav, label=tc_labels[i], linewidth=2, color=tcc)
    plt.legend(fontsize=22)
    plt.ylabel('$\Omega$ [molec cm$^{-2}$]',fontsize=22)
    plt.xlabel('month',fontsize=22)
    plt.title('Multiyear monthly mean total columns HCHO ($\Omega$)',fontsize=25)
    plt.xlim([-0.5, 11.5])
    plt.xticks(range(12),['J','F','M','A','M','J','J','A','S','O','N','D'])
    plt.savefig(pname_mya)
    plt.close()
    print("Saved ",pname_mya)



################
### UNCERTAINTY
################

def uncertainty_time_series(d1=datetime(2012,12,31)):
    '''
        Uncertainty time series plotting
        TODO: Time series plots updated, other plots moved elsewhere
    '''
    
    # Read product with pixels and uncertainty
    # OMHCHORP: 
    #   Need: VC_OMI, col_uncertainty_OMI, entries
    # E_new: 
    #   Need: E_PP_lr, ModelSlope, 
    d0=datetime(2005,1,1)
    d1=datetime(2012,12,31)
    
    # Uncertainty (portional for VC, % for slope)
    uncertkeys = ['VCC_rerr_lr', 'slope_rerr_lr', 'E_PP_err_lr']#,'E_PPm_err_lr']
    
    labels      = {'VCC_rerr_lr':__Oomi__, 'slope_rerr_lr':'S', 'E_PP_err_lr':__apost__}
    colours     = {'VCC_rerr_lr':'orange', 'slope_rerr_lr':'m', 'E_PP_err_lr':'k'}
    otherkeys   = ['E_PP_lr', 'E_PPm_err_lr', 'pixels_lr']
    
    
    # Read from E_new
    enew    = E_new(d0,d1, dkeys=uncertkeys+otherkeys)
    dates   = np.array(enew.dates)
    months  = np.array(util.list_months(dates[0],dates[-1]))
    pix     = enew.pixels_lr
    lats    = enew.lats_lr
    lons    = enew.lons_lr
    highlim = 2.0
    
    # remove ocean squares
    oceanmask=util.oceanmask(lats,lons)
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],len(dates),axis=0)
    oceanmask3dm= np.repeat(oceanmask[np.newaxis,:,:],len(months),axis=0) 
    for key in ['VCC_rerr_lr','E_PP_err_lr','E_PP_lr']:
        data=getattr(enew,key) 
        data[oceanmask3d] = np.NaN
        setattr(enew,key,data)
    # some are monthly
    for key in ['slope_rerr_lr', 'E_PPm_err_lr']:
        data=getattr(enew,key)
        data[oceanmask3dm] = np.NaN
        setattr(enew,key,data)
    
    Eppm = util.monthly_averaged(dates,enew.E_PP_err_lr,keep_spatial=True)['mean']
    Em = Eppm/enew.E_PPm_err_lr    
    
    # Relative uncertainty time series
    for key in uncertkeys:
        # pull out regions:
        data=getattr(enew,key)
        print(key, data.shape)
        label=labels[key]
        if key == 'E_PP_err_lr':
            data = data/enew.E_PP_lr # error into relative error
            data[enew.E_PP_lr == 0] = np.NaN
        
        # mean over space
        data = np.nanmean(data,axis=(1,2))
        toohigh = data > highlim
        n_toohigh = np.sum(toohigh)
        label = label + ' (%d entries > %d)'%(n_toohigh,highlim)
        
        # some are monthly
        if key in ['slope_rerr_lr']:
            pp.plot_time_series(months, data, label=label, color=colours[key], linestyle='--', marker='+' ,markersize=10)
            #pp.plot_time_series(months[toohigh], np.ones(np.sum(toohigh)), color=colours[key], linestyle='', marker='^' )
        else:
            pp.plot_time_series(dates, data, label=label, color=colours[key], linestyle='', marker='+' )
            #pp.plot_time_series(dates[toohigh], np.ones(np.sum(toohigh)), color=colours[key], linestyle='', marker='^' )
        
    # finally add monthly E relative error
    Emser = np.nanmean(Em,axis=(1,2))
    n_toohigh = np.sum(Emser>highlim)
    pp.plot_time_series(months, Emser, label='monthly '+__apost__+ ' (%d entries > %d)'%(n_toohigh,highlim), 
                        linestyle='--',color='blue',marker='+',markersize=10)
    
    plt.ylim([0,highlim+0.05*highlim])
    plt.title('Relative uncertainty')
    plt.legend()
    pname1='Figs/rerr_summary.png'
    plt.savefig(pname1)
    print("Saved ",pname1)
    plt.close()
    
    # Also add a pixel count summary (multiyear-avg)
    pix[oceanmask3d]=np.NaN
    PlotMultiyear(pix, dates,lats,lons,weekly=True)
    plt.suptitle('Mean pixel count per 2$^{\circ}$x2.5$^{\circ}$ land grid square')
    pname2='Figs/pix_mya.png'
    plt.savefig(pname2)
    print("Saved ",pname2)
    plt.close()
    
    
    # Now plot maps for summer/winter mean uncertainties over Australia
    f=plt.figure()
    vmin=0
    vmax=2
    
    E=enew.E_PP_err_lr / enew.E_PP_lr
    
    S=enew.slope_rerr_lr
    O=enew.VCC_rerr_lr
    
    # pull out summer, winter means
    
    
    
    # plot each thing
    plt.subplot(3,2,1)
    pp.createmap(np.nanmean(E,axis=0), lats, lons, aus=True, linear=True,
                 vmin=vmin,vmax=vmax,
                 title='Summer')
    plt.ylabel(__apost__)
    plt.subplot(3,2,2)
    pp.createmap(np.nanmean(E,axis=0), lats, lons, aus=True, linear=True,
                 title='Winter')
    
    plt.subplot(3,2,3)
    pp.createmap(np.nanmean(S,axis=0), lats, lons, aus=True, linear=True,
                 vmin=vmin,vmax=vmax)
    plt.ylabel('Slope')
    plt.subplot(3,2,4)
    pp.createmap(np.nanmean(S,axis=0), lats, lons, aus=True, linear=True,
                 vmin=vmin,vmax=vmax)
    
    plt.subplot(3,2,5)
    pp.createmap(np.nanmean(O,axis=0), lats, lons, aus=True, linear=True,
                 vmin=vmin,vmax=vmax)
    plt.ylabel(__Oomi__)
    plt.subplot(3,2,6)
    pp.createmap(np.nanmean(O,axis=0), lats, lons, aus=True, linear=True,
                 title='Winter')
    
    pname='test_uncert.png'
    plt.savefig(pname)
    plt.close(pname)

def pixel_counts_summary():
    
    #dstr = d0.strftime("%Y%m%d")
    pname2 = 'Figs/pixel_count_barchart.png'
    pname3 = 'Figs/pixel_count_barchart_unfiltered.png'
    
    #read satellite overpass outputs
    DF = read_overpass_timeseries()
    dates = [datetime.strptime(dstr, '%Y-%m-%d') for dstr in DF.index]
    
    key = 'pixels_PP_lr'
    keyf= 'pixels_PP'
    # AT SOME POINT THIS GOT BUGGED!
    keyu= 'pixels_PP_u'
    suptitle='Mean pixel count per grid square per day'
    suptitle2='Mean total-pixel-count per season per region'
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
        plt.ylabel(labels[i], color=colors[i], fontsize=24)
        
        if i==0:
            plt.legend(loc='best', fontsize=16)
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

def relative_error_summary():
    '''
        Plots: MYA seasonal relative error in E, Omega, and S
        Plots2: summer vs winter relative error in E
        Prints: summer/winter mean and IQR of regionally averaged rerrs
    '''
    
    d0,dN = datetime(2005,1,1),datetime(2012,12,31)
    dkeys=['E_PP_lr', 'pixels_PP_lr', # emissiosn and pixel counts
           'E_PP_err_lr','E_PPm_err_lr','E_PPm_rerr_lr', # Error in emissions estimate
           'SC_err_lr', 
           'VCC_PP_lr', # Omega 
           'VCC_err_lr','VCC_rerr_lr', # daily OmegaPP error in
           'slope_rerr_lr'] # monthly portional error in slope

    # READ EMISSIONS AND ERROR
    enew=E_new(d0,dN,dkeys=dkeys)
    dates=enew.dates
    months=util.list_months(d0,dN)
    lats,lons=enew.lats_lr, enew.lons_lr
    pix=enew.pixels_PP_lr
    # GET MONTHLY TOTAL PIXELS
    pixm=util.monthly_averaged(dates,pix,keep_spatial=True)['sum']

    for key in dkeys:
        print(key, getattr(enew,key).shape)

    # MASK OCEANS, 
    E = enew.E_PP_lr
    E[enew.oceanmask3d_lr] = np.NaN
    Em      = util.monthly_averaged(dates,E,keep_spatial=True)['mean']
    Eerr   = enew.E_PP_err_lr
    Eerrm  = enew.E_PPm_err_lr
    Ererr  = Eerr/E
    Ererr[~np.isfinite(Ererr)] = np.NaN
    Ererr[np.isclose(E,0.0)] = np.NaN
    Ererrm = Eerrm/Em
    Ererrm[~np.isfinite(Ererrm)] = np.NaN
    Ererrm[np.isclose(Em,0.0)] = np.NaN

    Srerrm  = enew.slope_rerr_lr

    # monthly VCC error:from per pixel error divided by pixels in the month
    O   = enew.VCC_PP_lr
    Om  = util.monthly_averaged(dates,O,keep_spatial=True)['mean']
    Oerr  = enew.VCC_err_lr * np.sqrt(pix) # error has already been divided by sqrt daily pix
    Oerrm = util.monthly_averaged(dates,Oerr,keep_spatial=True)['mean'] /  np.sqrt(pixm)
    # Same as Enew monthly error:replace error with NaN and set relative error to 100%
    # Enew monthly negatives are replaced with zeros, but not VCCm 
    Orerrm = Oerrm / Om
    negerr = (Om < 0)+(Oerrm<0)
    Orerrm[negerr] = 1.0
    

    # 3d monthly oceanmask:
    oceanmask=np.repeat(enew.oceanmask_lr[np.newaxis,:,:], len(months), axis=0)
    print("Checking Oerr")
    # Definitely includes ocean squares
    print(np.nanmean(Orerrm), np.nanmean(Orerrm[oceanmask]))
    Orerrm[oceanmask] = np.NaN
    print("Checking Serr")
    # also
    print(np.nanmean(Srerrm), np.nanmean(Srerrm[oceanmask]))
    Srerrm[oceanmask] = np.NaN
    print("Checking Eerr")
    # does not seem to have ocean squares (good)
    print(np.nanmean(Ererrm), np.nanmean(Ererrm[oceanmask]))
    Ererrm[oceanmask] = np.NaN


    # first lets do A seasonal plot of relative monthly error
    arrs=[Ererrm, Srerrm, Orerrm]
    titles = ['relative error in monthly a posteriori', 
              'relative error in monthly slope',
              'relative error in monthly $\Omega$']
    pnames = ['Figs/mya_Ererr.png',
                'Figs/mya_Srerr.png',
                'Figs/mya_Orerr.png']
    ylims = [ [0,1.5], [0.2, 0.5], [0,3]]
    
    # First do Srerr
    plt.figure(figsize=[10,14])
    ylim,pname=ylims[1],pnames[1]
    Sret,x,ax = PlotMultiyear(Srerrm,months,lats,lons,weekly=False,
                                     median=True,ylims=ylim)
    plt.suptitle(titles[1],fontsize=20)
    plt.yticks(np.linspace(ylim[0], ylim[1], 4))
    plt.savefig(pname)
    print("SAVED ",pname)
    plt.close()
    
    # Now do Omega rerr
    plt.figure(figsize=[10,14])
    ylim,pname=ylims[2],pnames[2]
    Oret,x,ax = PlotMultiyear(Orerrm,months,lats,lons,weekly=False,
                                     median=True,ylims=ylim)
    plt.suptitle(titles[2],fontsize=20)
    plt.yticks(np.linspace(ylim[0], ylim[1], 4))
    plt.savefig(pname)
    print("SAVED ",pname)
    plt.close()
    
    # Finally do the a postiori summary error
    plt.figure(figsize=[10,14])
    ylim,pname=ylims[0],pnames[0]
    label="$\Delta$%s/%s"%(__apost__,__apost__)
    Eret,x,axes = PlotMultiyear(Ererrm,months,lats,lons,weekly=False,
                                     median=True,ylims=ylim,label=label)
    plt.suptitle(titles[0],fontsize=20)
    plt.yticks(np.linspace(ylim[0], ylim[1], 4))
    for i,ax in enumerate(axes):
        plt.sca(ax)
        plt.plot(x,Sret[i], '--',label='$\Delta$S/S')
        plt.plot(x,Oret[i], ':',label='$\Delta \Omega / \Omega$')
        if i==0:
            plt.legend(loc='best')
    # will add legend in paint
    plt.savefig(pname)
    print("SAVED ",pname)
    plt.close()
    
    ###############################
    # Summer Winter Plots##########
    mya = util.multi_year_average_spatial(E, dates)
    myam = util.multi_year_average_spatial(Ererrm, months)


    summer=np.array([0,1,11])
    winter=np.array([5,6,7])
    summersm = np.nanmean(myam['mean'][summer,:,:],axis=0)
    wintersm = np.nanmean(myam['mean'][winter,:,:],axis=0)
    Esummer = np.nanmean(mya['mean'][summer,:,:],axis=0)
    Ewinter = np.nanmean(mya['mean'][winter,:,:],axis=0)


    plt.close()

    # Plot summer,winter maps of monthly rerr
    plt.figure(figsize=[14,14])
    vmin,vmax=1e11,1e13
    linear=False

    ccm2s='C cm$^{-2}$ s$^{-1}$'
    plt.subplot(2,2,1)
    pp.createmap(Esummer,lats,lons,aus=True,
                 linear=linear, vmin=vmin,vmax=vmax,
                 clabel=ccm2s,
                 title='mean a posteriori')
    plt.ylabel("Summer")
    plt.subplot(2,2,2)
    pp.createmap(summersm,lats,lons,aus=True,
                 linear=True, vmin=0,vmax=1,
                 clabel='Portional',
                 title='relative a posteriori error')
    plt.ylabel("Summer")

    plt.subplot(2,2,3)
    pp.createmap(Ewinter, lats,lons, aus=True,
                 linear=linear, vmin=vmin,vmax=vmax,
                 clabel=ccm2s,
                 title='mean a posteriori')
    plt.ylabel("Winter")
    plt.subplot(2,2,4)
    pp.createmap(wintersm,lats,lons,aus=True,
                 linear=True, vmin=0,vmax=1, 
                 clabel='Portional',
                 title='relative a posteriori error',
                 pname='Figs/Ererr_map_summerwinter.png')
    plt.ylabel("Winter")
    

if __name__ == "__main__":
    
    start=timeit.default_timer()
    
    # set up plotting parameters like font sizes etc..
    fullpageFigure()
    
    # Run time series creation
    
    #save_overpass_timeseries()
    
    ## METHOD PLOTS
    
    #check_modelled_background() # finished? 24/2/19
    #Examine_Model_Slope() # finished ~ 20/2/19
    
    ## Results Plots
    
    # print out seasonal regional means and STDs
    #compare_model_outputs()
    
    #  trend analysis plots, printing slopes for tabularisation
    #trend_analysis()
    #seasonal_differences()
    #[ozone_sensitivity(aa) for aa in [True, False] ]
    
    ## CAMPAIGN COMPARISONS
    # time series mumba,sps1,sps2
    [campaign_vs_GC(flag) for flag in [True,False]]
    # FTIR comparison
    #FTIR_Comparison()
    
    
    # Day cycle for each month compared to sin wave from postiori
    #Seasonal_daycycle() # done with updated suptitle 4/4/19
    
    
    # TODO: time series compared to satellite HCHO
    # TODO: Seasonal regional multiyear comparison
    #regional_seasonal_comparison()
    
    #time_series()
    # TODO: 
    
    ## UNCERTAINTY
    #TODO: implement
    #uncertainty_time_series()
    # TODO: add sums to analysis TS
    # todo: discuss plot output from 
    #pixel_counts_summary()
    ## summarised uncertainty
    #relative_error_summary()

    ### Record and time STUJFFS
    
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for stuff"%((end-start)/60.0))
    

