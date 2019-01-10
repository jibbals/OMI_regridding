#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:33:05 2018

@author: jesse
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import timeit

import new_emissions
from utilities import utilities as util
from classes.E_new import E_new
from classes import GC_class
from utilities import GC_fio

# plotting
from utilities import plotting as pp
import matplotlib.pyplot as plt

####
## GLOBALS
####

__VERBOSE__=True

####
## TESTS
####

def alpha_creation():
    '''
        Create isoprene scaling factors monthly over Australia
          using difference from top-down and MEGAN emissions at midday

        Create yearly plots and compare with multi-year avg version
    '''
    label_meg='$E_{GC}$'
    label_topd='$E_{OMI}$'

    yearly_alphas=[]
    yearly_megan=[]
    yearly_topd=[]
    # first loop over year and create yearly alpha plots
    for year in range(2005,2013):
        dat = new_emissions.calculate_alpha(year,mya=False)
        topd=dat['Enew']
        megan=dat['Emeg']
        alpha=dat['alpha']
        yearly_alphas.append(alpha)
        yearly_topd.append(dat['Enewm'])
        yearly_megan.append(dat['Emegm'])
        lats=dat['lats']
        lons=dat['lons']
        dates=dat['dates']
        months=dat['months']
        months=[m+timedelta(days=15) for m in months]
        region=[-50,105,-7,155]
        vmin,vmax = 0, 2
        sydlat,sydlon = pp.__cities__['Syd']

        lati,loni = util.lat_lon_index(sydlat,sydlon,lats,lons)

        plt.figure(figsize=[15,13])
        # first plot alpha in jan, then alpha in
        plt.subplot(221)
        pp.createmap(alpha[0],lats, lons, linear=True, region=region, title='Jan. alpha',vmin=vmin,vmax=vmax)
        # then plot alpha in June
        plt.subplot(222)
        pp.createmap(alpha[6],lats, lons, linear=True, region=region, title='Jul alpha',vmin=vmin,vmax=vmax)
        #finally plot time series at sydney of alpha, megan, and topdown emissions
        plt.subplot(212)
        plt.title('examine Sydney')
        plt.plot_date(dates, megan[:,lati,loni], 'm-', label=label_meg)
        plt.plot_date(dates, topd[:,lati,loni], '-', label=label_topd, color='cyan')
        plt.ylim(1e11,2e13)
        plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]')
        plt.legend()
        plt.sca(plt.twinx())
        plt.plot_date(months, alpha[:,lati,loni], 'k-', linewidth=3, label='alpha')
        plt.ylim(vmin,vmax)
        plt.ylabel('Alpha')
        plt.suptitle('Alpha for %4d'%year)
        plt.savefig('Figs/new_emiss/alpha_%4d.png'%year)
        plt.close()

    allalpha = np.concatenate([yearly_alphas[i] for i in range(8)], axis=0)
    allmegan = np.concatenate([yearly_megan[i] for i in range(8)], axis=0)
    alltopd  = np.concatenate([yearly_topd[i] for i in range(8)], axis=0)

    # finally create/plot mya alpha
    dat = new_emissions.calculate_alpha(mya=True)
    topd=dat['Enewm']
    megan=dat['Emegm']
    alpha=dat['alpha']
    lats=dat['lats']
    lons=dat['lons']
    dates=dat['dates']
    months=dat['months']
    months=[m+timedelta(days=15) for m in months]
    allmonths=util.list_months(dates[0],dates[-1])
    print(np.shape(allmonths), np.shape(allalpha), np.shape(megan))

    region=[-45,110,-10,155]
    vmin,vmax = 0, 2
    sydlat,sydlon = pp.__cities__['Syd']

    lati,loni = util.lat_lon_index(sydlat,sydlon,lats,lons)

    plt.figure(figsize=[15,13])
    # first plot alpha in jan, then alpha in
    plt.subplot(321)
    fixed=np.copy(alpha)
    fixed[fixed == 1] =np.NaN
    pp.createmap(fixed[0],lats, lons, linear=True, region=region, title='January alpha',vmin=vmin,vmax=vmax)
    # then plot alpha in June
    plt.subplot(322)
    pp.createmap(fixed[6],lats, lons, linear=True, region=region, title='July alpha',vmin=vmin,vmax=vmax)
    #finally plot time series at sydney of alpha, megan, and topdown emissions
    plt.subplot(312)
    X=range(12)
    plt.plot(X, megan[:,lati,loni], 'm-', label=label_meg)
    plt.plot(X, topd[:,lati,loni], '-', label=label_topd, color='cyan')
    plt.ylim(1e11,2e13)
    plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]')
    plt.legend()
    plt.title('examine Sydney')
    plt.sca(plt.twinx())
    plt.plot(X, alpha[:,lati,loni], 'k-', linewidth=3, label='alpha')
    plt.plot([X[0],X[-1]], [1,1], 'k--', linewidth=1) # dotted line
    plt.xlim(-0.5,11.5)
    plt.xticks(X)
    plt.gca().set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

    plt.ylim(vmin,vmax)
    plt.ylabel('Alpha')

    plt.subplot(313)
    plt.plot_date(allmonths, allmegan[:,lati,loni], 'm-', label=label_meg)
    plt.plot_date(allmonths, alltopd[:,lati,loni], '-', label=label_topd, color='cyan')
    plt.ylim(1e11,2e13)
    plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]')
    plt.legend()
    plt.title('monthly average')
    plt.sca(plt.twinx())
    plt.plot_date(allmonths,allalpha[:,lati,loni], 'k-', linewidth=3)
    plt.plot_date([allmonths[0],allmonths[-1]], [1,1], 'k--',linewidth=1) # dotted line
    plt.ylabel('alpha')
    plt.ylim(vmin,vmax)

    plt.suptitle('Alpha for multi-year average 2005-2012')
    plt.savefig('Figs/new_emiss/alpha_mya.png')
    plt.close()

def check_old_vs_new_emission_diags(month=datetime(2005,1,1)):
    # compare new_emissions hemco output
    d0=datetime(month.year,month.month,1)
    d1=util.last_day(d0)

    # old emissions are hourly
    old, oldattrs = GC_fio.read_Hemco_diags(d0,d1, new_emissions=False)
    # new emissions are daily
    new, newattrs = GC_fio.read_Hemco_diags(d0,d1, new_emissions=True)

    lats=old['lat']
    lons=old['lon']

    old_isop = np.mean(old['ISOP_BIOG'],axis=0) # hourly -> 1 month avg
    new_isop = np.mean(new['ISOP_BIOG'],axis=0) # daily -> 1 month avg

    pname=month.strftime('Figs/new_emiss/check_old_new_emission_diags_%Y%m.png')
    pp.compare_maps([old_isop,new_isop], [lats,lats],[lons,lons],linear=True,
                    rmin=-400,rmax=400, vmin=0,vmax=0.8e-9, titles=['old','new'],
                    region=pp.__GLOBALREGION__, pname=pname)

def isop_biog_time_series(d0=datetime(2005,1,1), d1=None):

    # Read whole year and do time series for both
    old, oldattrs = GC_fio.read_Hemco_diags(d0,None, new_emissions=False)
    new, newattrs = GC_fio.read_Hemco_diags(d0,None, new_emissions=True)
    old_isop = old['ISOP_BIOG']
    new_isop = new['ISOP_BIOG']
    old_dates= old['dates']
    new_dates= new['dates']
    lats,lons= new['lats'],new['lons']

    # remove last index from old data, and old dates
    lastindex=len(old_dates)-1
    old_dates = np.delete(old_dates,[lastindex])
    old_isop  = np.delete(old_isop, [lastindex],axis=0)

    # pull out regions and compare time series
    r_olds, r_lats, r_lons = util.pull_out_subregions(old_isop,
                                                     lats, lons,
                                                     subregions=pp.__subregions__)
    r_news, r_lats, r_lons = util.pull_out_subregions(new_isop,
                                                     lats, lons,
                                                     subregions=pp.__subregions__)
    f,axes = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    for i, [label, color] in enumerate(zip(pp.__subregions_labels__, pp.__subregions_colors__)):
        # set current axis
        plt.sca(axes[i])

        # plot time series for each subregion
        r_old_hourly = np.nanmean(r_olds[i],axis=(1,2)) # hourly regional avg
        # change hourly into daily time series
        r_old = np.array(pd.Series(r_old_hourly,index=old_dates).resample('D').mean())
        r_new = np.nanmean(r_news[i],axis=(1,2)) # daily

        pp.plot_time_series(new_dates,r_old, label='a priori', linestyle='--', color=color, linewidth=2)
        pp.plot_time_series(new_dates,r_new, label='a posteriori', linestyle='-', color=color, linewidth=2)
        plt.title(label,fontsize=20, color=color)
        if i==0:
            plt.ylabel('kgC m$^{-2}$ s$^{-1}$')
            plt.legend(loc='best')

    pname = 'Figs/new_emiss/E_isop_series_2005.png'
    plt.ylabel('kgC m$^{-2}$ s$^{-1}$')
    plt.suptitle('Daily mean biogenic isoprene emissions',fontsize=26)

    plt.savefig(pname)
    print('SAVED FIGURE ',pname)
    plt.close(f)

def hcho_ozone_timeseries(d0,d1):
    '''
    '''
    suptitle_prefix='Daily'
    dstr = d0.strftime("%Y%m%d_") + d1.strftime("%Y%m%d")
    pname1 = 'Figs/new_emiss/HCHO_total_columns_%s.png'%dstr
    pname2 = 'Figs/new_emiss/O3_trop_columns_%s.png'%dstr
    
    satkeys = ['IJ-AVG-$_ISOP', 'IJ-AVG-$_CH2O', 'IJ-AVG-$_NO2',     # NO2 in ppbv
               'IJ-AVG-$_O3', ] + GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('GEOS-Chem satellite outputs read 2005')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)
    
    new_sat_tc  = new_sat.get_total_columns(keys=['hcho'])
    tropchem_sat_tc  = tropchem_sat.get_total_columns(keys=['hcho'])
    # TOTAL column HCHO
    new_hcho_tc = new_sat_tc['hcho']
    tropchem_hcho_tc = tropchem_sat_tc['hcho']
    
    new_sat_tropc  = new_sat.get_trop_columns(keys=['O3'])
    tropchem_sat_tropc  = tropchem_sat.get_trop_columns(keys=['O3'])
    # trop column O3
    new_o3_tropc = new_sat_tropc['O3']
    tropchem_o3_tropc = tropchem_sat_tropc['O3']
    
    
    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates

    ## read old satellite hcho columns...
    # OMI total columns, PP corrected total columns
    Enew = E_new(d0, d1, dkeys=['VCC_OMI','VCC_PP'])
    
    # grab total columns
    vcc_omi     = Enew.VCC_OMI
    vcc_pp      = Enew.VCC_PP
    lats2, lons2= Enew.lats, Enew.lons
    # Enew lats,lons are in high resolution

    # pull out regions and compare time series
    new_sat_o3s, r_lats, r_lons = util.pull_out_subregions(new_o3_tropc,
                                                           lats, lons,
                                                           subregions=pp.__subregions__)
    tropchem_sat_o3s, r_lats, r_lons = util.pull_out_subregions(tropchem_o3_tropc,
                                                                lats, lons,
                                                                subregions=pp.__subregions__)

    new_sat_hchos, r_lats, r_lons = util.pull_out_subregions(new_hcho_tc,
                                                         lats, lons,
                                                         subregions=pp.__subregions__)
    
    tropchem_sat_hchos, r_lats, r_lons = util.pull_out_subregions(tropchem_hcho_tc,
                                                         lats, lons,
                                                         subregions=pp.__subregions__)
    
    vcc_omis, r_lats2, r_lons2 = util.pull_out_subregions(vcc_omi,
                                                         lats2, lons2,
                                                         subregions=pp.__subregions__)

    vcc_pps, r_lats3, r_lons3 = util.pull_out_subregions(vcc_pp,
                                                         lats2, lons2,
                                                         subregions=pp.__subregions__)

    # for plotting we may want daily, weekly, or monthly averages
    def baseresample(datain, bins='M'):
        return pd.Series(datain, index=dates).resample(bins)

    # by default do nothing
    resample = lambda datain : datain
    newdates = dates
    suptitle_prefix = 'Daily'

    if (d1-d0).days > 100: # after 100 days switch to weekly averages
        suptitle_prefix='Weekly'
        bins='7D'
        if (d1-d0).days > 500: # use monthly for > 500 days
            suptitle_prefix='Monthly'
            bins='M'
        resample = lambda datain: np.array(baseresample(datain, bins).mean())
        newdates = baseresample(np.arange(len(dates)), bins).mean().index.to_pydatetime()


    # will be printing mean difference between estimates
    print('area,   new_emiss hcho,   tropchem hcho,   OMI hcho,       OMI$_{PP}$ hcho')


    f1,axes1 = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    f2,axes2 = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    for i, [label, color] in enumerate(zip(pp.__subregions_labels__, pp.__subregions_colors__)):
        
        # plot time series for each subregion
        hcho_new_emiss = np.nanmean(new_sat_hchos[i], axis=(1,2)) # daily
        hcho_tropchem = np.nanmean(tropchem_sat_hchos[i],axis=(1,2))
        o3_new_emiss = np.nanmean(new_sat_o3s[i], axis=(1,2))
        o3_tropchem = np.nanmean(tropchem_sat_o3s[i], axis=(1,2))
        
        
        # change hourly into daily time series
        #r_old = np.array(pd.Series(r_old_hourly,index=old_dates).resample('D').mean())
        hcho_omi = np.nanmean(vcc_omis[i],axis=(1,2)) # daily
        hcho_pp  = np.nanmean(vcc_pps[i],axis=(1,2)) # daily
        
        
        # resample daily into something else:
        hcho_new_emiss = resample(hcho_new_emiss)
        hcho_tropchem = resample(hcho_tropchem)
        hcho_omi = resample(hcho_omi)
        hcho_pp = resample(hcho_pp)
        o3_new_emiss = resample(o3_new_emiss)
        o3_tropchem = resample(o3_tropchem)
        
        arr = np.array([np.nanmean(hcho_new_emiss), np.nanmean(hcho_tropchem), np.nanmean(hcho_omi), np.nanmean(hcho_pp)])
        print(label, arr[0], arr[1], arr[2], arr[3])
        print('   ,', 100*(arr - arr[2])/arr[2]  ) # difference from OMI orig
        print('   ,', 100*(arr - arr[3])/arr[3]  ) # difference from OMI PP
        
        # Fig1: HCHO time series
        plt.sca(axes1[i])
        pp.plot_time_series(newdates,hcho_new_emiss, label='new_emiss run', linestyle='-.', color=color, linewidth=2)
        pp.plot_time_series(newdates,hcho_tropchem, label='tropchem run', linestyle='--', color=color, linewidth=2)
        pp.plot_time_series(newdates,hcho_omi, dfmt='%Y%m%d', label='OMI', linestyle='-', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_pp, label='OMI recalculated', linestyle=':', color=color, linewidth=2)
        plt.title(label,fontsize=20)
        if i==0:
            plt.ylabel('HCHO cm$^{-2}$')
            plt.legend(loc='best')
        
        # Fig2: Ozone timeseries
        plt.sca(axes2[i])
        pp.plot_time_series(newdates,o3_new_emiss, label='new_emiss run', linestyle='-.', color=color, linewidth=2)
        pp.plot_time_series(newdates,o3_tropchem, label='tropchem run', linestyle='--', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_omi, dfmt='%Y%m%d', label='OMI', linestyle='-', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_pp, label='OMI recalculated', linestyle=':', color=color, linewidth=2)
        plt.title(label,fontsize=20)
        if i==0:
            plt.ylabel('HCHO cm$^{-2}$')
            plt.legend(loc='best')

    # final touches figure 1
    plt.sca(axes1[i])
    plt.ylabel('HCHO cm$^{-2}$')
    plt.suptitle('%s mean $\Omega_{HCHO}$'%suptitle_prefix, fontsize=26)

    plt.savefig(pname1)
    print('SAVED FIGURE ',pnam1)
    plt.close(f1)
    
    # final touches figure 1
    plt.sca(axes2[i])
    plt.ylabel('O$_3$ cm$^{-2}$')
    plt.suptitle('%s mean O$_3$ tropospheric column'%suptitle_prefix, fontsize=26)

    plt.savefig(pname2)
    print('SAVED FIGURE ',pname2)
    plt.close(f2)

if __name__ == '__main__':

    print("Testing alpha creation")

    start=timeit.default_timer()
    alpha_creation()
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for alpha_creation"%((end-start)/60.0))