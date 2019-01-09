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


if __name__ == '__main__':

    print("Testing alpha creation")

    start=timeit.default_timer()
    alpha_creation()
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for alpha_creation"%((end-start)/60.0))