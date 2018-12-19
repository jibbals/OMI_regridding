#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:33:05 2018

@author: jesse
"""

import numpy as np
from datetime import datetime, timedelta
import timeit

import new_emissions
from utilities import utilities as util
from classes.E_new import E_new


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

if __name__ == '__main__':

    print("Testing alpha creation")

    start=timeit.default_timer()
    alpha_creation()
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for alpha_creation"%((end-start)/60.0))