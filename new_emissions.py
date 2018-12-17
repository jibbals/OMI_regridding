#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:39:30 2018

Handle things for the new_emissions GEOS-Chem runs

@author: jesse jwg366@uowmail.edu.au
"""

import xarray
import numpy as np
from datetime import datetime

from utilities import utilities as util
from classes.E_new import E_new

__VERBOSE__=False


def save_alpha(alpha, elats, elons, path='Data/isoprene_scale_mask_unnamed.nc'):

    # read template isoprene scale mask and modify it for each month
    isop    = xarray.open_dataset('Data/isoprene_scale_mask.nc')
    da      = isop['isop_mask'].data
    lats    = isop['lat'].data
    lons    = isop['lon'].data
    lons[lons>179] = lons[lons>179] - 360 # make it from -180 -> 180

    # match enew lats and mask lats
    lati    = np.where((lats >= elats[0]) * (lats <= [elats[-1]]))
    loni    = np.where((lons >= elons[0]) * (lons <= [elons[-1]]))
    lat0, lat1  = lati[0][0], lati[0][-1]+1
    lon0, lon1  = loni[0][0], loni[0][-1]+1

    # set scale mask over australia
    da[:,lat0:lat1,lon0:lon1] = alpha
    # update the xarray
    isop['isop_mask'].data = da
    # save the new scale mask to a netcdf file
    isop.to_netcdf(path)

def alpha_year(year=2005, test=True):
    '''
        Create isoprene scaling factors monthly over Australia
          using difference from top-down and MEGAN emissions at midday
    '''
    d0 = datetime(year,1,1)
    dn = datetime(year,12,31)

    # Read new top down, and MEGAN emissions
    # ...
    Enew    = E_new(d0,dayn=dn,dkeys=['E_MEGAN','E_PP_lr'])
    dates   = Enew.dates
    elats   = Enew.lats_lr
    elons   = Enew.lons_lr
    megan   = Enew.E_MEGAN
    topd    = Enew.E_PP_lr
    topd[topd<0] = np.NaN
    print(len(dates), np.shape(megan), np.shape(topd))

    # calculate monthly averages, don't worry about NaNs
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')#, r'All-NaN (slice|axis) encountered')
        meganmya   = util.monthly_averaged(dates, Enew.E_MEGAN, keep_spatial=True)['mean']
        topdmya    = util.monthly_averaged(dates, Enew.E_PP_lr, keep_spatial=True)['mean']

    # new / megan = scale
    # ...
    alpha   = topdmya / meganmya
    alpha[np.isnan(alpha)] = 1.0
    alpha[np.isinf(alpha)] = 1.0
    
    if not test:
        save_alpha(alpha,elats,elons, path='Data/isoprene_scale_mask_%4d.nc'%year)
    else:
        # test
        region=[-55,100,-7,165]
        vmin,vmax = 0, 2
        from utilities import plotting as pp
        import matplotlib.pyplot as plt
        sydlat,sydlon = pp.__cities__['Syd']
        months=util.list_months(d0,dn)
        lati,loni = util.lat_lon_index(sydlat,sydlon,elats,elons)
        
        f = plt.figure(figsize=[15,15])
        # first plot alpha in jan, then alpha in 
        plt.subplot(221)
        pp.createmap(alpha[0],elats, elons, linear=True, region=region, title='alpha[0]',vmin=vmin,vmax=vmax)
        # then plot alpha in June
        plt.subplot(222)
        pp.createmap(alpha[6],elats, elons, linear=True, region=region, title='alpha[6]',vmin=vmin,vmax=vmax)
        #finally plot time series at sydney of alpha, megan, and topdown emissions
        plt.subplot(212)
        plt.plot_date(dates, megan[:,lati,loni], 'm-', label='megan')
        plt.plot_date(dates, topd[:,lati,loni], '-', label='Enew', color='cyan')
        plt.ylim(1e11,2e13)
        plt.ylabel('Emissions')
        plt.legend()
        plt.sca(plt.twinx())
        plt.plot_date(months, alpha[:,lati,loni], 'k-', linewidth=3, label='alpha')
        plt.ylabel('Alpha')
        plt.suptitle('Alpha for %4d'%year)
        plt.savefig('alpha_test_%4d.png'%year)
        plt.close()

def alpha_multiyear():
    '''
        take all the E_new dataset and create a multiyear monthly mean alpha
    '''
    d0=datetime(2005,1,1)
    dN=datetime(2012,12,31)
    if __VERBOSE__:
        print('reading Enew')
    start = timeit.default_timer()
    Emiss = E_new(d0,dN,dkeys=['E_PP_lr','E_MEGAN'])
    if __VERBOSE__:
        print("Took %6.2f minutes to read all the Enew"%((timeit.default_timer()-start)/60.0))
    lats=Emiss.lats_lr
    lons=Emiss.lons_lr
    dates=Emiss.dates
    Enew=Emiss.E_PP_lr
    Enew[Enew<0.0] = np.NaN # effectively remove where GC slope is negative...
    Emeg=Emiss.E_MEGAN


    # calculate monthly averages, don't worry about NaNs
    start = timeit.default_timer()
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')#, r'All-NaN (slice|axis) encountered')
        megan   = util.multi_year_average_spatial(Emeg, dates)['mean']
        topd    = util.multi_year_average_spatial(Enew, dates)['mean']
    if __VERBOSE__:
        print("Took %6.2f minutes to get mya"%((timeit.default_timer()-start)/60.0))


    # new / megan = scale
    # ...
    alpha   = topd / megan
    alpha[np.isnan(alpha)] = 1.0
    alpha[np.isinf(alpha)] = 1.0
    
    
    start = timeit.default_timer()
    save_alpha(alpha, lats, lons, path='Data/isoprene_scale_mask_mya.nc')
    if __VERBOSE__:
        print("Took %6.2f minutes to save the alpha"%((timeit.default_timer()-start)/60.0))
    

if __name__=='__main__':

    import timeit

    start=timeit.default_timer()
    alpha_multiyear()

    print("Took %6.2f minutes to run multiyear alpha creation"%((timeit.default_timer()-start)/60.0))

    start=timeit.default_timer()
    for year in np.arange(2005,2012):
        alpha_year(year)

    print("Took %6.2f minutes to run for 1 day"%((timeit.default_timer()-start)/60.0))
