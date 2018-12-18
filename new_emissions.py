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

def calculate_alpha(year=2005, mya=False):
    '''
        take Enew top down / Emegan to create monthly alpha
    '''
    if mya:
        d0 = datetime(2005,1,1)
        dn = datetime(2012,12,1)
    else:
        d0 = datetime(year,1,1)
        dn = datetime(year,12,31)
    months=util.list_months(d0,dn)

    # Read new top down, and MEGAN emissions
    # ...
    Enew    = E_new(d0,dayn=dn,dkeys=['E_MEGAN','E_PP_lr'])
    dates   = Enew.dates
    elats   = Enew.lats_lr
    elons   = Enew.lons_lr
    megan   = Enew.E_MEGAN
    topd    = Enew.E_PP_lr
    with np.errstate(invalid='ignore'):
        topd[topd<0] = 0 # no longer np.NaN

    if __VERBOSE__:
        print(' calculating alpha for this many dates ')
        print(len(dates), np.shape(megan), np.shape(topd))

    # calculate monthly averages, don't worry about NaNs
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')#, r'All-NaN (slice|axis) encountered')
        meganmya   = util.multi_year_average_spatial(megan,dates)['mean']
        topdmya    = util.multi_year_average_spatial(topd,dates)['mean']

    # new / megan = scale
    # ...
    alpha   = topdmya / meganmya
    alpha[np.isnan(alpha)] = 1.0
    alpha[np.isinf(alpha)] = 1.0

    return {'alpha':alpha, 'lats':elats, 'lons':elons,
            'dates':dates, 'months':months, 'Enew':topd, 'Emeg':megan,
            'Enewm':topdmya, 'Emegm':meganmya}

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

def create_alpha_file(year=2005, mya=False):
    '''
        take the E_new dataset and create a monthly mean alpha
        can do one year or take the multiyearavg (mya=True)
    '''
    dat = calculate_alpha(year,mya=mya)
    alpha=dat['alpha']
    lats=dat['lats']
    lons=dat['lons']
    #dates=dat['dates']
    #months=dat['months']

    path='Data/isoprene_scale_mask_%4d.nc'%year
    if mya:
        path='Data/isoprene_scale_mask_mya.nc'

    save_alpha(alpha, lats, lons, path=path)


if __name__=='__main__':

    import timeit

    start=timeit.default_timer()
    create_alpha_file(mya=True)

    print("Took %6.2f minutes to run multiyear alpha creation"%((timeit.default_timer()-start)/60.0))

    start=timeit.default_timer()
    for year in np.arange(2005,2012):
        create_alpha_file(year=year, mya=False)

    print("Took %6.2f minutes to run for 1 day"%((timeit.default_timer()-start)/60.0))
