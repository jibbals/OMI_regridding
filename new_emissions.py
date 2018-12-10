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

def alpha_year(year=2005):
    '''
        Create isoprene scaling factors monthly over Australia
          using difference from top-down and MEGAN emissions at midday
    '''
    d0 = datetime(year,1,1)
    dn = datetime(year,12,31)

    # Read new top down, and MEGAN emissions
    # ...
    Enew    = E_new(d0,dayn=dn,dkeys=['E_MEGAN','E_VCC_PP_lr'])
    dates   = Enew.dates
    elats   = Enew.lats_lr
    elons   = Enew.lons_lr
    megan   = Enew.E_MEGAN
    topd    = Enew.E_PP_lr
    topd[topd<0] = np.NaN

    # calculate monthly averages, don't worry about NaNs
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')#, r'All-NaN (slice|axis) encountered')
        megan   = util.monthly_averaged(dates, Enew.E_MEGAN, keep_spatial=True)['mean']
        topd    = util.monthly_averaged(dates, Enew.E_VCC_PP_lr, keep_spatial=True)['mean']

    # new / megan = scale
    # ...
    alpha   = topd / megan
    alpha[np.isnan(alpha)] = 1.0
    alpha[np.isinf(alpha)] = 1.0

    save_alpha(alpha,elats,elons, path='Data/isoprene_scale_mask_%4d.nc'%year)

    ## test
    #region=[-60,85,-5,170]
    #vmin,vmax = 0, 2
    #import matplotlib.pyplot as plt
    #from utilities import plotting as pp
    #f, axes = plt.subplots(2,2,figsize=[15,15])
    #plt.sca(axes[0,0])
    #pp.createmap(da[0],lats, lons, linear=True, region=region, title='da[0]',vmin=vmin,vmax=vmax)
    #plt.sca(axes[0,1])
    #pp.createmap(alpha[0],elats, elons, linear=True, region=region, title='alpha[0]',vmin=vmin,vmax=vmax)
    #plt.sca(axes[1,0])
    #pp.createmap(da[6],lats, lons, linear=True, region=region, title='da[6]',vmin=vmin,vmax=vmax)
    #plt.sca(axes[1,1])
    #pp.createmap(alpha[6],elats, elons, linear=True, region=region, title='alpha[6]',vmin=vmin,vmax=vmax)
    #plt.savefig('alpha_test.png')
    #plt.close()

def alpha_multiyear():
    '''
        take all the E_new dataset and create a multiyear monthly mean alpha
    '''
    d0=datetime(2005,1,1)
    dN=datetime(2012,12,31)
    Emiss = E_new(d0,dN,dkeys=['E_PP_lr','E_MEGAN'])
    lats=Emiss.lats
    lons=Emiss.lons
    dates=Emiss.dates
    Enew=Emiss.E_PP_lr
    Enew[Enew<0.0] = np.NaN # effectively remove where GC slope is negative...
    Emeg=Emiss.E_MEGAN

    # calculate monthly averages, don't worry about NaNs
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')#, r'All-NaN (slice|axis) encountered')
        megan   = util.multi_year_average_spatial(Emeg, dates)['mean']
        topd    = util.multi_year_average_spatial(Enew, dates)['mean']

    # new / megan = scale
    # ...
    alpha   = topd / megan
    alpha[np.isnan(alpha)] = 1.0
    alpha[np.isinf(alpha)] = 1.0

    save_alpha(alpha, lats, lons, path='Data/isoprene_scale_mask_mya.nc')