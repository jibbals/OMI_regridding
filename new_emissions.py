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


def make_isoprene_scale_factor(year=2005):
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
    topd    = Enew.E_VCC_PP_lr
    topd[topd<0] = 0.0
    # calculate monthly averages, don't worry about NaNs
    with np.errstate(divide='ignore', invalid='ignore'):
        megan   = util.monthly_averaged(dates, Enew.E_MEGAN, keep_spatial=True)['mean']
        topd    = util.monthly_averaged(dates, Enew.E_VCC_PP_lr, keep_spatial=True)['mean']

    # new / megan = scale
    # ...
    alpha   = topd / megan

    # read template isoprene scale mask and modify it for each month
    isop    = xarray.open_dataset('Data/isoprene_scale_mask.nc')
    da      = isop['isop_mask'].data
    lats    = isop['lat'].data
    lons    = isop['lon'].data
    # match enew lats and mask lats
    lati    = (lats >= elats[0]) * (lats <= [elats[-1]])
    loni    = (lons >= elons[0]) * (lons <= [elons[-1]])

    #lons[lons>179] = lons[lons>179] - 360 # make it from -180 -> 180
    #mlons,mlats = np.meshgrid(lons,lats)
    #ausmap=(mlats > -60) * (mlats < -10) * (mlons > 110) * (mlons < 160)
    ausmap=np.repeat(ausmap[np.newaxis,:,:],12,axis=0)

    # set scale mask over australia
    da[ausmap] = alpha
    # update the xarray
    isop['isop_mask'].data = da
    # save the new scale mask to a netcdf file
    isop.to_netcdf('Data/isoprene_scale_mask_%4d.nc'%year)
