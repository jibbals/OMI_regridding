#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:39:30 2018

Handle things for the new_emissions GEOS-Chem runs

@author: jesse jwg366@uowmail.edu.au
"""

import xarray
import numpy as np

from classes import GC_class


def make_isoprene_scale_factor(year=2005):
    '''
        Create isoprene scaling factors monthly over Australia
          using difference from top-down and MEGAN emissions at midday
    '''

    # Read HEMCO diags showing MEGAN isoprene
    #...

    # Read top down estimates
    # ...

    # new / megan = scale
    # ...


    # read template isoprene scale mask and modify it for each month
    isop = xarray.open_dataset('Data/isoprene_scale_mask.nc')
    da=isop['isop_mask'].data
    lats=isop['lat'].data
    lons=isop['lon'].data
    mlons,mlats = np.meshgrid(lons,lats)
    #ausmap=(mlats > -60) * (mlats < -10) * (mlons > 110) * (mlons < 160)
    #ausmap=np.repeat(ausmap[np.newaxis,:,:],12,axis=0)
    #da[ausmap] = 0.0
    isop['isop_mask'].data = da

    new_ds=isop.to_netcdf('Data/isoprene_scale_mask_%4d.nc'%year)