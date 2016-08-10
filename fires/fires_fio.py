# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:11:42 2016

File specifically for reading and writing the MODIS (AQUA) fire data

@author: jesse
"""

# Python script reading hdf5 junk

## Modules
# module for hdf eos 5
import h5py
from datetime import datetime as dt
import numpy as np
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata

def read_8dayfire(date=dt(2005,1,1,0)):
    '''
    Read file containing date
    '''
    # filenames are all like *yyyyddd.h5, where ddd is day of the year, one for every 8 days
    tt = date.timetuple()
    daymatch= int(np.floor(tt.tm_yday/8)*8) # only every 8 days matches a file
    if daymatch == 0:
        daymatch = 1
    filepath='Fires/MYD14C8H.%4d%03d.h5' % (tt.tm_year, daymatch)
    return read_8dayfire_path(filepath)
    

def read_8dayfire_path(path):
    '''
    Read fires file using given path
    '''    
    ## Fields to be read:
    # Count of fires in each grid box over 8 days
    corrFirePix = 'CorrFirePix'
    cloudCorrFirePix = 'CloudCorrFirePix'
    
    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        cfp     = in_f[corrFirePix].value
        ccfp    = in_f[cloudCorrFirePix].value
    
    # from document at http://www.fao.org/fileadmin/templates/gfims/docs/MODIS_Fire_Users_Guide_2.4.pdf
    # latitude = 89.75 - 0.5 * y
    # longitude = -179.75 + 0.5 * x 
    lats    = np.arange(90,-90,-0.5) - 0.25
    lons    = np.arange(-180,180, 0.5) + 0.25
    return (cfp, lats, lons)
    
def read_8dayfire_interpolated(date,latres,lonres):
    '''
    Read the date, interpolate data to match lat/lon resolution, return data
    '''
    ##original lat/lons:
    #lats = np.arange(90,-90,-0.5) - 0.25
    #lons = np.arange(-180,180, 0.5) + 0.25

    fires, lats, lons = read_8dayfire(date)
    
    newlats= np.arange(-90,90, latres) + latres/2.0
    newlons= np.arange(-180,180, lonres) + lonres/2.0

    mlons,mlats = np.meshgrid(lons,lats) # this order keeps the lats/lons dimension order
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)    
    
    
    interp = griddata( (mlats.ravel(), mlons.ravel()), fires.ravel(), (mnewlats, mnewlons), method='nearest')
    return interp, newlats, newlons
    
# for testing:
#ret=read_8dayfire_interpolated(dt(2005,1,1), 0.25, 0.25)