#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:45:41 2018

handle mask creation and reading/writing

@author: jwg366
"""

import Inversion # for smearing ... should move here?
from utilities import utilities as util
from utilities import plotting as pp
from utilities import fio


import concurrent.futures # parallel reading of data
from datetime import datetime
import numpy as np
import timeit # see how slow stuff is

### GLOBALS ###
#1.5hrs at 0.2 yield up to 4hrs at .4 yield minus 20% and rounded to 100
__smearminlit__ = 800
__smearmaxlit__ = 4600


def make_smear_mask_file(year, region=pp.__AUSREGION__, use_GC_lifetime=True):
    '''
        Estimate expected yield (assuming HCHO lifetime=2.5 or using GC hcho loss to approximate)
        determine bounds for acceptable smearing range
        create a 3d mask for the year of smearing min and max values
        saves smear_mask.nc: lifetimes, yields, smearmin, smearmax, smear, smearmask
    '''

    d0      = datetime(year,1,1)
    dN      = datetime(year,12,31)
    dates   = util.list_days(d0,dN)
    months  = util.list_months(d0,dN)

    # Read first month of HCHO lifetime
    #tau0, dates0, lats, lons = hcho_lifetime(d0, region=region)
    ## read year of smear
    # first read month
    smear0, dates0, lats, lons = Inversion.smearing(d0,region=region)

    # set up variables we will save
    #tau         = np.zeros([len(dates), len(lats), len(lons)])
    #yields      = np.zeros(tau.shape)
    #slope       = np.zeros(tau.shape)
    smear       = np.zeros([len(dates), len(lats), len(lons)])
    smearmin    = np.zeros(smear.shape) # dynamic smear mask if possible from geos-chem
    smearmax    = np.zeros(smear.shape) # dynamic smear mask ..
    smearmask   = np.zeros(smear.shape, dtype=np.int) # smearmask from GEOS-Chem or Caaba Mecca if possible


    smearmasklit= np.zeros(smear.shape, dtype=np.int) # smearmask from literature (800-4600)

    # first date index
    di0 = util.date_index(d0,dates,util.last_day(d0))

    #    ## first read year of GC lifetimes
    #    # one month at a time: starting at jan
    #    tau[di0] = tau0
    #    print("CHECK: Reading year of GC loss rates and concentrations to get lifetimes")
    #    for i, month in enumerate(months[1:]):
    #        taui, datesi, latsi, lonsi = hcho_lifetime(month, region)
    #        di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
    #        tau[di] = taui

    smear[di0]  = smear0
    print("CHECK: Reading year of GC columns and emissions to get smearing")
    for i, month in enumerate(months[1:]):
        smeari, datesi, latsi, lonsi = Inversion.smearing(month, region)
        di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
        smear[di] = smeari

    ## Smearmask based on literature
    #
    smearmasklit[smear < __smearminlit__] = 1
    smearmasklit[smear > __smearmaxlit__] = 2

    ## read monthly slope
    #
    #    print("CHECK: Reading year of biogenic GC columns to get slope")
    #    for month in months:
    #        gc=GC_class.GC_biogenic(month)
    #        model_slope=gc.model_slope(region=region)
    #        slopei = model_slope['slope']
    #        days=util.list_days(month,month=True)
    #        datesi = util.date_index(days[0], dates, days[-1])
    #        # Also repeat Slope array along time axis to avoid looping
    #        slopei  = np.repeat(slopei[np.newaxis,:,:], len(days), axis=0)
    #        slope[datesi] = slopei

    ## S = Y/k
    #Yield is just k(=1/lifetime) * Slope
    #yields = slope / tau

    ## Now create mask based on smearing number
    ##
    # If we assume lifetimes are roughly half that estimated from GEOS-Chem, we
    # can get a range of acceptable yields for each grid square..


    ## Finally save the data to a file
    ## add attributes to be saved in file
    #
    dattrs = {'smearmask':{'units':'int','desc':'0 or 1: grid square potentially affected by smearing'},
              'smearmasklit':{'units':'int','desc':'0 or 1 or 2: smearing outside range of %d to %d (less than range=1, greater than range = 2)'%(__smearminlit__,__smearmaxlit__)},
              'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of anthromask array'},
              'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
              'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'},
              'smear':{'units':'s','desc':'Daily midday smearing from GEOS-Chem'},
              'yields':{'units':'molec_HCHO/atom_C','desc':'HCHO molecules per Atom C isoprene emissions'},
              'slope':{'units':'s','desc':'modelled slope between HCHO columns and E_isop, repeated from monthly to a daily timescale'},
              'tau':{'utits':'hrs','desc':'hcho lifetime modelled from GEOS-Chem'}}
    ## data dictionary to save to hdf
    #
    dates=util.gregorian_from_dates(dates)
    datadict={'smearmask':smearmask,'dates':dates,'lats':lats,'lons':lons,
              'smearmasklit':smearmasklit,
              'smear':smear,} #'yields':yields,'tau':tau, 'slope':slope}

    # filename and save to h5 file
    path='Data/smearmask_%d.h5'%year
    fio.save_to_hdf5(path, datadict, attrdicts=dattrs)

def read_smearmask(d0, dN=None, keys=None):
    '''
        Read smearmask (or extra keys) between d0 and dN
    '''
    path= 'Data/smearmask_%d.h5'%d0.year
    data, attrs = fio.read_hdf5(path)
    # subset to rerquested dates, after converting greg to numpy datetime
    dates = util.date_from_gregorian(data['dates'])
    data['dates'] = np.array(dates)
    attrs['dates']['units'] = 'numpy datetime'

    di = util.date_index( d0, dates, dN)
    for key in ['smearmask','smearmasklit','smear','yields','tau','slope', 'dates']:
        data[key]=data[key][di]

    # subset to desired keys
    if keys is not None:
        for key in ['smearmask','smearmasklit','smear','yields','tau','slope']:
            if key not in keys:
                removed = data.pop(key)
                removed = attrs.pop(key)
    return data, attrs

def get_smear_mask(d0, dN=None, region=None, uselitmask=True):
    '''
        Just grab smearing mask in true/false from d0 to dN
    '''
    key = ['smearmask','smearmasklit'][uselitmask]
    data, attrs = read_smearmask(d0, dN, keys=[key,])
    smearmask=data[key] > 0.5
    dates=data['dates']
    lats,lons = data['lats'],data['lons']

    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[smearmask],has_time_dim=True)
        smearmask=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']

    return smearmask, dates, lats, lons

