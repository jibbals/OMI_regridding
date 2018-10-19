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
from classes import GC_class


import concurrent.futures # parallel reading of data
from datetime import datetime
import numpy as np
import timeit # see how slow stuff is

def hcho_lifetime(month, region=pp.__AUSREGION__):
    '''
    Use tau = HCHO / Loss    to look at hcho lifetime over a month
    return lifetimes[day,lat,lon],
    '''
    print("CHECK: TRYING HCHO_LIFETIME:",month, region)

    d0=util.first_day(month)
    dN=util.last_day(month)

    # read hcho and losses from trac_avg
    # ch20 in ppbv, air density in molec/cm3,  Loss HCHO mol/cm3/s
    keys=['IJ-AVG-$_CH2O','BXHGHT-$_N(AIR)', 'PORL-L=$_LHCHO']
    run = GC_class.GC_tavg(d0,dN, keys=keys) # [time, lat, lon, lev]
    print('     :READ GC_tavg')
    # TODO: Instead of surface use tropospheric average??
    hcho = run.hcho[:,:,:,0] # [time, lat, lon, lev=47] @ ppbv
    N_air = run.N_air[:,:,:,0] # [time, lat, lon, lev=47] @ molec/cm3
    Lhcho = run.Lhcho[:,:,:,0] # [time, lat, lon, lev=38] @ mol/cm3/s  == molec/cm3/s !!!!
    lats=run.lats
    lons=run.lons

    # [ppbv * 1e-9 * (molec air / cm3) / (molec/cm3/s)] = s
    tau = hcho * 1e-9 * N_air  /  Lhcho
    print('     :CALCULATED tau')
    # change to hours
    tau=tau/3600.
    #

    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region,[tau],has_time_dim=True)
        tau=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']
    print("CHECK: MANAGED HCHO_LIFETIME:",month)
    return tau, run.dates, lats, lons

def make_smear_mask_file(year, region=pp.__AUSREGION__, use_GC_lifetime=True, max_procs=2):
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
    tau0, dates0, lats, lons = hcho_lifetime(d0, region=region)

    # set up variables we will save
    tau         = np.zeros([len(dates), len(lats), len(lons)])
    #yields      = np.zeros(tau.shape)
    slope       = np.zeros(tau.shape)
    smear       = np.zeros(tau.shape)
    smearmin    = np.zeros(tau.shape)
    smearmax    = np.zeros(tau.shape)
    smearmask   = np.zeros(tau.shape, dtype=np.int)

    # first read year of GC lifetimes
    di0 = util.date_index(d0,dates,util.last_day(d0))
    tau[di0] = tau0
    print("CHECK: Reading year of GC loss rates and concentrations to get lifetimes")
    if max_procs>1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            procreturns=executor.map(hcho_lifetime, months[1:], [region]*11)
            for i,pret in enumerate(procreturns):
                datesi = pret[1] # dates for this month
                di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
                tau[di]=pret[0]
    else:
        for i, month in enumerate(months[1:]):
            taui, datesi, latsi, lonsi = hcho_lifetime(month, region)
            di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
            tau[di] = taui

    ## read year of smear
    # first read month
    smear0, dates0, lats, lons = Inversion.smearing(d0,region=region)
    smear[di0]  = smear0
    print("CHECK: Reading year of GC columns and emissions to get smearing")
    if max_procs>1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            procreturns=executor.map(Inversion.smearing, months[1:], [region]*11)
            for i,pret in enumerate(procreturns):
                datesi = pret[1] # dates for this month
                di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
                smear[di]=pret[0]
    else:
        for i, month in enumerate(months[1:]):
            smeari, datesi, latsi, lonsi = Inversion.smearing(month, region)
            di = util.date_index(datesi[0],dates,util.last_day(datesi[0]))
            smear[di] = smeari

    ## read monthly slope
    #
    print("CHECK: Reading year of biogenic GC columns to get slope")
    for month in months:
        gc=GC_class.GC_biogenic(month)
        model_slope=gc.model_slope(region=region)
        slopei = model_slope['slope']
        days=util.list_days(month,month=True)
        datesi = util.date_index(days[0], dates, days[-1])
        # Also repeat Slope array along time axis to avoid looping
        slopei  = np.repeat(slopei[np.newaxis,:,:], len(days), axis=0)
        slope[datesi] = slopei

    ## S = Y/k
    #Yield is just k(=1/lifetime) * Slope
    yields = slope / tau

    ## Finally save the data to a file
    ## add attributes to be saved in file
    #
    dattrs  = {'smearmask':{'units':'int','desc':'0 or 1: grid square potentially affected by smearing'},
              'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of anthromask array'},
              'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
              'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'},
              'smear':{'units':'s','desc':'year average of NO2'},
              'yields':{'units':'molec_HCHO/atom_C','desc':'HCHO molecules per Atom C isoprene emissions'},
              'slope':{'units':'s','desc':'modelled slope between HCHO columns and E_isop, repeated from monthly to a daily timescale'},
              'tau':{'utits':'hrs','desc':'hcho lifetime modelled from GEOS-Chem'}}
    ## data dictionary to save to hdf
    #
    dates=util.gregorian_from_dates(dates)
    datadict={'smearmask':smearmask,'dates':dates,'lats':lats,'lons':lons,
              'smear':smear, 'yields':yields,'tau':tau, 'slope':slope}

    # filename and save to h5 file
    path=year.strftime('Data/smearmask_%Y.h5')
    fio.save_to_hdf5(path, datadict, attrdicts=dattrs)