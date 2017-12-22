#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:23:10 2017

@author: jesse
"""
import xarray
from datetime import datetime,timedelta
import numpy as np

__VERBOSE__=True

def dataset_to_dicts(ds,keys=None):
    '''
    '''
    if keys is None:
        keys=ds.keys()

    data,attrs={},{}
    # First read coordinates:
    for key in ds.coords.keys():
        data[key]=np.array(ds.coords[key]) # could also keep xarray or dask array
        attrs[key]=ds[key].attrs

    # then read keys
    for key in keys:
        if key not in ds:
            print("WARNING: %s not in dataset"%key)
            continue
        data[key]=np.array(ds[key])
        attrs[key]=ds[key].attrs
        if 'scale' in attrs[key].keys():
            data[key] = data[key]*float(attrs[key]['scale'])
            if __VERBOSE__:
                print("%s scaled by %.2e"%(key,float(attrs[key]['scale'])))

    return data,attrs

def read_netcdf(path,keys=None,multi=False):
    '''
        Read generic netcdf file into dictionary
        keys = keys you want to read
    '''

    data={}
    attrs={}
    args={'decode_cf':False}

    if multi:
        ds=xarray.open_mfdataset(path,**args)
    else:
        ds=xarray.open_dataset(path,**args)

    data,attrs=dataset_to_dicts(ds,keys)

    return data,attrs

def read_spei(date0=datetime(2005,1,1),date1=datetime(2005,12,1)):

    # read file into dictionaries:
    spei,attrs=read_netcdf('spei01.nc')
    #for key in spei.keys():
    #    print(key, spei[key].shape, attrs[key])

    datetimes=[]
    for days in spei['time']:
        # 'days since 1900-1-1'
        date=datetime(1900,1,1)+timedelta(days=days)
        # date is roughly middle of each month
        # set day to 1
        date1=date.replace(day=1)
        datetimes.append(date1)
    dates=np.array(datetimes)

    inds= (dates>=date0) * (dates <= date1)
    for key in spei.keys():
        if spei[key].shape[0] == len(dates):
            if __VERBOSE__:
                print('read_spei now subsetting ', key)
            spei[key] = spei[key][inds]


    spei['dates']=dates[inds]
    attrs['dates']={'desc':'datetimes at start of each month of monthly averaged data'}

    return spei, attrs

