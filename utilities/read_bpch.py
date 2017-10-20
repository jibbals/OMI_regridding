#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:30:45 2017

@author: jesse
"""
import numpy as np
from xbpch import open_bpchdataset
from datetime import datetime as dt

###########################
######## GLOBALS ##########
###########################
__tavg_mainkeys__=['lev','lon','lat','time',
                   'IJ_AVG_S_ISOP','IJ_AVG_S_CH2O','BIOGSRCE_ISOP',
                   'PEDGE_S_PSURF','BXHGHT_S_BXHEIGHT','BXHGHT_S_AD',
                   'BXHGHT_S_AVGW','BXHGHT_S_N(AIR)','DXYP_DXYP']
###########################
######## Funcs ##########
###########################

def read_bpch(path,keys=__tavg_mainkeys__):
    '''
        Read  generic bpch file into dictionary
        keys = keys you want to read
    '''
    # assume tracerinfo and diaginfo in same folder:
    splt=path.split('/')
    splt[-1]='tracerinfo.dat'
    tracinf='/'.join(splt)
    splt[-1]='diaginfo.dat'
    diaginf='/'.join(splt)

    # get bpch file:
    data={}
    attrs={}
    with open_bpchdataset(path, tracerinfo_file=tracinf,diaginfo_file=diaginf) as ds:

        # First read coordinates:
        for key in ds.coords.keys():
            # here we actually go in and read the data
            data[key]=np.array(ds.coords[key])
            attrs[key]=ds[key].attrs

        # then read keys
        for key in keys:
            if key not in ds:
                print("WARNING: %s not in dataset")
                continue
            data[key]=np.array(ds[key])
            attrs[key]=ds[key].attrs

    return data,attrs


print("Testing xbpch")

bpch='Data/GC_Output/geos5_2x25_tropchem/trac_avg/trac_avg.geos5_2x25_tropchem.200501010000'
data,attrs=read_bpch(bpch)

for key in data.keys():
    print(key, data[key].shape)#, attrs[key])

