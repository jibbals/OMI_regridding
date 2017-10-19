#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:30:45 2017

@author: jesse
"""
import numpy as np
from xbpch import open_bpchdataset


ddir="Data/GC_Output/geos5_2x25_tropchem/trac_avg/"
bpch_fname=ddir+"trac_avg.geos5_2x25_tropchem.200503010000"
diaginf=ddir+"diaginfo.dat"
tracinf=ddir+"tracerinfo.dat"
print("Testing xbpch on %s"%bpch_fname)

#with open_bpchdataset(bpch_fname, tracerinfo_file=tracinf,diaginfo_file=diaginf) as ds:
ds=open_bpchdataset(bpch_fname, tracerinfo_file=tracinf,diaginfo_file=diaginf)
print(ds.keys())
ds.coords # shows coordinates

#d=ds.to_dict() # TAKES FOREVER, must read everything at this point.

# Just read coords and one or two fields:

# first get coords:
lats=ds.coords['lat'].data
for key in ds.coords['lat'].attrs:
    print(key," = ", ds.coords['lat'].attrs[key])
lons=ds.coords['lon'].data
levs=ds.coords['lev'].data

# then get isop and hcho:
keys=['DXYP_DXYP','IJ_AVG_S_CH2O','BIOGSRCE_ISOP']
data={}
for key in keys:
    data[key]=np.array(ds.data_vars[key].data)

# DASK ARRAYS ARE GOOD FOR LARGE DATASETS.
hcho=ds.data_vars[keys[1]].data

first_day_surface=hcho[0,:,:,0] # slicing operation is delayed until compute is called
print(first_day_surface.compute())
