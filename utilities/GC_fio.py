# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Reading from GEOS-Chem methods are defined here
'''
## Modules
#import netCDF4 as nc
from xbpch import open_bpchdataset, open_mfbpchdataset
import xarray # for xarray reading of netcdf (hemco diags)
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent folder to path
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#sys.path.insert(0,os.path.dirname(currentdir))

import utilities.utilities as util

#sys.path.pop(0)


##################
#####GLOBALS######
##################

__VERBOSE__=True

run_number={"tropchem":0,"UCX":1,"halfisop":2,"zeroisop":3}
runs=["geos5_2x25_tropchem","UCX_geos5_2x25",
      "geos5_2x25_tropchem_halfisoprene",
      "geos5_2x25_tropchem_noisoprene"]

def _datapaths():
    ''' get location of datafiles, handles either NCI or desktop '''
    folder_location="Data/GC_Output/"

    desktop_dir= "/media/jesse/My Book/jwg366/rundirs/"
    if Path(desktop_dir).is_dir():
        folder_location=desktop_dir

    # NCI folder_location="/home/574/jwg574/OMI_regridding/Data/GC_Output"
    paths=["%s%s/trac_avg"%(folder_location,rstr) for rstr in runs]
    return paths

paths = _datapaths()

# Make these nice name dictionaries in GC_Output class file
__tavg_mainkeys__=['lev','lon','lat','time',
                   'IJ-AVG-$_ISOP','IJ-AVG-$_CH2O','BIOGSRCE_ISOP', 'BIOBSRCE_CH20',
                   'PEDGE-$_PSURF','BXHGHT-$_BXHEIGHT','BXHGHT-$_AD',
                   'BXHGHT-$_AVGW','BXHGHT-$_N(AIR)','DXYP_DXYP',
                   'TR-PAUSE_TP-LEVEL']

__sat_mainkeys__=['lev','lon','lat',
                  'IJ-AVG-$_ISOP','IJ-AVG-$_CH2O',
                  'BIOGSRCE_ISOP',
                  'PEDGE-$_PSURF','BXHGHT-$_BXHEIGHT',
                  'TIME-SER_AIRDEN', #'BXHGHT-$_AD',
                  'TR-PAUSE_TPLEV', # Added satellite output for ppamf
                  'CHEM-L=$_OH',
                  ]

################
###FUNCTIONS####
################

def dataset_to_dicts(ds,keys):
    '''
    '''
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

def read_bpch(path,keys,multi=False):
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

    # Improve read performance by only reading requested fields:
    fields=set(); categories=set()
    for key in keys:
        if '_' in key:
            c,_,f = key.rpartition('_')
            categories.add(c)
            fields.add(f)
        else:
            fields.add(key)
    if __VERBOSE__:
        print("categories: ",categories)
        print("fields: ",fields)

    # get bpch file:
    data={}
    attrs={}
    bpchargs={'fields':list(fields), 'categories':list(categories),
              'tracerinfo_file':tracinf,'diaginfo_file':diaginf,
              'decode_cf':False,'dask':True}
    if not multi:
        ds=open_bpchdataset(path,**bpchargs)
    else:
        ds=open_mfbpchdataset(path,**bpchargs)

    data,attrs=dataset_to_dicts(ds,keys)

    return data,attrs

def read_Hemco_diags(day,month=False):
    '''
        Read Hemco diag output, one day or month at a time
    '''
    fpre='Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.'
    fend=day.strftime(["%Y%m%d","%Y%m"][month]) + "*.nc"
    with xarray.open_mfdataset(fpre+fend) as ds:
        data,attrs=dataset_to_dicts(ds,['ISOP_BIOG'])

    return data,attrs


def determine_trop_column(ppbv, N_air, boxH, tplev):
    '''
        Inputs:
            ppbv[lev,lat,lon]: ppbv of chemical we want the trop column of
            N_air[lev,lat,lon]: number density of air (molec/m3)
            boxH[lev,lat,lon]: level heights (m)
            tplev[lat,lon]: where is tropopause
        Outputs:
            tropcol[lat,lon]: tropospheric column (molec/cm2)
    '''
    dims=np.shape(ppbv)

    # (molec_x/1e9 molec_air) * 1e9 * molec_air/m3 * m * m2/cm2
    X = ppbv * 1e-9 * N_air * boxH * 1e-4 # molec/cm2

    out=np.zeros([dims[1],dims[2]])
    for lat in range(dims[1]):
        for lon in range(dims[2]):
            trop=int(np.floor(tplev[lat,lon]))
            extra=tplev[lat,lon] - trop
            out[lat,lon]= np.sum(X[0:trop,lat,lon]) + extra*X[trop,lat,lon]
    return out


def _test_trop_column_calc():
    '''
    This tests the trop column calculation with a trivial case
    '''
    # lets check for a few dimensions:
    for dims in [ [10,1,1], [100,100,100]]:
        print("testing dummy trop column calculation for dims %s"%str(dims))
        ppbv=np.zeros(dims)+100.    # molec/1e9molecA
        N_air=np.zeros(dims)+1e13   # molecA/m3
        boxH=np.zeros(dims)+1.      # m
        tplev=np.zeros([dims[1],dims[2]]) + 4.4
        # should give 100molec/cm2 per level : tropcol = 440molec/cm2
        out= determine_trop_column(ppbv, N_air, boxH, tplev)
        assert out.shape==(dims[1],dims[2]), 'trop column calc shape is wrong'
        print("PASS: shape is ok from intputs to output")
        assert np.isclose(out[0,0],440.0), 'trop column calc value is wrong, out=%f'%out[0,0]
        print("PASS: Calculated trop column is OK")
        assert np.isclose(np.min(out), np.max(out)), 'Trop column calc has some problem'
        print("PASS: every box matches")


if __name__=='__main__':
    #get_tropchem_data()
    _test_trop_column_calc()
