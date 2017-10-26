# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Reading from GEOS-Chem methods are defined here
'''
## Modules
#import netCDF4 as nc
from xbpch import open_bpchdataset
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
                   'IJ-AVG-$_ISOP','IJ-AVG-$_CH2O','BIOGSRCE_ISOP',
                   'PEDGE-$_PSURF','BXHGHT-$_BXHEIGHT','BXHGHT-$_AD',
                   'BXHGHT-$_AVGW','BXHGHT-$_N(AIR)','DXYP_DXYP']

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


def read_bpch(path,keys):
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
    ds=open_bpchdataset(path,
                        fields=list(fields), categories=list(categories),
                        tracerinfo_file=tracinf,diaginfo_file=diaginf, decode_cf=False)

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

    return data,attrs

def read_tavg(date,run='tropchem',keys=__tavg_mainkeys__):
    '''
        Read tracer avg file for particular date
        run = ['tropchem'|'halfisop'|'UCX']
    '''
    # Determine path of file:
    dstr=date.strftime("%Y%m%d0000")
    pathd={'tropchem':'Data/GC_Output/geos5_2x25_tropchem/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
           'halfisop':'Data/GC_Output/geos5_2x25_tropchem_halfisoprene/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
           'UCX':'Data/GC_Output/UCX_geos5_2x25/trac_avg/trac_avg_geos5_2x25_UCX_updated.%s'}
    path=pathd[run]%dstr

    # grab data from bpch:
    return read_bpch(path,keys)


#def read_trac_avg_nc(date=datetime(2005,1,1), runtype='tropchem', fname=None):
#    '''
#        Read the UCX netcdf file:
#        if fname is set read that from datadir
#        DEPRECATED TODO:REPLACE FULLY WITH BPCH READING
#
#    '''
#    # Using runtype and test flag, determine where trac avg file is
#    fi=run_number[runtype]
#    dstr=date.strftime('%Y%m')
#    filename='trac_avg_%s.nc'%dstr
#    if runtype=='UCX':
#        filename='trac_avg_UCX.nc'
#
#    # Can use non default filename:
#    if fname is not None:
#        filename=fname
#
#    fullpath="%s/%s"%(paths[fi], filename)
#
#    # read the file and return it
#    print("Reading %s"%fullpath)
#    ncfile=nc.Dataset(fullpath,'r')
#    return(ncfile)

#def get_tropchem_data(date=datetime(2005,1,1),runtype='tropchem', monthavg=False, surface=False, fname=None):
#    ''' return a subset of the tropchem data '''
#    tf=read_trac_avg(date, runtype=runtype, fname=fname)
#    Tau=tf.variables['time'][:] # tau dimension
#
#    # get the subset of data: (most are [time, [lev, ]lat,lon])
#    data={}
#
#    for key in tf.variables.keys():
#        # Only reading keys mapped by the classes/variable_names_mapped.py file
#        if key not in GC_trac_avg_map:
#            if __VERBOSE__:
#                print('not reading %s'%key)
#            continue
#        tmp=tf.variables[key][:]
#        if __VERBOSE__:
#            print("%s originally %s, with min, mean, max = (%.2e, %.2e %.2e)"%
#                (key, str(tmp.shape), np.nanmin(tmp), np.nanmean(tmp), np.nanmax(tmp)))
#        # take the monthavg
#        if monthavg:
#            dims=np.shape(tmp)
#            Tdim=0 # time is first dimension
#            if dims[Tdim] == len(Tau): # check we have the time dimension
#                tmp=np.nanmean(tmp,axis=Tdim)
#                if __VERBOSE__:
#                    print("%s averaged over time dimension(%2d)"%(key,Tdim))
#            else:
#                if __VERBOSE__:
#                    print ("%s has no time dimension"%(key))
#
#        if surface: # take surface slice
#            dims=np.shape(tmp)
#            # Ldim could be first or second.
#            Ldim=np.where(dims == 47)[0][0]
#            if Ldim==1:
#                tmp=tmp[:,0]
#            elif Ldim==0:
#                tmp=tmp[0]
#            if __VERBOSE__:
#                print("%s had surface slice taken, now %s"%(key,str(np.shape(tmp))))
#        name=GC_trac_avg_map[key]
#        data[name]=tmp
#        if __VERBOSE__:
#            print("%s (%s) has shape: %s"%(key,name,str(np.shape(data[name]))))
#
#    # return the data we want
#    tf.close()
#    return(data)
#
#
#def get_UCX_data(date=datetime(2005,1,1), surface=False, fname=None):
#    ''' get a month of UCX output '''
#    dstr=date.strftime("%Y%m%d")
#    uf=read_trac_avg(runtype='UCX', fname=fname)
#    Tau     = uf.variables['time'][:] # Tau(time): hours since 19850101:0000
#    Press   = uf.variables['lev'][:] # Pressure(altitude): hPa, midpoints
#
#    # get the subset of data: (most are [time,[lev],lat,lon])
#    data={}
#
#    # find the month index:
#    di=util.index_from_gregorian(Tau,date)
#    if __VERBOSE__:
#        print ("date index = ")
#        print (di)
#    for key in uf.variables.keys():
#        # Only reading keys mapped by the classes/variable_names_mapped.py file
#        if key not in GC_trac_avg_map:
#            if __VERBOSE__:
#                print('not reading %s'%key)
#            continue
#        tmp=uf.variables[key][:]
#        if __VERBOSE__:
#            print("%s originally %s, with min, mean, max = (%.2e, %.2e %.2e)"%
#                (key, str(tmp.shape), np.nanmin(tmp), np.nanmean(tmp), np.nanmax(tmp)))
#        dims=np.shape(tmp)
#        Tdim=0 # time is first dimension
#        if dims[Tdim] == len(Tau): # check we have the time dimension
#            tmp=np.squeeze(tmp[di])
#            if __VERBOSE__:
#                print("%s date extracted(%s), now shape=%s"%(key,dstr,str(tmp.shape)))
#        else:
#            if __VERBOSE__:
#                print ("%s has no time dimension"%(key))
#        # Take surface level if desired
#        if surface:
#            dims=np.shape(tmp)
#            if (len(dims) > 0) and (dims[0] == len(Press)):
#                tmp=np.squeeze(tmp[0]) # take surface slice
#                if __VERBOSE__:
#                    print("%s had surface slice taken"%(key))
#            else:
#                if __VERBOSE__:
#                    print ("%s has no level dimension"%(key))
#        name=GC_trac_avg_map[key]
#        data[name]=tmp
#        if __VERBOSE__:
#            print("%s (%s) has shape: %s"%(key,name,str(np.shape(data[name]))))
#
#    # return the data we want
#    uf.close()
#    return(data)

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
