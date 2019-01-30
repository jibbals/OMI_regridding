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
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
import os.path, time # for file modification time

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

import utilities.utilities as util
from utilities import GMAO

#from classes.GC_class import __coords__ as GC_coords
GC_coords = ['lev','lon','lat','time']
sys.path.pop(0)


##################
#####GLOBALS######
##################

__VERBOSE__=True

run_number={"tropchem":0,"UCX":1,"halfisop":2,"zeroisop":3,"nochem":4}
runs=["geos5_2x25_tropchem","UCX_geos5_2x25",
      "geos5_2x25_tropchem_halfisoprene",
      "geos5_2x25_tropchem_noisoprene",
      "nochem"]

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
            if __VERBOSE__:
                print("WARNING: %s not in dataset"%key)
            continue
        data[key]=np.array(ds[key])
        attrs[key]=ds[key].attrs
        if 'scale' in attrs[key].keys():
            data[key] = data[key]*float(attrs[key]['scale'])
            if __VERBOSE__:
                print("%s scaled by %.2e"%(key,float(attrs[key]['scale'])))

    return data,attrs

def read_bpch(path,keys):
    '''
        Read  generic bpch file into dictionary
        keys = keys you want to read
    '''
    paths=path
    if __VERBOSE__:
        print('GC_fio.read_bpch called on paths:')
        print(path)
    multi=False
    if isinstance(path,list):
        path=path[0]
        if len(path) > 1:
            multi=True
    if '*' in path:
        multi=True

    # make sure coordinates are in keys list
    keys = list(set(keys + GC_coords)) # set removes any duplicates

    # assume tracerinfo and diaginfo in same folder:
    # otherwise use my generic one with coalesced info
    splt=path.split('/')
    splt[-1]='tracerinfo.dat'
    tracinf='/'.join(splt)
    if not os.path.isfile(tracinf):
        tracinf='Data/GC_Output/tracerinfo.dat'

    splt[-1]='diaginfo.dat'
    diaginf='/'.join(splt)
    if not os.path.isfile(diaginf):
        diaginf='Data/GC_Output/diaginfo.dat'


    # Improve read performance by only reading requested fields:
    fields=set(); categories=set()
    for key in keys:
        if '_' in key:
            # Split key on the underscores: Category_Field
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
    mod_times=[]
    if multi:
        ds=open_mfbpchdataset(paths,**bpchargs)
        for p in paths:
            mod_times.append(time.ctime(os.path.getmtime(p)))
    else:
        ds=open_bpchdataset(path,**bpchargs)
        mod_times=[time.ctime(os.path.getmtime(path))]

    data,attrs=dataset_to_dicts(ds,keys)
    data['modification_times']=np.array(mod_times)
    attrs['modification_times']={'desc':'When was file last modified'}
    return data,attrs

def make_overpass_output_files():
    '''
        Read GEOS-Chem satellite output for tropchem and new_emissions runs
            create one he5 with all the data from 2005-201305
    '''
    #for folder in 
    # match all files with YYYYMM[DD] tag
    #folder=['geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog',
    #        'new_emissions/diagnostics/emiss_a'][new_emissions]
    keys_to_keep=[  'IJ-AVG-$NO', #    ppbv   144  91  47
                    'IJ-AVG-$O3', #    ppbv   144  91  47
   3 : IJ-AVG-$   20     ISOP      6            ppbC 175536.25 2005011000  144  91  47
   4 : IJ-AVG-$   20     CH2O     20            ppbv 175536.25 2005011000  144  91  47
   5 : IJ-AVG-$   20      NO2     64            ppbv 175536.25 2005011000  144  91  47
   6 :  PEDGE-$   20    PSURF  10001             hPa 175536.25 2005011000  144  91  47
   7 : DAO-FLDS   20    PARDF  11020            W/m2 175536.25 2005011000  144  91  47
   8 : DAO-FLDS   20    PARDR  11021            W/m2 175536.25 2005011000  144  91  47
   9 : DAO-FLDS   20       TS  11005               K 175536.25 2005011000  144  91  47
  10 : DAO-3D-$   20     TMPU  12003               K 175536.25 2005011000  144  91  47
  11 : OD-MAP-$   20    OPSO4  14006        unitless 175536.25 2005011000  144  91  47
  12 : OD-MAP-$   20     OPBC  14009        unitless 175536.25 2005011000  144  91  47
  13 : OD-MAP-$   20     OPOC  14012        unitless 175536.25 2005011000  144  91  47
  14 : OD-MAP-$   20    OPSSa  14015        unitless 175536.25 2005011000  144  91  47
  15 : OD-MAP-$   20    OPSSc  14018        unitless 175536.25 2005011000  144  91  47
  16 : OD-MAP-$   20      OPD  14004        unitless 175536.25 2005011000  144  91  47
  17 : OD-MAP-$   20     OPD1  14021        unitless 175536.25 2005011000  144  91  47
  18 : OD-MAP-$   20     OPD2  14022        unitless 175536.25 2005011000  144  91  47
  19 : OD-MAP-$   20     OPD3  14023        unitless 175536.25 2005011000  144  91  47
  20 : OD-MAP-$   20     OPD4  14024        unitless 175536.25 2005011000  144  91  47
  21 : OD-MAP-$   20     OPD5  14025        unitless 175536.25 2005011000  144  91  47
  22 : OD-MAP-$   20     OPD6  14026        unitless 175536.25 2005011000  144  91  47
  23 : OD-MAP-$   20     OPD7  14027        unitless 175536.25 2005011000  144  91  47
  24 : CHEM-L=$   20       OH  16001       molec/cm3 175536.25 2005011000  144  91  47
  25 : TIME-SER   20           19002       UNDEFINED 175536.25 2005011000  144  91  47
  26 : TIME-SER   20           19007       molec/cm3 175536.25 2005011000  144  91  47
  27 : TIME-SER   20           19009           m2/m2 175536.25 2005011000  144  91  47
  28 : BIOGSRCE   20     ISOP  21001     atomC/cm2/s 175536.25 2005011000  144  91  47
  29 : BXHGHT-$   20 BXHEIGHT  24001               m 175536.25 2005011000  144  91  47
  30 : TR-PAUSE   20           26015           level 175536.25 2005011000  144  91  47
    bpch = 'new_emissions/diagnostics/satellite_output/ts_satellite_altered.%s.bpch'
    fpre='Data/GC_Output/%s.'%bpch
    # FOR TESTING JUST DO 2005,2006
    years=util.list_years(datetime(2005,1,1),datetime(2006,2,2))
    yearly_data=[]
    for year in years:
        d0=datetime(year.year,1,1)
        d1=datetime(year.year,12,31)
        if year.year==2013:
            d1=datetime(2013,5,31) # special case in 2013
        dlist=util.list_days(d0,d1)
        
        # file names have date strings in name
        files=[]
        for day in dlist:
            fend=day.strftime("%Y%m%d")
            files.extend(glob(fpre%fend))

        files.sort() # make sure they're sorted or the data gets read in poorly
    
        print('check overpass files being read (show 1 in 24):', files[::24])
        
        # now read the data from all those files
        with xarray.open_mfdataset(files) as ds:
            data,attrs=dataset_to_dicts(ds,['ISOP_BIOG'])

    mod_times=[]
    for p in files:
        mod_times.append(time.ctime(os.path.getmtime(p)))
    data['modification_times']=np.array(mod_times)
    attrs['modification_times']={'desc':'When was file last modified'}

    return data,attrs



def read_Hemco_diags_hourly(d0,d1=None,month=False,new_emissions=False):
    '''
        Read Hemco diag output, one day or month at a time
    '''
    # match all files with YYYYMM[DD] tag
    folder=['geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog',
            'new_emissions/diagnostics/emiss_a'][new_emissions]
    fpre='Data/GC_Output/%s.'%folder
    dlist=util.list_days(d0,d1,month=month)
    # for each day: glob matching files to list
    files=[]
    for day in dlist:
        fend=day.strftime("%Y%m%d") + "*.nc"
        files.extend(glob(fpre+fend))

    # also get zero hour of next day:
    nextday=dlist[-1] + timedelta(days=1)
    fend2=nextday.strftime("%Y%m%d0000") + ".nc"


    files.extend(glob(fpre+fend2)) # add final hour
    files.sort() # make sure they're sorted or the data gets read in poorly
    #print(files)
    # and remove the zero hour of the first day (avg of prior hour)
    # if it is a zero hour from prior day
    if '0000.nc' in files[0]:
        del files[0]

    with xarray.open_mfdataset(files) as ds:
        data,attrs=dataset_to_dicts(ds,['ISOP_BIOG'])

    mod_times=[]
    for p in files:
        mod_times.append(time.ctime(os.path.getmtime(p)))
    data['modification_times']=np.array(mod_times)
    attrs['modification_times']={'desc':'When was file last modified'}

    return data,attrs

def read_Hemco_diags(d0,d1=None, month=False, new_emissions=False):
    '''
        Read Hemco diag output, one day or month at a time
    '''
    if d1 is None:
        d1 = datetime(d0.year,12,31)
    if month:
        d1 = util.last_day(d0)

    # maybe want more than 1 year of data
    fpre='Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_'
    if new_emissions:
        fpre='Data/GC_Output/new_emissions/diagnostics/emiss_a_'
    ylist=util.list_years(d0,d1)
    # for each day: glob matching files to list
    files=[]
    for y in ylist:
        fend=y.strftime("%Y") + ".nc"
        files.extend(glob(fpre+fend))

    files.sort() # make sure they're sorted or the data gets read in poorly

    with xarray.open_mfdataset(files) as ds:
        data,attrs=dataset_to_dicts(ds,['ISOP_BIOG'])

    mod_times=[]
    for p in files:
        mod_times.append(time.ctime(os.path.getmtime(p)))
    data['modification_times']=np.array(mod_times)
    attrs['modification_times']={'desc':'When was file last modified'}

    # subset to requested dates
    dates = util.datetimes_from_np_datetime64(data['time'])
    # subtract an hour so that index represents following hour rather than prior hour
    dates = np.squeeze([d-timedelta(hours=1) for d in dates])

    di = util.date_index(d0, dates, d1)
    #print(di)

    # and store as datetime object
    data['dates'] = np.squeeze(np.array(dates)[di])
    data['time'] = np.squeeze(data['time'][di])
    attrs['dates'] = {'desc':'python datetime objects converted from np.datetime64 "time" dim'}
    #print('DEBUG: ', np.shape(data['ISOP_BIOG']))
    data['ISOP_BIOG'] = np.squeeze(data['ISOP_BIOG'][di])
    #print('DEBUG2: ',np.shape(data['ISOP_BIOG']))
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

    # How to check fields
    sat_fold='Data/GC_Output/geos5_2x25_tropchem/satellite_output/'
    sat_file=sat_fold+'ts_satellite_omi.20050101.bpch'
    tracinf=sat_fold+'tracerinfo.dat'
    diaginf=sat_fold+'diaginfo.dat'

    ds=xbpch.open_bpchdataset(sat_file,tracerinfo_file=tracinf,diaginfo_file=diaginf)
    ds.keys()
