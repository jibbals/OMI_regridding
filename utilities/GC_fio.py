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
#from classes.GC_class import __coords__ as GC_coords
GC_coords = ['lev','lon','lat','time']
sys.path.pop(0)


##################
#####GLOBALS######
##################

__VERBOSE__=False

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

def read_Hemco_diags(d0,d1=None,month=False):
    '''
        Read Hemco diag output, one day or month at a time
    '''
    # match all files with YYYYMM[DD] tag
    fpre='Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.'
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

    #    # Handle if we have too many files
    #    # split into lists of length 30, to each be read and combined afterwards
    #    sfiles=util.list_to_lists(files,30)
    #    datalist=[]
    #    attrslist=[]
    #
    #    for i,filelist in enumerate(sfiles):
    #        with xarray.open_mfdataset(filelist) as ds:
    #            datai,attrsi=dataset_to_dicts(ds,['ISOP_BIOG'])
    #            datalist.append(datai)
    #            attrslist.append(attrsi)
    #    # now combine the lists extending the time dimension
    with xarray.open_mfdataset(files) as ds:
        data,attrs=dataset_to_dicts(ds,['ISOP_BIOG'])

    mod_times=[]
    for p in files:
        mod_times.append(time.ctime(os.path.getmtime(p)))
    data['modification_times']=np.array(mod_times)
    attrs['modification_times']={'desc':'When was file last modified'}

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


def hcho_lifetime(month):
    '''
    Use tau = HCHO / Loss    to look at hcho lifetime over a month
    return lifetimes[day,lat,lon],
    '''

    d0=util.first_day(month)
    dN=util.last_day(month)

    # read hcho and losses from trac_avg
    # ch20 in ppbv, air density in molec/cm3,  Loss HCHO mol/cm3/s
    keys=['IJ-AVG-$_CH2O','BXHGHT-$_N(AIR)', 'PORL-L=$_LHCHO']
    run = GC_class.GC_tavg(d0,dN, keys=keys) # [time, lat, lon, lev]
    # TODO: Instead of surface use tropospheric average??
    hcho = run.hcho[:,:,:,0] # [time, lat, lon, lev=47] @ ppbv
    N_air = run.N_air[:,:,:,0] # [time, lat, lon, lev=47] @ molec/cm3
    Lhcho = run.Lhcho[:,:,:,0] # [time, lat, lon, lev=38] @ mol/cm3/s  == molec/cm3/s !!!!

    # [ppbv * 1e-9 * (molec air / cm3) / (molec/cm3/s)] = s
    tau = hcho * 1e-9 * N_air  /  Lhcho

    # change to hours
    tau=tau/3600.
    #

    return tau, run.dates, run.lats, run.lons

def make_smear_mask_file(year, use_GC_lifetime=True, max_procs=4):
    '''
        Estimate expected yield (assuming HCHO lifetime=2.5 or using GC hcho loss to approximate)
        determine bounds for acceptable smearing range
        create a 3d mask for the year of smearing min and max values
        saves smear_mask.nc: lifetimes, yields, smearmin, smearmax, smear, smearmask
    '''

    # first read year of GC lifetimes
    if max_procs>1:
        ndates=dates[1:]
        n = len(ndates)
        nlatres=[latres] * n # list with n instances of latres
        nlonres=[lonres] * n # also for lonres
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            procreturns=executor.map(read_MOD14A1_interpolated,ndates,nlatres,nlonres)
            for i,pret in enumerate(procreturns):
                retfires[i+1]=pret[0]
    else:
        for i,day in enumerate(dates[1:]):
            firei,lats,lons=read_MOD14A1_interpolated(date=day,latres=latres,lonres=lonres)
            retfires[i+1] = firei



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
