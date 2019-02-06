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
import h5py
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
import os.path, time # for file modification time

# Add parent folder to path
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#sys.path.insert(0,os.path.dirname(currentdir))

import utilities.utilities as util
from utilities import GMAO

#from classes.GC_class import __coords__ as GC_coords
GC_coords = ['lev','lon','lat','time']
#sys.path.pop(0)


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

# COPIED FROM fio.py 6/2/19
def save_to_hdf5(outfilename, arraydict, fillvalue=np.NaN,
                 attrdicts={}, fattrs={},
                 verbose=False):
    '''
        Takes a bunch of arrays, named in the arraydict parameter, and saves
        to outfilename as hdf5 using h5py (with fillvalue specified), and gzip compression

        INPUTS:
            outfilename: name of file to save
            arraydict: named arrays of data to save using given fillvalue and attributes
            attrdicts is an optional dictionary of dictionaries,
                keys should match arraydict, values should be dicts of attributes
            fattrs: extra file level attributes
    '''
    print("saving "+outfilename)
    with h5py.File(outfilename,"w") as f:
        if verbose:
            print("Inside fio.save_to_hdf5()")
            print(arraydict.keys())

        # attribute creation
        # give the HDF5 root some more attributes
        f.attrs['Filename']        = outfilename.split('/')[-1]
        f.attrs['creator']          = 'fio.py, Jesse Greenslade'
        f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        f.attrs['h5py_version']     = h5py.version.version
        f.attrs['Fill_Value']       = fillvalue
        # optional extra file attributes from argument
        for key in fattrs.keys():
            if verbose:
                print(key,fattrs[key], type(fattrs[key]))
            f.attrs[key] = fattrs[key]


        for name in arraydict.keys():
            # create dataset, using arraydict values
            darr=np.array(arraydict[name])
            if verbose:
                print((name, darr.shape, darr.dtype))

            # handle datetime type and boolean types
            # turn boolean into int8
            if darr.dtype == np.dtype('bool'):
                darr=np.int8(darr)
                attrdicts={}
                if not name in attrdicts:
                    attrdicts[name]={}
                attrdicts[name]['conversion']={'from':'boolean','to':'int8','by':'fio.save_to_hdf5()'}
            # harder to test for datetime type...

            # Fill array using darr,
            #
            dset=f.create_dataset(name,fillvalue=fillvalue,
                                  data=darr, compression_opts=9,
                                  chunks=True, compression="gzip")
            # for VC items and RSC, note the units in the file.
            if name in attrdicts:
                for attrk, attrv in attrdicts[name].items():
                    dset.attrs[attrk]=attrv

        # force h5py to flush buffers to disk
        f.flush()
    print("Saved "+outfilename)

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

def make_overpass_output_files(run='tropchem'):
    '''
        Read GEOS-Chem satellite output for tropchem and new_emissions runs
            create one he5 with all the data from 2005-201305
        runs: 'tropchem' | 'new_emissions'
    '''
    print("Making overpass file for ",run)
    outfilename='Data/GC_Output/overpass_%s.h5'%run
    if run=='new_emissions':
        bpch = 'new_emissions/satellite_output/ts_satellite_altered.%s.bpch'
    elif run=='tropchem':
        bpch = 'geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s.bpch'
    else:
        assert False, 'run type not handled yet:'+run
    fpre='Data/GC_Output/%s'%bpch
    # FOR TESTING JUST DO 2005,2006
    firstday=datetime(2005,1,1)
    lastday = datetime(2006,2,2)
    years=util.list_years(firstday,lastday)
    
    #for folder in 
    # match all files with YYYYMM[DD] tag
    #folder=['geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog',
    #        'new_emissions/diagnostics/emiss_a'][new_emissions]
    keys_to_keep=[  'IJ-AVG-$NO', #    ppbv   144  91  47
                    'IJ-AVG-$O3', #    ppbv   144  91  47
                    'IJ-AVG-$ISOP', #  ppbC   144  91  47
                    'IJ-AVG-$CH2O',  #           ppbv   144  91  47
                    'IJ-AVG-$NO2',  #           ppbv   144  91  47
                    'PEDGE-$PSURF',   #            hPa   144  91  47
                    #DAO-FLDS   20    PARDF   #           W/m2   144  91  47
                    #DAO-FLDS   20    PARDR   #           W/m2   144  91  47
                    'DAO-FLDS_TS',   #              K   144  91  47
                    'DAO-3D-$TMPU',   #              K   144  91  47
                    #CHEM-L=$   20  #     OH  16001       molec/cm3 175536.25 2005011000  144  91  47
                    #TIME-SER   20            #      UNDEFINED 1  144  91  47
                    #TIME-SER   20            #      molec/cm3   144  91  47
                    #TIME-SER   20            #          m2/m2   144  91  47
                    #BIOGSRCE   20     ISOP   #    atomC/cm2/s   144  91  47
                    'BXHGHT-$BXHEIGHT',   #              m   144  91  47
                    'TR-PAUSE', #          level   144  91  47
                    ]
    
    
    all_data={}
    for year in years:
        print("Reading year ",year.year)
        d0=datetime(year.year,1,1)
        d1=datetime(year.year,12,31)
        if year.year==2013:
            d1=datetime(2013,5,31) # special case in 2013
        dlist=util.list_days(d0,d1)
        print(dlist)
        print(d0,d1)
        # file names have date strings in name
        files=[]
        for day in dlist:
            fend=day.strftime("%Y%m%d")
            files.extend(glob(fpre%fend))

        files.sort() # make sure they're sorted or the data gets read in poorly
    
        print('check overpass files being read (show 1 in 24):', files[::24])
        
        # now read the data from all those files
        data,attrs=read_bpch(files,keys_to_keep)
        
        if year.year == years[0].year:
            all_data = data
        else:
            for key in keys_to_keep:
                print('appending all_data[%s] '%key,np.shape(all_data[key]), ' with data[%s] '%key, np.shape(data[key]))
                all_data[key] = np.append(all_data[key], data[key], axis=0)
                print('    -> ',np.shape(all_data[key]))
    
    all_data['dates'] = util.gregorian_from_dates(util.list_days(firstday,lastday))
    attrs['dates'] = {'desc':'gregorian dates: hours since 1985,1,1,0,0'}
    
    save_to_hdf5(outfilename, all_data, fillvalue=np.NaN,
                 attrdicts=attrs, fattrs={}, verbose=True)
    


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

    ds=open_bpchdataset(sat_file,tracerinfo_file=tracinf,diaginfo_file=diaginf)
    ds.keys()
