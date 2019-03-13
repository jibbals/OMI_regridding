# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslade April 2016

Reads OMI datasets (level 2 and level 2 gridded)
Reads regridded and reprocessed datasets
Reads AURA Fire datasets (MYD14C8H: 8 day averages)

Writes HDF5 files (used to create regridded and reprocessed datasets)

TODO: combine with GC_fio.py
'''

### Modules ###

# plotting module, and something to prevent using displays(can save output but not display it)

# module for hdf eos 5
import h5py
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import csv
from os.path import isfile
import os.path, time # for modification times
import timeit # to check how long stuff takes
import warnings

# for paralellel file reading
import concurrent.futures

# interpolation method for ND arrays
# todo: remove once this is ported to reprocess.py
from scipy.interpolate import griddata
import xarray
import pandas as pd

from utilities.GMAO import __LATRES__
from utilities.GMAO import __LONRES__
import utilities.utilities as util

###############
### GLOBALS ###
###############
#just_global_things, good hashtag
datafieldsg = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/'
geofieldsg  = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Geolocation Fields/'
datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

__VERBOSE__=False

__Thresh_NO2_d__ = 2e15 # daily threshhold for anthro filter
__Thresh_NO2_y__ = 1.5e15 # yearly avg threshhold
__Thresh_AAOD__  = 0.03 # AAOD smoke threshhold
__Thresh_fires__ = 1 # active fire pixel count threshhold


# TODO set these up and use
__dir_fire__='Data/MOD14A1_D_FIRE/'
__dir_anthro__='Data/OMNO2d/'
__dir_smoke__='Data/OMAERUVd/'

__GCHCHO_KEYS__ = [
    'LONGITUDE','LATITUDE',
    'VCHCHO',           # molecs/m2
    'NHCHO',            # molecs/m3 profile
    'NAIR',
    'SHAPEZ',           # 1/m
    'SHAPESIGMA',       # unitless
    'PMIDS',            # geometric pressure mids (hPa)
    'PEDGES',           # pressure edges hPa
    'SIGMA',            # Sigma dimension
    'BOXHEIGHTS']       # box heights (m)

# Coords for omhchorp:
__OMHCHORP_COORDS__=[
                     'latitude','longitude',
                     ]

# Keys for omhchorp:
__OMHCHORP_KEYS__ = [
    'gridentries',   # how many satellite pixels make up the pixel
    'ppentries',     # how many pixels we got the PP_AMF for
    'RSC',           # The reference sector correction [rsc_lats, 60,3]
    'RSC_latitude',  # latitudes of RSC
    'RSC_region',    # RSC region [S,W,N,E]
    'RSC_GC',        # GEOS-Chem RSC [RSC_latitude] (molec/cm2)
    'VCC_GC',           # The vertical column corrected using the RSC
    'VCC_PP',        # Corrected Paul Palmer VC
    'AMF_GC',        # AMF calculated using by GEOS-Chem
    'AMF_GCz',       # secondary way of calculating AMF with GC
    'AMF_OMI',       # AMF from OMI swaths
    'AMF_PP',        # AMF calculated using Paul palmers code
    'SC',            # Slant Columns
    'VC_GC',         # GEOS-Chem Vertical Columns
    'VC_OMI',        # OMI VCs
    'VC_PP',         # VCs from PP amf
    'VCC_OMI',       # OMI VCCs from original satellite swath outputs
    'VCC_OMI_newrsc', # OMI VCCs using original VC_OMI and new RSC corrections
    'col_uncertainty_OMI',
    #'fires',         # Fire count
    #'AAOD',          # AAOD from omaeruvd
    #'firemask',      # two days prior and adjacent fire activity
    #'smokemask',     # aaod over threshhold (0.03)
    #'anthromask',    # true if no2 for the year is over 1.5e15, or no2 on the day is over 1e15
    ]
    #'fire_mask_8',   # true where fires occurred over last 8 days
    #'fire_mask_16' ] # true where fires occurred over last 16 days

# attributes for omhchorp
__OMHCHORP_ATTRS__ = {
    'gridentries':          {'desc':'satellite pixels averaged per gridbox'},
    'ppentries':            {'desc':'PP_AMF values averaged per gridbox'},
    'VC_OMI':               {'units':'molec/cm2',
                             'desc':'regridded OMI swathe VC'},
    'VC_GC':                {'units':'molec/cm2',
                             'desc':'regridded VC, using OMI SC recalculated using GEOSChem shapefactor'},
    'VC_PP':                {'units':'molec/cm2',
                             'desc':'regridded VC, using OMI SC recalculated using PP AMF'},
    'SC':                   {'units':'molec/cm2',
                             'desc':'OMI slant colums'},
    'VCC_GC':               {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using GEOS-Chem shape factor and reference sector correction'},
    'VCC_PP':               {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using PPalmer and LSurl\'s lidort/GEOS-Chem based AMF'},
    'VCC_OMI':              {'units':'molec/cm2',
                             'desc':'OMI\'s RSC corrected VC'},
    'VCC_OMI_newrsc':       {'units':'molec/cm2',
                             'desc':'OMI\'s VC, using new GEOS-Chem RSC corrections'},
    'RSC':                  {'units':'molec/cm2',
                             'desc':'GEOS-Chem/OMI based Reference Sector Correction: is applied to pixels based on latitude and track number. Third dimension is for AMF applied using [OMI, GC, PP] calculations'},
    'RSC_latitude':         {'units':'degrees',
                             'desc':'latitude centres for RSC'},
    'RSC_GC':               {'units':'molec/cm2',
                             'desc':'GEOS-Chem HCHO over reference sector (monthly avg, interp to 500 lats)'},
    'col_uncertainty_OMI':  {'units':'molec/cm2',
                             'desc':'mean OMI pixel uncertainty'},
    'AMF_GC':               {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_OMI':              {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_PP':               {'desc':'AMF based on PPalmer code using OMI and GEOS-Chem'},
    #'fire_mask_16':         {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current or prior 8 day block"},
    #'fire_mask_8':          {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current 8 day block"},
    #'fires':                {'desc':"daily gridded fire count from AQUA/TERRA"},
    #'AAOD':                 {'desc':'daily smoke AAOD_500nm from AURA (OMAERUVd)'},
    #'firemask':             {'desc':'fire mask using two days prior and adjacent fire activity'},
    #'smokemask':            {'desc':'aaod over threshhold (0.03)'},
    #'anthromask':           {'desc':'true (1) if no2 for the year is over 1.5e15, or no2 on the day is over 1e15'}
    }

###############
### METHODS ###
###############

def read_csv_p(filename):
    '''
    Read csv with pandas

    returns dataframe

    To get array from dataframe: df.values
    '''

    print("Reading %s"%filename)

    return pd.read_csv(filename)


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

def read_csv(filename, delimiter=',', hasheader=True):
    '''
        read a csv into a structure
        headerline is nth line read as the names for the columns
    '''
    print("Reading %s"%filename)
    #data=np.genfromtxt(filename, delimiter=delimiter, names=hasheader)

    ret={}
    with open(filename) as csvfile:
        reader=csv.DictReader(csvfile)

        for i,row in enumerate(reader):
            #print(i, row)
            # headerline is titles:
            for k in row.keys():
                if i == 0:
                    ret[k]=[]
                ret[k].append(row[k])

    return ret

def read_netcdf(filename):
    '''
        read all of some netcdf file...
    '''
    print("Trying to read netcdf file: %s"%filename)
    data,attrs={},{}

    with xarray.open_dataset(filename) as ds: # read netcdf datafile

        # First read coordinates:
        for key in ds.coords.keys():
            data[key]=np.array(ds.coords[key]) # could also keep xarray or dask array
            attrs[key]=ds[key].attrs

        # then read variables
        for key in ds.variables.keys():
            data[key]=np.array(ds[key])
            attrs[key]=ds[key].attrs
            if 'scale' in attrs[key].keys():
                data[key] = data[key]*float(attrs[key]['scale'])
                if __VERBOSE__:
                    print("%s scaled by %.2e"%(key,float(attrs[key]['scale'])))

    return data,attrs

def read_hdf5(filename):
    '''
        Should be able to read hdf5 files created by my method above...
        Returns data dictionary and attributes dictionary
    '''
    retstruct={}
    retattrs={}
    with h5py.File(filename,'r') as in_f:
        if __VERBOSE__:
            print('reading from file '+filename)

        # READ DATA AND ATTRIBUTES:
        for key in in_f.keys():
            if __VERBOSE__: print(key)
            retstruct[key]=in_f[key].value
            attrs=in_f[key].attrs
            retattrs[key]={}
            # print the attributes
            for akey,val in attrs.items():
                if __VERBOSE__: print("reading %s(attr)   %s:%s"%(key,akey,val))
                retattrs[key][akey]=val

        # ADD FILE ATTRIBUTES TO ATTRS
        retattrs['file']={}
        for fkey in in_f.attrs.keys():
            retattrs['file'][fkey] = in_f.attrs[fkey]

    return retstruct, retattrs



def determine_filepath(date, latres=__LATRES__,lonres=__LONRES__, omhcho=False, gridded=False, regridded=False, reprocessed=False, geoschem=False, metaData=False):
    '''
    Make filename based on date, resolution, and type of file.
    '''

    # if not created by me just return the filepath(s) using date variable and glob
    if gridded:
        return glob('Data/omhchog/OMI-Aura*%4dm%02d%02d*.he5'%(date.year, date.month, date.day))[0]
    if metaData:
        return ('Data/omhchorp/metadata/metadata_%s.he5'%(date.strftime('%Y%m%d')))
    if omhcho:
        return glob(date.strftime('Data/omhcho/%Y/OMI-Aura_L2-OMHCHO_%Ym%m%d*'))

    # geos chem output created via idl scripts match the following
    if geoschem:
        return ('Data/gchcho/hcho_%4d%2d.he5'%(date.year,date.month))


    res_date='%1.2fx%1.2f_%4d%02d%02d'%(latres,lonres,date.year,date.month,date.day)

    if reprocessed:
        fpath="Data/omhchorp/omhchorp_%s.he5"%res_date
        return date.strftime(fpath)

    # reprocessed and regridded match the following:
    if regridded:
       fpath="Data/omhchorg/omhcho_1g%s.he5"%res_date

    return(fpath)

def read_CPC_temp(d0, dn=None, regrid=True):
    '''
        Read CPC temperature, pull out date (or date range) and interpolate to GMAO grid
        Returns Data[dates,lats,lons], dates, lats, lons
    '''
    fname = d0.strftime('Data/CPC_Temperatures/tmax.%Y.nc')
    data,attrs=read_netcdf(fname)

    # strings to datetimes
    dates = util.datetimes_from_np_datetime64(data['time'])
    # the dates which we want
    dinds = util.date_index(d0, dates, dn=dn)

    # lats top to bottom (360 of them)
    # 89.75, 89.25, ... -89.75
    lats=data['lat']
    # lons 0 to 360 (720 of them)
    # 0.25, 0.75, ..., 359.75
    lons=data['lon']
    # tmax is in variable 'tmax'
    tmax = data['tmax'][dinds,:,:]
    dates= np.array(dates)[dinds]

    # For any basemap application this works better with -180 to 180 grid
    lons[lons>180] = lons[lons>180] - 360
    #lons=np.roll(lons,360)
    #if len(dinds) < 2:
        #tmax=np.roll(tmax,360,axis=0)
    #else:
        #tmax=np.roll(tmax,360,axis=1)

    out=np.copy(tmax)

    # Regrid onto gmao higher resolution
    if regrid:
        newlats,newlons, _late,_lone=util.lat_lon_grid()
        out=np.zeros([len(dinds),len(newlats),len(newlons)])+np.NaN
        if dn is None:
            out = util.regrid_to_higher(tmax,lats,lons, newlats,newlons)
        else:
            for i in range(len(dinds)):
                out[i] = util.regrid_to_higher(tmax[i],lats,lons, newlats,newlons)
        lats=newlats
        lons=newlons
    return out, dates, lats, lons

def read_AAOD(date):
    '''
        Read OMAERUVd 1x1 degree resolution for a particular date
    '''
    fname=date.strftime('OMI-Aura_L3-OMAERUVd_%Ym%m%d*.he5')
    fpaths=glob('Data/OMAERUVd/'+fname)

    # Seems that the 1x1 grid orientation is as follows:
    lats=np.linspace(-89.5,89.5,180)
    lons=np.linspace(-179.5,179.5,360)
    # Field names of desired fields:
    # AAOD
    field_aaod500 = '/HDFEOS/GRIDS/Aerosol NearUV Grid/Data Fields/FinalAerosolAbsOpticalDepth500'

    # handle missing days of data.
    if len(fpaths)==0:
        print("WARNING: %s does not exist!!!!"%fname)
        print("WARNING:     continuing with nans for %s"%date.strftime("%Y%m%d"))
        return np.zeros([len(lats),len(lons)])+np.NaN, lats,lons

    fpath=fpaths[0]

    if __VERBOSE__:
        print("Reading AAOD from ",fpath)

    # read he5 file...
    with h5py.File(fpath,'r') as in_f:
        ## get data arrays
        aaod  = in_f[field_aaod500].value     #[ 180, 360 ]
    aaod[aaod<0] = np.NaN
    return aaod,lats,lons

def read_AAOD_interpolated(date, latres=__LATRES__,lonres=__LONRES__):
    '''
        Read OMAERUVd interpolated to a lat/lon grid
    '''
    newlats,newlons,newlats_e,newlons_e= util.lat_lon_grid(latres,lonres,regular=False)
    aaod,lats,lons=read_AAOD(date)

    newaaod=util.regrid(aaod,lats,lons,newlats,newlons)
    return newaaod,newlats,newlons

def read_smoke(d0,dN, latres=__LATRES__, lonres=__LONRES__):
    '''
        Read AAOD interpolated over some date dimension
        returns AAOD, dates,lats,lons
    '''
    dates=util.list_days(d0,dN,month=False)
    smoke0,lats,lons=read_AAOD_interpolated(date=d0,latres=latres,lonres=lonres)


    retsmoke=np.zeros([len(dates),len(lats),len(lons)])
    retsmoke[0] = smoke0
    if len(dates)>1:
        for i,day in enumerate(dates[1:]):
            smokei,lats,lons=read_AAOD_interpolated(date=day,latres=latres,lonres=lonres)
            retsmoke[i+1] = smokei

    return retsmoke, dates, lats, lons



def read_MOD14A1(date=datetime(2005,1,1), per_km2=False):
    '''
        Read the modis product of firepix/1000km2/day

        Returns firepix/km2/day or firepix/day
    '''
    # lats are from top to bottom when read using pandas
    # lons are from left to right
    lats=np.linspace(89.9,-89.9,1799)
    lons=np.linspace(-180,179.9,3600)

    # file looks like:
    #'Data/MOD14A1_D_FIRE/2005/MOD14A1_D_FIRE_2005-01-02.CSV'
    fpath='Data/MOD14A1_D_FIRE/'+date.strftime('%Y/MOD14A1_D_FIRE_%Y-%m-%d.CSV')
    if not isfile(fpath):
        print("WARNING: %s does not exist!!!!"%fpath)
        print("WARNING:     continuing with nans for %s"%date.strftime("%Y%m%d"))
        return np.zeros([len(lats),len(lons)])+np.NaN, lats,lons
    if __VERBOSE__:
        print("Reading ",fpath)

    start=timeit.default_timer()

    fires=pd.read_csv(fpath).values

    fires[fires>9000] = 0. # np.NaN # ocean squares
    fires[fires==0.1] = 0. # land squares but no fire I think...



    if not per_km2:
        # area per gridbox in km2:
        area=util.area_grid(lats,lons)
        fires=fires * 1e3 * area # now fire_pixels/day

    if __VERBOSE__:
        print("TIMEIT: it took %6.2f seconds to read the day of fire"%(timeit.default_timer()-start))

    return fires,lats,lons


def read_MOD14A1_interpolated(date=datetime(2005,1,1), latres=__LATRES__,lonres=__LONRES__):
    '''
        Read firepixels/day from MOD14A1 daily gridded product
        returns fires, lats, lons
    '''
    newlats,newlons,_nlate,_nlone= util.lat_lon_grid(latres=latres,lonres=lonres,regular=False)
    fires,lats,lons=read_MOD14A1(date,per_km2=False)

    # can skip regridding if file is missing...
    if np.all(np.isnan(fires)):
        return np.zeros([len(newlats),len(newlons)]), newlats, newlons

    newfires=util.regrid_to_lower(fires,lats,lons,newlats,newlons,np.nansum)
    return newfires, newlats, newlons

def read_fires(d0, dN, latres=__LATRES__, lonres=__LONRES__,max_procs=1):
    '''
        Read fires from MOD14A1 into a time,lat,lon array
        Returns Fires[dates,lats,lons], dates,lats,lons
        try to use multiple processors if nprocesses>1
    '''
    dates=util.list_days(d0,dN,month=False)
    fire0,lats,lons=read_MOD14A1_interpolated(date=d0,latres=latres,lonres=lonres)


    retfires=np.zeros([len(dates),len(lats),len(lons)])
    retfires[0] = fire0
    if len(dates) > 1:
        # Can use multiple processes
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

    return retfires, dates, lats, lons


def read_E_new_month(month=datetime(2005,1,1), oneday=None, filename=None):
    '''
    Function to read the recalculated Emissions output
    Inputs:
        oneday = None : read a single day - set this arg to a datetime to do this
        filename=None : if set then read this file ( used for testing )
    Output:
        Structure containing E_new dataset
    '''

    # First get filename:
    dstr=month.strftime(format='%Y%m')
    fpath='Data/Isop/E_new/emissions_%s.h5'%dstr

    if filename is not None:
        fpath=filename

    if __VERBOSE__:
        print("Reading E_new: %s"%fpath)

    datastruct, attributes=read_hdf5(fpath)

    # get datetimes from yyyymmdd ints
    dates=[datetime.strptime(str(d),'%Y%m%d') for d in datastruct['time']]
    datastruct['dates']=dates

    if oneday is not None:
        dind=np.where(dates==oneday)[0]
        for key in datastruct.keys():
            # Date is first dimension, so easy to pull out one day
            datastruct[key]=datastruct[key][dind]

    return datastruct, attributes




def read_omhcho(path, szamax=60, screen=[-5e15, 1e17], maxlat=60, verbose=False):
    '''
    Read info from a single swath file
    NANify entries with main quality flag not equal to zero
    NANify entries where xtrackqualityflags aren't zero
    Argument based filters:
        NANify entries outside of screen (argument)
        NANify entries with abs(latitude) > maxlat
        NANify entries with sza > szamax
    Returns:{'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
            'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
            'rad_ref_col',
            'qualityflag':qf, 'xtrackflag':xqf,
            'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}
    '''



    # Total column amounts are in molecules/cm2
    field_hcho  = datafields+'ColumnAmount'
    field_ref_c = datafields+'RadianceReferenceColumnAmount'
    # other useful fields
    field_amf   = datafields+'AirMassFactor'
    field_amfg  = datafields+'AirMassFactorGeometric'
    field_apri  = datafields+'GasProfile'
    field_plevs = datafields+'ClimatologyLevels'
    field_w     = datafields+'ScatteringWeights'
    field_qf    = datafields+'MainDataQualityFlag'
    field_clouds= datafields+'AMFCloudFraction'
    field_ctp   = datafields+'AMFCloudPressure'
    field_rsc   = datafields+'ReferenceSectorCorrectedVerticalColumn' # molec/cm2
    field_xqf   = geofields +'XtrackQualityFlags'
    field_lon   = geofields +'Longitude'
    field_lat   = geofields +'Latitude'
    field_sza   = geofields +'SolarZenithAngle'
    field_vza   = geofields +'ViewingZenithAngle'
    # uncertainty flags
    field_colUnc    = datafields+'ColumnUncertainty' # also molec/cm2
    field_fitflag   = datafields+'FitConvergenceFlag'
    field_fitRMS    = datafields+'FittingRMS'

    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        lats    = in_f[field_lat].value     #[ 1644, 60 ]
        lons    = in_f[field_lon].value     #
        hcho    = in_f[field_hcho].value    #
        VCC_OMI = in_f[field_rsc].value     # ref sector corrected vc
        amf     = in_f[field_amf].value     #
        amfg    = in_f[field_amfg].value    # geometric amf
        clouds  = in_f[field_clouds].value  # cloud fraction
        ctp     = in_f[field_ctp].value     # cloud top pressure
        qf      = in_f[field_qf].value      #
        xqf     = in_f[field_xqf].value     # cross track flag
        sza     = in_f[field_sza].value     # solar zenith angle
        vza     = in_f[field_vza].value     # viewing zenith angle

        # uncertainty arrays                #
        cunc    = in_f[field_colUnc].value  # uncertainty
        fcf     = in_f[field_fitflag].value # convergence flag
        frms    = in_f[field_fitRMS].value  # fitting rms
        #                                   # [ 47, 1644, 60 ]
        w       = in_f[field_w].value       # scattering weights
        apri    = in_f[field_apri].value    # apriori
        plevs   = in_f[field_plevs].value   # pressure dim
        #                                   # [ 60 ]
        ref_c   = in_f[field_ref_c].value   # reference radiance col for swath tracks

        #
        ## remove missing values and bad flags:
        # QF: missing<0, suss=1, bad=2
        if verbose:
            print("%d pixels in %s prior to filtering"%(np.sum(~np.isnan(hcho)),path))
        suss       = qf != 0
        if verbose:
            print("%d pixels removed by main quality flag"%np.nansum(suss))
        hcho[suss] = np.NaN
        lats[suss] = np.NaN
        lons[suss] = np.NaN
        amf[suss]  = np.NaN

        # remove xtrack flagged data
        xsuss       = xqf != 0
        if verbose:
            removedcount= np.nansum(xsuss+suss) - np.nansum(suss)
            print("%d further pixels removed by xtrack flag"%removedcount)
        hcho[xsuss] = np.NaN
        lats[xsuss] = np.NaN
        lons[xsuss] = np.NaN
        amf[xsuss]  = np.NaN

        # remove pixels polewards of maxlat
        with np.errstate(invalid='ignore'):
            rmlat   = np.abs(lats) > maxlat
        if verbose:
            removedcount=np.nansum(rmlat+xsuss+suss) - np.nansum(xsuss+suss)
            print("%d further pixels removed as |latitude| > 60"%removedcount)
        hcho[rmlat] = np.NaN
        lats[rmlat] = np.NaN
        lons[rmlat] = np.NaN
        amf[rmlat]  = np.NaN

        # remove solarzenithangle over 60 degrees
        rmsza       = sza > szamax
        if verbose:
            removedcount= np.nansum(rmsza+rmlat+xsuss+suss) - np.nansum(rmlat+xsuss+suss)
            print("%d further pixels removed as sza > 60"%removedcount)
        hcho[rmsza] = np.NaN
        lats[rmsza] = np.NaN
        lons[rmsza] = np.NaN
        amf[rmsza]  = np.NaN

        # remove VCs outside screen range
        if screen is not None:
            # ignore warnings from comparing NaNs to Values
            with np.errstate(invalid='ignore'):
                rmscr   = (hcho<screen[0]) + (hcho>screen[1]) # A or B
            if verbose:
                removedcount= np.nansum(rmscr+rmsza+rmlat+xsuss+suss)-np.nansum(rmsza+rmlat+xsuss+suss)
                print("%d further pixels removed as value is outside of screen"%removedcount)
            hcho[rmscr] = np.NaN
            lats[rmscr] = np.NaN
            lons[rmscr] = np.NaN
            amf[rmscr]  = np.NaN

    #return everything in a structure
    return {'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
            'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
            'rad_ref_col':ref_c, 'VCC_OMI':VCC_OMI, 'ctp':ctp,
            'qualityflag':qf, 'xtrackflag':xqf, 'sza':sza, 'vza':vza,
            'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}

def read_omhcho_day(day=datetime(2005,1,1),verbose=False, max_procs=4):
    '''
    Read an entire day of omhcho swaths
    '''
    fnames=determine_filepath(day,omhcho=True)
    if len(fnames)==0:
        print("WARNING: OMHCHO missing for %s"%day.strftime("%Y%m%d"))
        ret = { retkey:[np.NaN] for retkey in ['HCHO','lats','lons','AMF','AMFG',
                'apriori','cloudfrac','rad_ref_col','VCC_OMI','ctp',
                'qualityflag','xtrackflag','sza', 'vza','coluncertainty',
                'convergenceflag','fittingRMS'] }
        ret['omega']=[np.zeros([47])+np.NaN]
        ret['plevels']=[np.zeros([47])+np.NaN]
        return ret

    data=read_omhcho(fnames[0],verbose=verbose) # read first swath
    swths=[]
    if max_procs < 2:
        for fname in fnames[1:]: # read the rest of the swaths
            swths.append(read_omhcho(fname,verbose=verbose))
    else: # use multiple processes
        n = len(fnames[1:])
        nverbose=[verbose] * n # list of n instances of verbose
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            procreturns=executor.map(read_omhcho,fnames[1:],nverbose)
            for pret in procreturns:
                swths.append(pret)

    # combine each swath into single structure listing pixel details
    for swth in swths:
        for key in swth.keys():
            axis= [0,1][key in ['omega','apriori','plevels']]
            data[key] = np.concatenate( (data[key], swth[key]), axis=axis)

    return data

def read_omhcho_month(month=datetime(2005,1,1),max_procs=4, keys=None):
    '''
    '''
    if keys is None:
        keys = ['HCHO','AMF','AMFG', 'apriori','cloudfrac',
                'rad_ref_col','VCC_OMI','ctp', 'qualityflag','xtrackflag',
                'sza', 'vza','coluncertainty', 'convergenceflag','fittingRMS',
                'omega', 'plevels','lats','lons']
    keys.extend([])
    dates=util.list_days(month,month=True)
    n=len(dates)
    retstruct={} # structure to hold all the data...
    datas = []
    for d in dates:
        datas.append(read_omhcho_day(d,max_procs=max_procs))

    # Combine datas along new first dimension of time
    for key in keys:
        keyinfo = [datas[i][key] for i in range(n)]
        retstruct[key] = np.stack(keyinfo,axis=0)
    retstruct['dates'] = np.array(dates)
    return retstruct


def read_omhcho_8days(day=datetime(2005,1,1)):
    '''
    Read in 8 days all at once
    '''
    data8=read_omhcho_day(day)
    for date in [ day + timedelta(days=d) for d in range(1,8) ]:
        data=read_omhcho_day(date)
        for key in data.keys():
            axis= [0,1][key in ['omega','apriori','plevels']]
            data8[key] = np.concatenate( (data8[key], data[key]), axis=axis)
    return data8

def read_omhchorp_day(date, latres=__LATRES__, lonres=__LONRES__, keylist=None, filename=None):
    '''
    Function to read a reprocessed omi file, by default reads an 8-day average (8p)
    Inputs:
        date = datetime(y,m,d) of file
        latres=__LATRES__
        lonres=__LONRES__
        keylist=None : if set to a list of strings, just read those data from the file, otherwise read all data
        filename=None : if set then read this file ( used for testing )
    Output:
        Structure containing omhchorp dataset
    '''
    if keylist is None:
        keylist=__OMHCHORP_KEYS__

        # make sure coords are included
    keylist=list(set(keylist+__OMHCHORP_COORDS__))

    if filename is None:
        fpath=determine_filepath(date,latres=latres,lonres=lonres,reprocessed=True)
    else:
        fpath=filename

    with h5py.File(fpath,'r') as in_f:
        #print('reading from file '+fpath)
        if keylist is None:
            keylist=in_f.keys()
        retstruct=dict.fromkeys(keylist)
        for key in keylist:
            try:
                retstruct[key]=in_f[key].value
            except KeyError as ke: # if there is no matching key then print an error and continue
                print("Key Error in %s"%fpath)
                print(ke)
                retstruct[key]=np.NaN

    retstruct['mod_times']= [time.ctime(os.path.getmtime(fpath))]
    return retstruct

def read_omhchorp(day0,dayn=None,keylist=None,latres=__LATRES__,lonres=__LONRES__):
    '''
    '''

    if keylist is None:
        keylist=__OMHCHORP_KEYS__

        # make sure coords are included
    keylist=list(set(keylist+__OMHCHORP_COORDS__))

    # Read the days we want to analyse:
    daylist = util.list_days(day0, dayn) # includes last day.
    nt=len(daylist)
    struct = []
    data={}
    for day in daylist:
        try:
            daystruct=read_omhchorp_day(date=day,
                                        latres=latres, lonres=lonres,
                                        keylist=keylist)
        except Exception as inst:
            print("WARNING: could not read omhchorp: ", day)
            print("       : ",inst)
            struct.append(util.__MISSING_OMHCHORP_DAY__)
            continue

        struct.append(daystruct)

    # Set all the data arrays in the same way, [[time],lat,lon]
    ret_keylist=struct[0].keys()
    for k in ret_keylist:
        if nt ==1: # one day only, no time dimension
            data[k] = np.squeeze(np.array(struct[0][k]))
        else:
            data[k] = np.array([struct[j][k] for j in range(nt)])
        if __VERBOSE__:
            print("Read from omhchorp: ",k, data[k].shape)

    # Reference Sector Correction latitudes don't change with time
    if 'RSC_latitude' in ret_keylist:
        data['RSC_latitude']=struct[0]['RSC_latitude'] # rsc latitude bins
    if 'RSC_region' in ret_keylist:
        data['RSC_region']=struct[0]['RSC_region']

    # Screen the Vert Columns to between these values:
    VC_screen=[-5e15,1e17]
    # Already screened when reading OMHCHO before recreating PP and GC AMFs
    data['VC_screen']=VC_screen
    #for vcstr in ['VCC_OMI','VCC_PP','VCC_GC']:
    #    if vcstr not in data.keys():
    #        continue
    #    attr=data[vcstr]

    #    screened=(attr<VC_screen[0]) + (attr>VC_screen[1])
    #    scrstr="[%.1e - %.1e]"%(VC_screen[0], VC_screen[1])
    #    print("Removing %d gridsquares from %s using screen %s"%(np.sum(screened), vcstr, scrstr))

    #    attr[screened]=np.NaN
    #    #TODO: Also update pixel counts...
    #    data[vcstr]= attr

    # Change latitude to lats...
    data['lats']=struct[0]['latitude']
    data['lons']=struct[0]['longitude']
    data.pop('latitude') # remove latitudes
    data.pop('longitude') # remove latitudes

    return data


def read_gchcho(date):
    '''
    Read the geos chem hcho column data into a dictionary
    '''
    #This one is the old files
    #fpath=glob('Data/gchcho/hcho_%4d%02d.he5' % ( date.year, date.month ) )[0]
    fpath=glob('Data/gchcho/ucx_shapefactor_%4d%02d.he5'%(date.year,date.month) )[0]
    ret_data={}
    with h5py.File(fpath, 'r') as in_f:
        dsetname='GC_UCX_HCHOColumns'
        dset=in_f[dsetname]

        for key in __GCHCHO_KEYS__:
            ret_data[key] = dset[key].squeeze()
    return ret_data

def read_omno2d_interpolated(date,latres=__LATRES__,lonres=__LONRES__):
    ''' Read one day of OMNO2d data'''

    #vcdname='HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2'
    #vcdcsname='HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2CloudScreened'
    tropname='HDFEOS/GRIDS/ColumnAmountNO2/Data Fields/ColumnAmountNO2TropCloudScreened'
    #latname='HDFEOS/GRIDS/ColumnAmountNO2/lat'
    #lonname='HDFEOS/GRIDS/ColumnAmountNO2/lon'
    ddir=__dir_anthro__+'data/'

    data=np.zeros([720,1440])+np.NaN
    # OMNO2d dataset has these lats/lons
    #   0.25x0.25 horizontal resolution into 720 lats x 1440 lons
    #   full coverage implies
    #   lats: -89.875 ... dy=0.25 ... 89.875
    #   lons: -179.875 ... dx=0.25 ... 179.875
    lats=np.arange(-90,90,0.25)+0.125
    lons=np.arange(-180,180,0.25)+0.125
    newlats,newlons, lats_e, lons_e=util.lat_lon_grid(latres,lonres)

    # filenames:
    fname=ddir+'OMI*%s*.he5'%date.strftime('%Ym%m%d')
    fpaths=glob(fname)
    if len(fpaths)==0:
        print("WARNING: %s does not exist!!!!"%fname)
        print("WARNING:     continuing with nans for %s"%date.strftime("%Y%m%d"))
        newdata=np.zeros([len(newlats),len(newlons)])+np.NaN
    else:
        fpath=fpaths[0]
        if __VERBOSE__:
            print('reading ',fpath)
            start=timeit.default_timer()
        with h5py.File(fpath,'r') as in_f:
            #for name in in_f[tropname]:
            #    print (name)
            trop=in_f[tropname].value
            trop[trop<-1e20] = np.NaN

            data=trop
        data=np.squeeze(data)

        if __VERBOSE__:
            print('done reading ',date.strftime('%Ym%m%d'),' took %6.2f seconds'%(timeit.default_timer()-start))
        # now we interpolate to resolution desired
        newdata=util.regrid(data,lats,lons,newlats,newlons)

    # trop column units: molec/cm2 from ~ 1e13-1e16
    attrs={'tropno2':{'desc':'tropospheric NO2 cloud screened for <30% cloudy pixels',
                      'units':'molec/cm2'},
           'lats':{'desc':'latitude midpoints'},
           'lons':{'desc':'longitude midpoints'},}
    ret={'tropno2':newdata,'lats':newlats,'lons':newlons,'lats_e':lats_e,'lons_e':lons_e}
    return ret,attrs


def read_omno2d(day0,dayN=None,latres=__LATRES__,lonres=__LONRES__, max_procs=1):
    '''
        Read daily gridded OMNO2d data
    '''

    dates=util.list_days(day0,dayN)

    lats=None
    lons=None
    lats_e=None
    lons_e=None
    no2=None
    # Can use multiple processes
    if max_procs>1:
        nlatres=[latres]*len(dates)
        nlonres=[lonres]*len(dates)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
            procreturns=executor.map(read_omno2d_interpolated,dates,nlatres,nlonres)
            # loop over returned dictionaries from read_omno2d()
            for ii, pret in enumerate(procreturns):
                omno2 = pret[0]['tropno2']
                if no2 is None:
                    no2=np.zeros([len(dates),omno2.shape[0],omno2.shape[1]]) # [dates, lats, lons]
                no2[ii]=omno2
                if lats is None:
                    lats=pret[0]['lats']
                if lons is None:
                    lons=pret[0]['lons']
                if lons_e is None:
                    lons_e=pret[0]['lons_e']
                if lats_e is None:
                    lats_e=pret[0]['lats_e']
    else:
        for ii, date in enumerate(dates):
            data,attrs = read_omno2d_interpolated(date,latres,lonres)
            omno2=data['tropno2']
            if no2 is None:
                no2=np.zeros([len(dates),omno2.shape[0],omno2.shape[1]]) # [dates, lats, lons]
            no2[ii] = omno2
            if lats is None:
                lats=data['lats']
            if lons is None:
                lons=data['lons']
            if lons_e is None:
                lons_e=data['lons_e']
            if lats_e is None:
                lats_e=data['lats_e']


    # trop column units: molec/cm2 from ~ 1e13-1e16
    attrs={'tropno2':{'desc':'tropospheric NO2 cloud screened for <30% cloudy pixels',
                      'units':'molec/cm2'},
           'lats':{'desc':'latitude midpoints'},
           'lons':{'desc':'longitude midpoints'},
           'dates':{'desc':'datetimes for each day of averaged pixels'}}
    ret={'tropno2':no2,'lats':lats,'lons':lons,'lats_e':lats_e,'lons_e':lons_e,'dates':dates}
    return ret,attrs

def read_omno2d_year(year=2005, d0=None, d1=None, region=None):
    '''
        Read anthromask_year.h5 file
            can subset to d0-d1, and/or region
        return no2, dates, lats, lons
    '''
    subsettime=True
    if d0 is None:
        d0=datetime(year,1,1)
        d1=datetime(year,12,31)
        subsettime=False


    # read file for anthro mask:
    path=__dir_anthro__ + 'anthromask_%4d.h5'%year
    assert isfile(path), 'NEED TO RUN make_anthro_mask_file(datetime(%4d,1,1))'%year
    data,attrs=read_hdf5(path)
    no2=data['no2']
    lats=data['lats']
    lons=data['lons']
    if np.all(data['dates']==0):
        dates = np.array(util.list_days(datetime(d0.year,1,1),datetime(d0.year,12,31)))
    else:
        dates = util.date_from_gregorian(data['dates'])


    # subset to desired time/area
    di = np.ones(len(dates)).astype(bool)
    if subsettime:
        di=util.date_index(d0,dates,d1) # indices of dates from d0 to dN
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[no2],has_time_dim=True)
        no2=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']

    #print(di, dates, len(di),len(dates))
    #print(type(di),type(no2),no2.shape, type(dates))
    return np.squeeze(no2[di,:,:]),np.squeeze(dates[di]),lats,lons



#def yearly_anthro_avg(date,latres=__LATRES__,lonres=__LONRES__,region=None,max_procs=4):
#    '''
#        Read and save the yearly avg no2 product at natural resolution unless it is already saved
#        read it, regrid it, subset it, and return it
#    '''
#    year=date.year
#
#    filepath=__dir_anthro__+'yearavg_%4d.h5'%year
#    # Read and save file if it doesn't exist yet:
#    if not isfile(filepath):
#        if __VERBOSE__:
#            print("Reading OMNO2d %d then writing year avg to %s"%(year,filepath))
#        y0=datetime(year,1,1)
#        yN=util.last_day(datetime(year,12,1))
#
#        # Read whole year of omno2d and save to a yearly file
#        omno2, omno2_attrs = read_omno2d(day0=y0, dayN=yN, month=False,max_procs=max_procs)
#
#        # Average over the year:
#        omno2['tropno2'] = np.nanmean(omno2['tropno2'],axis=0)
#        # remove dates arrays
#        omno2.pop('dates')
#        omno2_attrs.pop('dates')
#
#        # save the file:
#        save_to_hdf5(filepath, omno2,attrdicts=omno2_attrs)
#    else:
#        omno2, omno2_attrs = read_hdf5(filepath)
#
#    # omno2d is saved on 0.25x0.25 grid, regrid_to_lower should always be the choice
#    newlats,newlons,_late,_lone=util.lat_lon_grid(latres=latres,lonres=lonres)
#    lats,lons=omno2['lats'],omno2['lons']
#    no2=util.regrid_to_lower(omno2['tropno2'],lats,lons,newlats,newlons)
#
#    if region is not None:
#        subset=util.lat_lon_subset(newlats,newlons,region,[no2])
#        newlats=subset['lats']
#        newlons=subset['lons']
#        no2=subset['data'][0]
#
#    return no2,newlats,newlons

def read_slopes(d0=datetime(2005,1,1),dN=datetime(2012,12,1)):
    '''
        Monthly slopes are calculated and saved to slopes.h5, 
            These can be read by this function
    '''
    # keys which can be date subsetted
    dkeys=['ci','ci_sf','n','n_sf','r','r_sf','slope','slope_sf']

    # slope is monthly, using first day of month
    d0=util.first_day(d0)
    if dN is not None:
        dN = util.first_day(dN)

    data,attrs=read_hdf5('Data/GC_Output/slopes.h5')
    dates=util.date_from_gregorian(data['dates'])
    data['dates']=dates
    di=util.date_index(d0,dates,dN)

    if len(di) != len(data['dates']):
        #subset to requested date(s)
        for key in dkeys:
            data[key] = np.squeeze(data[key][di])
        data['dates'] = data['dates'][di] # don't squeeze dates, arr len of 1 is good
    return data, attrs

def get_slope(month,monthN=None):
    '''
        Read GC biogenic slope H=SE+b
        for month or multiple months
        use smear filtered slope
            if r<0.4 or n<10 use multiyearaverage
    '''
    sloped,slopea=read_slopes(month,monthN)

    # smear filtered slope:
    slope=sloped['slope_sf']
    lats=sloped['lats']
    lons=sloped['lons']
    dates=sloped['dates']
    print(type(dates),dates)
    r=sloped['r_sf']
    rmya=sloped['r_sf_mya']
    mya=sloped['slope_sf_mya']


    # ignore warning from comparing NaNs to number
    with np.errstate(invalid='ignore'):
        # remove negatives from mya
        mya[mya<0] = np.NaN
        # if r for mya is < 0.4, set to NaN
        mya[rmya<0.4] = np.NaN

    n=sloped['n_sf']

    if len(dates) == 1:
        # use multiyear avg where r is too low
        myai = dates[0].month -1
        if __VERBOSE__:
            print('check single month slope creation', slope.shape, mya.shape, myai)
        test=np.copy(slope)
        # ignore warning from comparing NaNs to number
        with np.errstate(invalid='ignore'):
            slope[r<0.4] = mya[myai][r<0.4]

            nans=np.isnan(test*slope)
            assert any(test[~nans] != slope[~nans]), 'no changes!'

            # also where count is too low
            slope[n<10] = mya[myai][n<10]

            # replace negatives with mya also
            slope[slope<0] = mya[myai][slope<0]

    # if we have multiple months then do it monthly
    else:
        for i,m in enumerate(dates):
            myai = m.month-1 # month index for mya
            nm = n[i] # this months counts
            rm = r[i] # this months regression coefficients
            sm = slope[i]

            # use multiyear avg where r is too low
            if __VERBOSE__:
                print('check multimonth slope creation', slope.shape, slope[i].shape, rm.shape, mya.shape, myai)
            test=np.copy(slope[i])

            with np.errstate(invalid='ignore'):
                sm[rm<0.4] = mya[myai][rm<0.4]
                nans=np.isnan(test*slope[i])
                assert any(test[~nans] != slope[i][~nans]), 'no changes!'

                # also where count is too low
                sm[nm<10] = mya[myai][nm<10]

                # replace negatives with mya also
                sm[sm<0] = mya[myai][sm<0]


    return slope,dates,lats,lons


def filter_high_latitudes(array, lats, has_time_dim=False, highest_lat=60.0):
    '''
    Read an array, assuming globally gridded at latres/lonres, set the values polewards of 60 degrees
    to nans
    '''

    highlats= np.where(np.abs(lats) > highest_lat)
    newarr=np.array(array)
    if has_time_dim:
        newarr[:,highlats,:] = np.NaN
    else:
        newarr[highlats, :] = np.NaN
    return (newarr)

def read_AMF_pp(date=datetime(2005,1,1),troprun=True):
    '''
    Read AMF created by randal martin code for this day
    along with pixel index for adding data to the good pixel list
    '''
    import os.path
    runstr=['ucxrunpathgoeshere','tropchem'][troprun]
    dstr=date.strftime('%Y%m%d')
    fname='Data/pp_amf/%s/amf_%s.csv'%(runstr,dstr)
    if not isfile(fname):
        print("WARNING: %s does not exist!!!!"%fname)
        print("WARNING:     continuing with nans for %s"%date.strftime("%Y%m%d"))
        return None, None
    #assert os.path.isfile(fname), "ERROR: file missing: %s"%fname

    inds=[]
    amfs=[]
    with open(fname,'r') as f:
        for line in f.readlines():
            s=line.split(',')
            inds.append(int(s[1]))
            amfs.append(float(s[2]))
    amfs=np.array(amfs)
    if __VERBOSE__:
        print ("%d of the PP_AMFs are < 0 on %s"%(np.sum(amfs<0),dstr))
        print("mean PP_AMF from %s = %f"%(fname,np.nanmean(amfs[amfs>0])))
    amfs[amfs<0.0001] = np.NaN # convert missing numbers or zeros to NaN
    amfs = list(amfs)
    return inds, amfs

def read_GC_output(date=datetime(2005,1,1), Isop=False,
    UCX=False, oneday=False, monthavg=False, surface=False):
    '''
        Wrapper for reading GC_output, requires Data/GC_fio.py
        Inputs:
            date: date of retrieval
            Isop: retrieve the Isop data along with the HCHO data
            UCX: get the UCX monthly data
            oneday: retrieve only one day (tropchem only)
            monthavg: take the average of the month (tropchem only)
            surface: Only take the surface level of data
    '''
    from Data.GC_fio import get_tropchem_data, get_UCX_data, UCX_HCHO_keys, tropchem_HCHO_keys, UCX_Isop_keys, tropchem_Isop_keys
    keys=[tropchem_HCHO_keys, UCX_HCHO_keys][UCX]
    if Isop:
        keys=[tropchem_Isop_keys, UCX_Isop_keys][UCX]

    data={}
    if UCX:
        data=get_UCX_data(date,keys=keys,surface=surface)
    else:
        data=get_tropchem_data(date,keys=keys,monthavg=monthavg,surface=surface)
    return data
##############
## Making masks
## Cant use omhchorp since omhchorp uses these to make the masks the first time
##############

def make_anthro_mask_file(year,
                          threshy=__Thresh_NO2_y__, threshd=__Thresh_NO2_d__,
                          latres=__LATRES__, lonres=__LONRES__,max_procs=4):
    '''
        Create anthro mask file for whole year
    '''

    ## year long list of datetimes
    d0=datetime(year.year,1,1)
    dN=datetime(year.year,12,31)
    dates=util.list_days(d0,dN)

    ## First read in year of satellite data (gridded to latres,lonres)
    #
    omno2, omno2_attrs = read_omno2d(day0=d0, dayN=dN, latres=latres, lonres=lonres, max_procs=max_procs)
    lats = omno2['lats']
    lons = omno2['lons']
    no2  = omno2['tropno2'] # [dates, lats, lons]
    no2mean = np.nanmean(no2,axis=0) # yearly average for threshy

    # Daily filter
    # ignore warning from comparing NaNs to number
    with np.errstate(invalid='ignore'):
        ret = no2 > threshd # day threshold
        yearmask=no2mean>threshy # year threshold

    ret[:,yearmask]=True # mask values where yearly avg threshhold is exceded

    ## to save an HDF we need to change boolean to int8 and dates to strings
    #
    dates=util.gregorian_from_dates(dates)
    anthromask=ret.astype(np.int8)

    ## add attributes to be saved in file
    #
    dattrs  = {'threshy':{'units':'molec/cm2','desc':'Threshold for yearly averaged NO2'},
              'threshd':{'units':'molec/cm2','desc':'Threshold for daily NO2'},
              'latres':{'units':'degrees','desc':'latitude resolution'},
              'lonres':{'units':'degrees','desc':'longitude resolution'},
              'anthromask':{'units':'int','desc':'0 or 1: grid square potentially affected by anthropogenic influence'},
              'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of anthromask array'},
              'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
              'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'},
              'yearavg':{'units':'molec/cm2','desc':'year average of NO2'},
              'no2':{'units':'molec/cm2','desc':'daily NO2 columns regridded'},  }
    ## data dictionary to save to hdf
    #
    datadict={'anthromask':anthromask,'dates':dates,'lats':lats,'lons':lons,
              'yearavg':no2mean, 'no2':no2}
    fattrs={'threshy':threshy, 'threshd':threshd, 'latres':latres, 'lonres':lonres}

    # filename and save to h5 file
    path=year.strftime(__dir_anthro__+'anthromask_%Y.h5')
    save_to_hdf5(path, datadict, attrdicts=dattrs,fattrs=fattrs)




def make_smoke_mask(d0, dN=None, aaod_thresh=__Thresh_AAOD__,
                    latres=__LATRES__, lonres=__LONRES__, region=None):
    '''
        Return smoke mask with dimensions [len(d0-dN), n_lats, n_lons]

        Read OMAERUVd AAOD(500nm), regrid into local resoluion, mask days above thresh

        Takes a couple of seconds to mask a day
    '''
    if dN is None:
        dN = d0

    if __VERBOSE__:
        print("Making smoke mask for %s - %s"%(d0,dN))
        print("Smoke mask being created for any square with aaod > %0.3f"%aaod_thresh)

    # smoke from OMAERUVd
    smoke, dates, lats, lons = read_smoke(d0, dN, latres=latres, lonres=lonres)

    # Return mask for aaod over threshhold
    if region is not None:
        subsets=util.lat_lon_subset(lats,lons,region,[smoke],has_time_dim=True)
        smoke=subsets['data'][0]
        lats,lons=subsets['lats'],subsets['lons']

    # ignore warning from comparing NaNs to number
    with np.errstate(invalid='ignore'):
        mask=smoke>aaod_thresh
    return mask, dates,lats,lons, smoke

def make_smoke_mask_file(year,aaod_thresh=__Thresh_AAOD__,
                         latres=__LATRES__, lonres=__LONRES__,):
    '''
    '''
    ## First make year long filter using method above
    #
    d0=datetime(year.year,1,1)
    dN=datetime(year.year,12,31)
    smokemask,dates,lats,lons, smoke=make_smoke_mask(d0,dN,aaod_thresh=aaod_thresh,
                                            latres=latres,lonres=lonres,
                                            region=None)

    ## to save an HDF we need to change boolean to int8 and dates to strings
    #
    dates=util.gregorian_from_dates(dates)
    smokemask=smokemask.astype(np.int8)

    ## add attributes to be saved in file
    #
    attrs  = {'aaod_thresh':{'units':'AAOD','desc':'aaod threshold for smoke influence'},
              'latres':{'units':'degrees','desc':'latitude resolution in degrees'},
              'lonres':{'units':'degrees','desc':'longitude resolution in degrees'},
              'smokemask':{'units':'int','desc':'0 or 1: grid square potentially affected by smoke'},
              'smoke':{'units':'AAOD','desc':'OMAERUVd AAOD at 550nm'},
              'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of firemask array'},
              'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
              'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'}, }
    ## data dictionary to save to hdf
    #
    datadict={'smokemask':smokemask,'smoke':smoke, 'dates':dates,'lats':lats,'lons':lons,}
    fattrs = {'aaod_thresh':aaod_thresh, 'latres':latres,'lonres':lonres}

    # filename and save to h5 file
    path=year.strftime(__dir_smoke__+'smokemask_%Y.h5')
    save_to_hdf5(path, datadict, attrdicts=attrs, fattrs=fattrs)


def make_fire_mask(d0, dN=None, prior_days_masked=2, fire_thresh=__Thresh_fires__,
                   adjacent=True, latres=__LATRES__, lonres=__LONRES__,
                   region=None, max_procs=1):
    '''
        Return fire mask with dimensions [len(d0-dN), n_lats, n_lons]
        looks at fires between [d0-days_masked+1, dN], for each day in d0 to dN

        mask is true where more than fire_thresh fire pixels exist.

        takes around 500 seconds to mask a day with 2 prior days,
        takes 13 seconds if using omhchorp

    '''
    # first day of filtering
    daylist=util.list_days(d0,dN)
    first_day=daylist[0]-timedelta(days=prior_days_masked)
    last_day=daylist[-1]
    has_time_dim= (len(daylist) > 1) + (prior_days_masked > 0)
    if __VERBOSE__:
        print("VERBOSE: make_fire_mask will return rolling %d day fire masks between "%(prior_days_masked+1), d0, '-', last_day)
        print("VERBOSE: They will filter gridsquares with more than %d fire pixels detected"%fire_thresh)

    # Takes a long time to read and collate all the fires files
    fires, _dates, lats, lons = read_fires(d0=first_day,dN=last_day,
                                           latres=latres,lonres=lonres,
                                           max_procs=max_procs)

    # mask squares with more fire pixels than allowed
    mask = fires>fire_thresh

    retmask= np.zeros([len(daylist),len(lats),len(lons)],dtype=np.bool)

    # actual mask is made up of sums of daily masks over prior days_masked

    if prior_days_masked>0:
        # from end back to first day in daylist
        for i in -np.arange(0,len(daylist)):
            tempmask=mask[i-prior_days_masked-1:] # look at last N days (N is prior days masked)
            if i < 0:
                tempmask=tempmask[:i] # remove days past the 'current'

            # mask is made up from prior N days, working backwards
            retmask[i-1]=np.sum(tempmask,axis=0)

    else:
        retmask = mask
    assert retmask.shape[0]==len(daylist), 'return mask is wrong!'

    # mask adjacent squares also (if desired)
    if adjacent:
        for i in range(len(daylist)):
            retmask[i]=util.set_adjacent_to_true(retmask[i])

    if region is not None:
        subsets=util.lat_lon_subset(lats,lons,region,[retmask,fires],has_time_dim=has_time_dim)
        retmask=subsets['data'][0]
        fires = subsets['data'][1]
        lats=subsets['lats']
        lons=subsets['lons']

    return retmask, daylist, lats,lons, fires

def make_fire_mask_file(year, prior_days_masked=2, fire_thresh=__Thresh_fires__,
                        adjacent=True, latres=__LATRES__,lonres=__LONRES__, max_procs=4):
    '''
        Create fire mask file for year, save into h5 file for re-use

        mask is true where more than fire_thresh fire pixels exist

    '''

    ## First make year long filter using method above
    #
    d0=datetime(year.year,1,1)
    dN=datetime(year.year,12,31)
    firemask,dates,lats,lons, fires = make_fire_mask(d0,dN,prior_days_masked=prior_days_masked,
                                            fire_thresh=fire_thresh, adjacent=adjacent,
                                            latres=latres,lonres=lonres,
                                            region=None,max_procs=max_procs)

    ## to save an HDF we need to change boolean to int8 and dates to strings
    #
    dates=util.gregorian_from_dates(dates)
    firemask=firemask.astype(np.int8)

    ## add attributes to be saved in file
    #
    dattrs  = {'firemask':{'units':'int','desc':'0 or 1: grid square potentially affected by fire'},
              'fires':{'units':'int','desc':'sum of fire detected pixels'},
              'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of firemask array'},
              'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
              'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'}, }

    fattrs = {'fire_thresh':fire_thresh, 'adjacent':np.int8(adjacent), 'latres':latres, 'lonres':lonres,
              'prior_days_masked':prior_days_masked}
    ## data dictionary to save to hdf
    #
    datadict={'firemask':firemask,'fires':fires,'dates':dates,'lats':lats,'lons':lons}

    # filename and save to h5 file
    path=year.strftime(__dir_fire__+'firemask_%Y.h5')
    save_to_hdf5(path, datadict, attrdicts=dattrs, fattrs=fattrs)


def get_fire_mask(d0, dN=None, prior_days_masked=2, fire_thresh=__Thresh_fires__,
                   adjacent=True, latres=__LATRES__,lonres=__LONRES__,
                   region=None):
    ''' read firemask from file or else create one  '''

    # look for file first
    path=__dir_fire__+d0.strftime('firemask_%Y.h5')

    assert isfile(path), "NO FIREMASK FILE CREATED: "+path
    data, attrs = read_hdf5(path)
    # check file uses settings we want
    # for now assume it does..
    #assert (attrs['file']['prior_days_masked'] == prior_days_masked) \
    #    and (attrs['file']['fire_thresh'] == fire_thresh) \
    #    and (attrs['file']['adjacent'] == adjacent) \
    #    and (attrs['file']['latres'] == latres) \
    #    and (attrs['file']['lonres'] == lonres), "FIREMASK ATTRIBUTES ARE UNEXPECTED"

    firemask=data['firemask']
    dates=data['dates']
    dates=util.date_from_gregorian(dates)
    lats=data['lats']
    lons=data['lons']
    # subset to requested dates, lats and lons
    di = util.date_index(d0,dates,dn=dN)
    print(d0, dates[0], dates[-1], dN)

    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[firemask],has_time_dim=True)
        firemask=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']
    print(np.shape(firemask))
    firemask=np.squeeze(firemask[di])
    print(np.shape(firemask))
    dates=np.squeeze(np.array(dates)[di])
    return firemask,dates,lats,lons

def get_anthro_mask(d0,dN,region=None,latres=__LATRES__, lonres=__LONRES__):
    '''
        Read anthro mask from d0 to dN.
            If the mask does not exist, Fail and tell me to make one
    '''
    # read file for anthro mask:
    path=__dir_anthro__ + 'anthromask_%4d.h5'%d0.year
    assert isfile(path), 'NEED TO RUN make_anthro_mask_file(datetime(%4d,1,1))'%d0.year
    data,attrs=read_hdf5(path)
    mask=data['anthromask'].astype(bool)
    lats=data['lats']
    lons=data['lons']
    if np.all(data['dates']==0):
        dates = util.list_days(datetime(d0.year,1,1),datetime(d0.year,12,31))
    else:
        dates = util.date_from_gregorian(data['dates'])


    # subset to desired time/area
    di=util.date_index(d0,dates,dN) # indices of dates from d0 to dN
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[mask],has_time_dim=True)
        mask=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']

    return np.squeeze(mask[di]),np.squeeze(dates[di]),lats,lons


def get_smoke_mask(d0,dN,region=None,latres=__LATRES__, lonres=__LONRES__):
    ''' read smoke mask '''
    # read file for anthro mask:
    path=__dir_smoke__ + 'smokemask_%4d.h5'%d0.year
    assert isfile(path), 'NEED TO RUN make_smoke_mask_file(datetime(%4d,1,1))'%d0.year
    data,attrs=read_hdf5(path)
    mask=data['smokemask'].astype(bool)
    lats=data['lats']
    lons=data['lons']
    dates=util.date_from_gregorian(data['dates'])

    # subset to desired time/area
    di=util.date_index(d0,dates,dN) # indices of dates from d0 to dN
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[mask],has_time_dim=True)
        mask=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']

    return np.squeeze(mask[di]),np.squeeze(dates[di]), lats,lons

#########################
###  READ MUMBA FROM JENNY
########################
def read_mumba_var(varname):
    # MUMBA directory
    mumba_dir = 'Data/campaigns/MUMBA/'

    def mumba_hdr(fname):
        # Return header lines for file
        switcher = {
            "MUMBA_PTRMS_2012-12-21_2013-02-15.tab"   : 25,
            "MUMBA_NOx_UOW_2012-11-21_2013-02-15.tab" : 18,
            "MUMBA_O3_2012-12-21_2013-02-15.tab"      : 20,
            "MUMBA_MET_2012-12-21_2013-01-25.tab"     : 18,
        }
        return switcher.get(fname,"error")

    def get_mumba_fname(varname):
        """Uses a GEOS-Chem species name to pick MUMBA filename"""
        # Default return original name if not found (may be special case)
        switcher = {
            "CH2O"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "MOH"     : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "ALD2"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "ACET"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "ISOP"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "MVK_MACR": "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "BENZ"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "TOLU"    : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "MONOT"   : "MUMBA_PTRMS_2012-12-21_2013-02-15.tab",
            "NO"      : "MUMBA_NOx_UOW_2012-11-21_2013-02-15.tab",
            "NO2"     : "MUMBA_NOx_UOW_2012-11-21_2013-02-15.tab",
            "NOX"     : "MUMBA_NOx_UOW_2012-11-21_2013-02-15.tab",
            "O3"      : "MUMBA_O3_2012-12-21_2013-02-15.tab",
            "TMPU"    : "MUMBA_MET_2012-12-21_2013-01-25.tab",
        }
        return switcher.get(varname.upper(), "error")

    # Get filename
    fname = get_mumba_fname(varname)

    # Error if this is not a MUMBA species
    if fname == "error":
       raise KeyError()

    # Use filename to get header info
    n_hdr = mumba_hdr(fname)

    # Read using pandas
    df = pd.read_csv(mumba_dir+fname,sep='\t',header=n_hdr,
                     index_col=[0],parse_dates=True)

    # Replace missing value with NaN
    df = df.resample('60min').mean()

    return df

