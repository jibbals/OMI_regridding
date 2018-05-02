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
# interpolation method for ND arrays
# todo: remove once this is ported to reprocess.py
from scipy.interpolate import griddata
import xarray
import pandas as pd

import utilities.utilities as util

###############
### GLOBALS ###
###############
#just_global_things, good hashtag
datafieldsg = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/'
geofieldsg  = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Geolocation Fields/'
datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

__VERBOSE__=True



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
    'RSC',           # The reference sector correction [rsc_lats, 60]
    'RSC_latitude',  # latitudes of RSC
    'RSC_region',    # RSC region [S,W,N,E]
    'RSC_GC',        # GEOS-Chem RSC [RSC_latitude] (molec/cm2)
    'VCC',           # The vertical column corrected using the RSC
    'VCC_PP',        # Corrected Paul Palmer VC
    'AMF_GC',        # AMF calculated using by GEOS-Chem
    'AMF_GCz',       # secondary way of calculating AMF with GC
    'AMF_OMI',       # AMF from OMI swaths
    'AMF_PP',        # AMF calculated using Paul palmers code
    'SC',            # Slant Columns
    'VC_GC',         # GEOS-Chem Vertical Columns
    'VC_OMI',        # OMI VCs
    'VC_OMI_RSC',    # OMI VCs with Reference sector correction?
    'col_uncertainty_OMI',
    'fires',         # Fire count
    'AAOD',          # AAOD from omaeruvd
    'firemask',      # two days prior and adjacent fire activity
    'smokemask',     # aaod over threshhold (0.03)
    'anthromask',    # true if no2 for the year is over 1.5e15, or no2 on the day is over 1e15
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
                             'desc':'regridded VC, using OMI SC recalculated using GEOSChem shape factor'},
    'SC':                   {'units':'molec/cm2',
                             'desc':'OMI slant colums'},
    'VCC':                  {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using GEOS-Chem shape factor and reference sector correction'},
    'VCC_PP':               {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using PPalmer and LSurl\'s lidort/GEOS-Chem based AMF'},
    'VC_OMI_RSC':           {'units':'molec/cm2',
                             'desc':'OMI\'s RSC corrected VC '},
    'RSC':                  {'units':'molec/cm2',
                             'desc':'GEOS-Chem/OMI based Reference Sector Correction: is applied to pixels based on latitude and track number'},
    'RSC_latitude':         {'units':'degrees',
                             'desc':'latitude centres for RSC'},
    'RSC_GC':               {'units':'molec/cm2',
                             'desc':'GEOS-Chem HCHO over reference sector'},
    'col_uncertainty_OMI':  {'units':'molec/cm2',
                             'desc':'mean OMI pixel uncertainty'},
    'AMF_GC':               {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_OMI':              {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_PP':               {'desc':'AMF based on PPalmer code using OMI and GEOS-Chem'},
    #'fire_mask_16':         {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current or prior 8 day block"},
    #'fire_mask_8':          {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current 8 day block"},
    'fires':                {'desc':"daily gridded fire count from AQUA/TERRA"},
    'AAOD':                 {'desc':'daily smoke AAOD_500nm from AURA (OMAERUVd)'},
    'firemask':             {'desc':'fire mask using two days prior and adjacent fire activity'},
    'smokemask':            {'desc':'aaod over threshhold (0.03)'},
    'anthromask':           {'desc':'true (1) if no2 for the year is over 1.5e15, or no2 on the day is over 1e15'}
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
            f.attrs[key] = fattrs[key]


        for name in arraydict.keys():
            # create dataset, using arraydict values
            darr=arraydict[name]
            if verbose:
                print((name, darr.shape, darr.dtype))

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
    from netCDF4 import Dataset

    out={}
    with Dataset(filename,'r') as nc_f: # read netcdf datafile
        nc_attrs=nc_f.ncattrs()
        nc_dims= [dim for dim in nc_f.dimensions]
        nc_vars= [var for var in nc_f.variables]
        for var in nc_vars:
            #print( var, nc_fid.variables[var].size )
            out[var]=nc_f.variables[var][:]
    return out

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
        #attrs=in_f.attrs

        for key in in_f.keys():
            if __VERBOSE__: print(key)
            retstruct[key]=in_f[key].value
            attrs=in_f[key].attrs
            retattrs[key]={}
            # print the attributes
            for akey,val in attrs.items():
                if __VERBOSE__: print("%s(attr)   %s:%s"%(key,akey,val))
                retattrs[key][akey]=val

    return retstruct, retattrs



def determine_filepath(date, latres=0.25,lonres=0.3125, gridded=False, regridded=False, reprocessed=False, geoschem=False, metaData=False):
    '''
    Make filename based on date, resolution, and type of file.
    '''

    # if not created by me just return the filepath(s) using date variable and glob
    if gridded:
        return glob('Data/omhchog/OMI-Aura*%4dm%02d%02d*.he5'%(date.year, date.month, date.day))[0]
    if metaData:
        return ('Data/omhchorp/metadata/metadata_%s.he5'%(date.strftime('%Y%m%d')))
    if not (regridded or reprocessed):
        return glob(date.strftime('Data/omhcho/%Y/OMI-Aura_L2-OMHCHO_%Ym%m%d*'))

    # geos chem output created via idl scripts match the following
    if geoschem:
        return ('Data/gchcho/hcho_%4d%2d.he5'%(date.year,date.month))

    # reprocessed and regridded match the following:
    avg='1' # one or 8 day average when applicable
    typ=['p','g'][regridded] # reprocessed or regridded
    res='%1.2fx%1.2f'%(latres,lonres) # resolution string
    d = 'Data/omhchor'+typ+'/' # directory
    fpath=d+"omhcho_%s_%4d%02d%02d.he5" %(avg+typ+res,date.year,date.month,date.day)
    return(fpath)

def read_AAOD(date):
    '''
        Read OMAERUVd 1x1 degree resolution for a particular date
    '''
    fpath=glob('Data/OMAERUVd/'+date.strftime('OMI-Aura_L3-OMAERUVd_%Ym%m%d*.he5'))[0]

    if __VERBOSE__:
        print("Reading AAOD from ",fpath)

    # Seems that the 1x1 grid orientation is as follows:
    lats=np.linspace(-89.5,89.5,180)
    lons=np.linspace(-179.5,179.5,360)

    # Field names of desired fields:
    # AAOD
    field_aaod500 = '/HDFEOS/GRIDS/Aerosol NearUV Grid/Data Fields/FinalAerosolAbsOpticalDepth500'

    # read he5 file...
    with h5py.File(fpath,'r') as in_f:
        ## get data arrays
        aaod  = in_f[field_aaod500].value     #[ 180, 360 ]
    aaod[aaod<0] = np.NaN
    return aaod,lats,lons

def read_AAOD_interpolated(date, latres=0.25,lonres=0.3125):
    '''
        Read OMAERUVd interpolated to a lat/lon grid
    '''
    newlats,newlons,newlats_e,newlons_e= util.lat_lon_grid(latres,lonres)
    aaod,lats,lons=read_AAOD(date)

    newaaod=util.regrid(aaod,lats,lons,newlats,newlons)
    return newaaod,newlats,newlons

def read_smoke(d0,dN, latres=0.25, lonres=0.3125):
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

def make_smoke_mask(d0, dN=None, aaod_thresh=0.05,
                    latres=0.25, lonres=0.3125, region=None):
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

    return smoke>aaod_thresh, dates,lats,lons

def read_MOD14A1(date=datetime(2005,1,1), per_km2=False):
    '''
        Read the modis product of firepix/1000km2/day

        Returns firepix/km2/day or firepix/day
    '''

    # file looks like:
    #'Data/MOD14A1_D_FIRE/2005/MOD14A1_D_FIRE_2005-01-02.CSV'
    fpath='Data/MOD14A1_D_FIRE/'+date.strftime('%Y/MOD14A1_D_FIRE_%Y-%m-%d.CSV')
    if __VERBOSE__:
        print("Reading ",fpath)
    fires=pd.read_csv(fpath).values

    fires[fires>9000] = 0. # np.NaN # ocean squares
    fires[fires==0.1] = 0. # land squares but no fire I think...

    lats=np.linspace(89.9,-89.9,1799) # lats are from top to bottom when read using pandas
    lons=np.linspace(-180,179.9,3600) # lons are from left to right

    if not per_km2:
        # area per gridbox in km2:
        area=util.area_grid(lats,lons)
        fires=fires * 1e3 * area # now fire_pixels/day

    return fires,lats,lons


def read_MOD14A1_interpolated(date=datetime(2005,1,1), latres=0.25,lonres=0.3125):
    '''
        Read firepixels/day from MOD14A1 daily gridded product
        returns fires, lats, lons
    '''
    newlats,newlons,_nlate,_nlone= util.lat_lon_grid(latres=latres,lonres=lonres)
    fires,lats,lons=read_MOD14A1(date,per_km2=False)

    newfires=util.regrid_to_lower(fires,lats,lons,newlats,newlons,np.nansum)
    return newfires,newlats,newlons

def read_fires(d0, dN, latres=0.25, lonres=0.3125):
    '''
        Read fires from MOD14A1 into a time,lat,lon array
        Returns Fires[dates,lats,lons], dates,lats,lons
    '''
    dates=util.list_days(d0,dN,month=False)
    fire0,lats,lons=read_MOD14A1_interpolated(date=d0,latres=latres,lonres=lonres)


    retfires=np.zeros([len(dates),len(lats),len(lons)])
    retfires[0] = fire0
    if len(dates) > 1:
        for i,day in enumerate(dates[1:]):
            firei,lats,lons=read_MOD14A1_interpolated(date=day,latres=latres,lonres=lonres)
            retfires[i+1] = firei

    return retfires, dates, lats, lons

def make_fire_mask(d0, dN=None, prior_days_masked=2, fire_thresh=1,
                   adjacent=True,
                   latres=0.25,lonres=0.3125, region=None):
    '''
        Return fire mask with dimensions [len(d0-dN), n_lats, n_lons]
        looks at fires between [d0-days_masked+1, dN], for each day in d0 to dN

        mask is true where more than fire_thresh fire pixels exist.

        takes around 13 seconds to mask a day with 2 prior days

    '''
    # first day of filtering
    daylist=util.list_days(d0,dN)
    first_day=daylist[0]-timedelta(days=prior_days_masked)
    last_day=daylist[-1]
    has_time_dim= (len(daylist) > 1) + (prior_days_masked > 0)
    if __VERBOSE__:
        print("VERBOSE: make_fire_mask will return rolling %d day fire masks between "%(prior_days_masked+1), d0, '-', last_day)
        print("VERBOSE: They will filter gridsquares with more than %d fire pixels detected"%fire_thresh)
        print("VERBOSE: fire mask will be read from omhchorp now.")

    # read fires[dates,lats,lons]
    # Takes too long!
    #fires, _dates, lats, lons = read_fires(d0=first_day,dN=last_day,
    #                                       latres=latres,lonres=lonres)
    om=read_omhchorp(first_day,last_day,keylist=['fires'],latres=latres,lonres=lonres)
    fires=om['fires']
    lats,lons = om['lats'], om['lons']

    # mask squares with more fire pixels than allowed
    mask = fires>fire_thresh

    retmask= np.zeros([len(daylist),len(lats),len(lons)]).astype(np.bool)

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
        subsets=util.lat_lon_subset(lats,lons,region,[retmask],has_time_dim=has_time_dim)
        retmask=subsets['data'][0]
        lats=subsets['lats']
        lons=subsets['lons']

    return retmask, daylist, lats,lons

# Fire code is for old fire product
#def read_8dayfire(date=datetime(2005,1,1,0)):
#    '''
#    Read 8 day fire average file
#    This function determines the filepath based on date input
#    '''
#    # filenames are all like *yyyyddd.h5, where ddd is day of the year(DOY), one for every 8 days
#    tt = date.timetuple()
#
#    # only every 8 days matches a file
#    # this will give us a multiple of 8 which matches our DOY
#    daymatch= int(np.floor(tt.tm_yday/8)*8) +1
#    filepath='Data/MYD14C8H/MYD14C8H.%4d%03d.h5' % (tt.tm_year, daymatch)
#    return read_8dayfire_path(filepath)
#
#def read_8dayfire_path(path):
#    '''
#    Read fires file using given path
#    '''
#    ## Fields to be read:
#    # Count of fires in each grid box over 8 days
#    corrFirePix = 'CorrFirePix'
#    #cloudCorrFirePix = 'CloudCorrFirePix'
#
#    ## read in file:
#    with h5py.File(path,'r') as in_f:
#        ## get data arrays
#        cfp     = in_f[corrFirePix].value
#        #ccfp    = in_f[cloudCorrFirePix].value
#
#    # from document at
#    # http://www.fao.org/fileadmin/templates/gfims/docs/MODIS_Fire_Users_Guide_2.4.pdf
#    # latitude = 89.75 - 0.5 * y
#    # longitude = -179.75 + 0.5 * x
#    lats    = np.arange(90,-90,-0.5) - 0.25
#    lons    = np.arange(-180,180, 0.5) + 0.25
#    return (cfp, lats, lons)
#
#def read_8dayfire_interpolated(date,latres,lonres):
#    '''
#    Read the date, interpolate data to match lat/lon resolution, return data
#    '''
#    ##original lat/lons:
#    fires, lats, lons = read_8dayfire(date)
#    #lats = np.arange(90,-90,-0.5) - 0.25
#    #lons = np.arange(-180,180, 0.5) + 0.25
#
#    newlats= np.arange(-90,90, latres) + latres/2.0
#    newlons= np.arange(-180,180, lonres) + lonres/2.0
#
#    mlons,mlats = np.meshgrid(lons,lats)
#    mnewlons,mnewlats = np.meshgrid(newlons,newlats)
#
#    interp = griddata( (mlats.ravel(), mlons.ravel()), fires.ravel(), (mnewlats, mnewlons), method='nearest')
#    return interp, newlats, newlons

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
        rsc_omi = in_f[field_rsc].value     # ref sector corrected vc
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
            'rad_ref_col':ref_c, 'RSC_OMI':rsc_omi, 'ctp':ctp,
            'qualityflag':qf, 'xtrackflag':xqf, 'sza':sza, 'vza':vza,
            'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}

def read_omhcho_day(day=datetime(2005,1,1),verbose=False):
    '''
    Read an entire day of omhcho swaths
    '''
    fnames=determine_filepath(day)
    data=read_omhcho(fnames[0],verbose=verbose) # read first swath
    swths=[]
    for fname in fnames[1:]: # read the rest of the swaths
        swths.append(read_omhcho(fname,verbose=verbose))
    for swth in swths: # combine into one struct
        for key in swth.keys():
            axis= [0,1][key in ['omega','apriori','plevels']]
            data[key] = np.concatenate( (data[key], swth[key]), axis=axis)
    return data

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

def read_omhchorp_day(date, latres=0.25, lonres=0.3125, keylist=None, filename=None):
    '''
    Function to read a reprocessed omi file, by default reads an 8-day average (8p)
    Inputs:
        date = datetime(y,m,d) of file
        latres=0.25
        lonres=0.3125
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
    return retstruct

def read_omhchorp(day0,dayn,keylist=None,latres=0.25,lonres=0.3125):
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
        struct.append(read_omhchorp_day(date=day,
                                        latres=latres, lonres=lonres,
                                        keylist=keylist))

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
    data['VC_screen']=VC_screen
    for vcstr in ['VC_OMI_RSC','VCC_PP','VCC']:
        if vcstr not in data.keys():
            continue
        attr=data[vcstr]

        screened=(attr<VC_screen[0]) + (attr>VC_screen[1])
        scrstr="[%.1e - %.1e]"%(VC_screen[0], VC_screen[1])
        print("Removing %d gridsquares from %s using screen %s"%(np.sum(screened), vcstr, scrstr))

        attr[screened]=np.NaN
        data[vcstr]= attr

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

def read_omno2d(day0,dayN=None,month=False):
    '''
        Read daily gridded OMNO2d data, optionally with time dimension
    '''
    # set dayN if not set
    if dayN is None:
        dayN=[day0, util.last_day(day0)][month]

    #vcdname='HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2'
    #vcdcsname='HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2CloudScreened'
    tropname='HDFEOS/GRIDS/ColumnAmountNO2/Data Fields/ColumnAmountNO2TropCloudScreened'
    #latname='HDFEOS/GRIDS/ColumnAmountNO2/lat'
    #lonname='HDFEOS/GRIDS/ColumnAmountNO2/lon'
    ddir='Data/OMNO2d/data/'

    dates=util.list_days(day0,dayN)
    data=np.zeros([len(dates),720,1440])+np.NaN
    lats=np.arange(-90,90,0.25)+0.125
    lons=np.arange(-180,180,0.25)+0.125
    lats_e=np.arange(-180,180.001,0.25)
    lons_e=np.arange(-90,90.001,0.25)
    # filenames:
    for ii, date in enumerate(dates):
        fname=ddir+'OMI*%s*.he5'%date.strftime('%Ym%m%d')
        fpath=glob(fname)[0]
        if __VERBOSE__:
            print('reading ',fpath)
        with h5py.File(fpath,'r') as in_f:
            #for name in in_f[tropname]:
            #    print (name)
            trop=in_f[tropname].value
            trop[trop<-1e20] = np.NaN

            data[ii]=trop

    data=np.squeeze(data)

    # trop column units: molec/cm2 from ~ 1e13-1e16
    attrs={'tropno2':{'desc':'tropospheric NO2 cloud screened for <30% cloudy pixels',
                      'units':'molec/cm2'},
           'lats':{'desc':'latitude midpoints'},
           'lons':{'desc':'longitude midpoints'},
           'dates':{'desc':'datetimes for each day of averaged pixels'}}
    ret={'tropno2':data,'lats':lats,'lons':lons,'lats_e':lats_e,'lons_e':lons_e,'dates':dates}
    return ret,attrs

def yearly_anthro_avg(date,latres=0.25,lonres=0.3125,region=None):
    '''
        Read and save the yearly avg no2 product at natural resolution unless it is already saved
        read it, regrid it, subset it, and return it
    '''
    year=date.year

    filepath='Data/OMNO2d/yearavg_%4d.h5'%year
    # Read and save file if it doesn't exist yet:
    if not isfile(filepath):
        if __VERBOSE__:
            print("Reading OMNO2d %d then writing year avg to %s"%(year,filepath))
        y0=datetime(year,1,1)
        yN=util.last_day(datetime(year,12,1))

        # Read whole year of omno2d and save to a yearly file
        omno2, omno2_attrs = read_omno2d(day0=y0, dayN=yN, month=False)

        # Average over the year:
        omno2['tropno2'] = np.nanmean(omno2['tropno2'],axis=0)
        # remove dates arrays
        omno2.pop('dates')
        omno2_attrs.pop('dates')

        # save the file:
        save_to_hdf5(filepath, omno2,attrdicts=omno2_attrs)
    else:
        omno2, omno2_attrs = read_hdf5(filepath)

    # omno2d is saved on 0.25x0.25 grid, regrid_to_lower should always be the choice
    newlats,newlons,_late,_lone=util.lat_lon_grid(latres=latres,lonres=lonres)
    lats,lons=omno2['lats'],omno2['lons']
    no2=util.regrid_to_lower(omno2['tropno2'],lats,lons,newlats,newlons)

    if region is not None:
        subset=util.lat_lon_subset(newlats,newlons,region,[no2])
        newlats=subset['lats']
        newlons=subset['lons']
        no2=subset['data'][0]

    return no2,newlats,newlons

def make_anthro_mask(d0,dN=None, threshy=1.5e15, threshd=1e15, latres=0.25, lonres=0.3125, region=None):
    '''
        Read year of OMNO2d
        Create filter from d0 to dN using yearly average over threshy
        and daily amount over threshd

        takes almost 5 mins to make a mask
    '''

    # Dates where we want filter
    dates=util.list_days(d0,dN,month=False)

    # Read the tropno2 columns for dates we want to look at
    omno2, omno2_attrs = read_omno2d(day0=d0, dayN=dN, month=False)
    lats = omno2['lats']
    lons = omno2['lons']
    no2  = omno2['tropno2'] # [[dates,] lats, lons]

    # regridding 0.25x0.25 to 0.25x0.3125 may cause issues, but lets see
    newlats, newlons, _nlate, _nlone = util.lat_lon_grid(latres=latres,lonres=lonres)

    # bool array we will return
    ret = np.zeros([len(dates),len(newlats),len(newlons)],dtype=np.bool)
    no2i= np.zeros(ret.shape)+np.NaN

    # Daily filter
    for i,day in enumerate(dates):
        no2day=[no2,no2[i]][len(dates)>1]
        no2i[i] = util.regrid_to_lower(no2day,lats,lons,newlats,newlons,func=np.nanmean)
        ret[i] = no2i[i] > threshd

    no2mean,_lats,_lons=yearly_anthro_avg(d0,latres=latres,lonres=lonres,region=region)
    #assert newlats==_lats, 'yearmask lats are fishy'
    #assert newlons==_lons, 'yearmask lons are fishy'

    yearmask=no2mean>threshy
    ret[i,yearmask]=True # mask values where yearly avg threshhold is exceded

    # subset to region
    if region is not None:
        subset=util.lat_lon_subset(newlats,newlons,region,[ret],has_time_dim=len(dates)>1)
        ret=subset['data'][0]
        lats,lons=subset['lats'],subset['lons']
    return ret, dates, lats, lons

def filter_high_latitudes(array, latres=0.25, lonres=0.3125, highest_lat=60.0):
    '''
    Read an array, assuming globally gridded at latres/lonres, set the values polewards of 60 degrees
    to nans
    '''
    # Array shape determines how many dimensions, and latdim tells us which is the latitude one
    lats=np.arange(-90,90,latres) + latres/2.0
    highlats= np.where(np.abs(lats) > highest_lat)
    newarr=np.array(array)
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
    assert os.path.isfile(fname), "ERROR: file missing: %s"%fname
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
    amfs[amfs<0.0] = np.NaN # convert missing numbers to NaN
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

