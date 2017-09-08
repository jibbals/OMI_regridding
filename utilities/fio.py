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
# interpolation method for ND arrays
# todo: remove once this is ported to reprocess.py
from scipy.interpolate import griddata

###############
### GLOBALS ###
###############
#just_global_things, good hashtag
datafieldsg = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/'
geofieldsg  = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Geolocation Fields/'
datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

__VERBOSE__=False

# Keys for omhchorp:
__OMHCHORP_KEYS__ = [
    'latitude','longitude',
    'gridentries',   # how many satellite pixels make up the pixel
    'ppentries',     # how many pixels we got the PP_AMF for
    'RSC',           # The reference sector correction [rsc_lats, 60]
    'RSC_latitude',  # latitudes of RSC
    'RSC_region',    # RSC region [S,W,N,E]
    'RSC_GC',        # GEOS-Chem RSC [RSC_latitude] (molec/cm2)
    'VCC',           # The vertical column corrected using the RSC
    'VCC_PP',        # Corrected Paul Palmer VC
    'AMF_GC',        # AMF calculated using by GEOS-Chem]
    'AMF_GCz',       # secondary way of calculating AMF with GC
    'AMF_OMI',       # AMF from OMI swaths
    'AMF_PP',        # AMF calculated using Paul palmers code
    'SC',            # Slant Columns
    'VC_GC',         # GEOS-Chem Vertical Columns
    'VC_OMI',        # OMI VCs
    'VC_OMI_RSC',    # OMI VCs with Reference sector correction? TODO: check
    'col_uncertainty_OMI',
    'fires',         # Fire count
    'fire_mask_8',   # true where fires occurred over last 8 days
    'fire_mask_16' ] # true where fires occurred over last 16 days

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
                             'desc':'OMI\'s RSC column amount'},
    'RSC':                  {'units':'molec/cm2',
                             'desc':'GEOS-Chem/OMI based Reference Sector Correction: is applied to pixels based on latitude and track number'},
    'RSC_latitude':         {'units':'degrees',
                             'desc':'latitude centres for RSC'},
    'RSC_GC':               {'units':'molec/cm2',
                             'desc':'GEOS-Chem HCHO over reference sector'},
    'col_uncertainty_OMI':  {'units':'molec/cm2',
                             'desc':'OMI\'s column uncertainty'},
    'AMF_GC':               {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_OMI':              {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_PP':               {'desc':'AMF based on PPalmer code using OMI and GEOS-Chem'},
    'fire_mask_16':         {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current or prior 8 day block"},
    'fire_mask_8':          {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current 8 day block"},
    'fires':                {'desc':"8 day fire count from AQUA"},
    }

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

###############
### METHODS ###
###############

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

def combine_dicts(d1,d2):
    '''
    Add two dictionaries together
    '''
    return dict(d1.items() + d2.items() + [ (k, d1[k] + d2[k]) for k in set(d2) & set(d1) ])

def determine_filepath(date, latres=0.25,lonres=0.3125, gridded=False, regridded=False, reprocessed=False, geoschem=False, oneday=True, metaData=False):
    '''
    Make filename based on date, resolution, and type of file.
    '''

    # if not created by me just return the filepath(s) using date variable and glob
    if gridded:
        return glob('Data/omhchog/OMI-Aura*%4dm%02d%02d*.he5'%(date.year, date.month, date.day))[0]
    if metaData:
        return ('Data/omhchorp/metadata/metadata_%s.he5'%(date.strftime('%Y%m%d')))
    if not (regridded or reprocessed):
        return glob('Data/omhcho/OMI-Aura*%4dm%02d%02d*'%(date.year, date.month, date.day))

    # geos chem output created via idl scripts match the following
    if geoschem:
        return ('Data/gchcho/hcho_%4d%2d.he5'%(date.year,date.month))

    # reprocessed and regridded match the following:
    avg=['8','1'][oneday] # one or 8 day average when applicable
    typ=['p','g'][regridded] # reprocessed or regridded
    res='%1.2fx%1.2f'%(latres,lonres) # resolution string
    d = 'Data/omhchor'+typ+'/' # directory
    fpath=d+"omhcho_%s_%4d%02d%02d.he5" %(avg+typ+res,date.year,date.month,date.day)
    return(fpath)

def read_8dayfire(date=datetime(2005,1,1,0)):
    '''
    Read 8 day fire average file
    This function determines the filepath based on date input
    '''
    # filenames are all like *yyyyddd.h5, where ddd is day of the year(DOY), one for every 8 days
    tt = date.timetuple()

    # only every 8 days matches a file
    # this will give us a multiple of 8 which matches our DOY
    daymatch= int(np.floor(tt.tm_yday/8)*8) +1
    filepath='Data/MYD14C8H/MYD14C8H.%4d%03d.h5' % (tt.tm_year, daymatch)
    return read_8dayfire_path(filepath)

def read_8dayfire_path(path):
    '''
    Read fires file using given path
    '''
    ## Fields to be read:
    # Count of fires in each grid box over 8 days
    corrFirePix = 'CorrFirePix'
    #cloudCorrFirePix = 'CloudCorrFirePix'

    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        cfp     = in_f[corrFirePix].value
        #ccfp    = in_f[cloudCorrFirePix].value

    # from document at
    # http://www.fao.org/fileadmin/templates/gfims/docs/MODIS_Fire_Users_Guide_2.4.pdf
    # latitude = 89.75 - 0.5 * y
    # longitude = -179.75 + 0.5 * x
    lats    = np.arange(90,-90,-0.5) - 0.25
    lons    = np.arange(-180,180, 0.5) + 0.25
    return (cfp, lats, lons)

def read_8dayfire_interpolated(date,latres,lonres):
    '''
    Read the date, interpolate data to match lat/lon resolution, return data
    '''
    ##original lat/lons:
    fires, lats, lons = read_8dayfire(date)
    #lats = np.arange(90,-90,-0.5) - 0.25
    #lons = np.arange(-180,180, 0.5) + 0.25

    newlats= np.arange(-90,90, latres) + latres/2.0
    newlons= np.arange(-180,180, lonres) + lonres/2.0

    mlons,mlats = np.meshgrid(lons,lats)
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)

    interp = griddata( (mlats.ravel(), mlons.ravel()), fires.ravel(), (mnewlats, mnewlons), method='nearest')
    return interp, newlats, newlons

def read_E_new(oneday=None, filename=None):
    '''
    Function to read the recalculated Emissions output
    Inputs:
        oneday = None : read a single day - set this arg to a datetime to do this
        filename=None : if set then read this file ( used for testing )
    Output:
        Structure containing E_new dataset
    '''

    # First get filename:
    ddir='Data/Isop/E_new/'
    fpath=glob(ddir+'emissions_20050101-*.h5')[-1]

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

def read_E_new_range(day0, dayN):
    '''
        Read E_new from day0 to dayN
    '''
    print("TODO: To be implemented")

def read_omhcho(path, szamax=60, screen=[-5e15, 1e17], maxlat=None, verbose=False):
    '''
    Read info from a single swath file
    NANify entries with main quality flag not equal to zero
    NANify entries where xtrackqualityflags aren't zero
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
        if maxlat is not None:
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
        if szamax is not None:
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

def read_omhchorp(date, oneday=False, latres=0.25, lonres=0.3125, keylist=None, filename=None):
    '''
    Function to read a reprocessed omi file, by default reads an 8-day average (8p)
    Inputs:
        date = datetime(y,m,d) of file
        oneday = False : read a single day average rather than 8 day average
        latres=0.25
        lonres=0.3125
        keylist=None : if set to a list of strings, just read those data from the file, otherwise read all data
        filename=None : if set then read this file ( used for testing )
    Output:
        Structure containing omhchorp dataset
    '''

    if keylist is None:
        keylist=__OMHCHORP_KEYS__
    retstruct=dict.fromkeys(keylist)
    if filename is None:
        fpath=determine_filepath(date,oneday=oneday,latres=latres,lonres=lonres,reprocessed=True)
    else:
        fpath=filename

    with h5py.File(fpath,'r') as in_f:
        #print('reading from file '+fpath)
        for key in keylist:
            try:
                retstruct[key]=in_f[key].value
            except KeyError as ke: # if there is no matching key then print an error and continue
                print("Key Error in %s"%fpath)
                print(ke)
                retstruct[key]=np.NaN
    return retstruct

def read_omhchorp_month(date):
    ''' read a month of omhchorp data. '''

def read_gchcho(date):
    '''
    Read the geos chem hcho column data into a dictionary
    '''
    fpath=glob('Data/gchcho/hcho_%4d%02d.he5' % ( date.year, date.month ) )[0]
    ret_data={}
    with h5py.File(fpath, 'r') as in_f:
        dsetname='GC_UCX_HCHOColumns'
        dset=in_f[dsetname]

        for key in __GCHCHO_KEYS__:
            ret_data[key] = dset[key].squeeze()
    return ret_data


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
    if not os.path.isfile(fname):
        return None,None
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
