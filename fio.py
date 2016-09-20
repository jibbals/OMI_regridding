# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslade April 2016

Reads OMI datasets (level 2 and level 2 gridded)
Reads regridded and reprocessed datasets
Reads AURA Fire datasets (MYD14C8H: 8 day averages)

Writes HDF5 files (used to create regridded and reprocessed datasets)
'''
## Modules
# plotting module, and something to prevent using displays(can save output but not display it)

# module for hdf eos 5
import h5py 
import numpy as np
from datetime import datetime, timedelta
from glob import glob
# my module with class definitions and that jazz
from gchcho import gchcho

# interpolation method for ND arrays
# todo: remove once this is ported to reprocess.py
from scipy.interpolate import griddata

#just_global_things, good hashtag
datafieldsg = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/'
geofieldsg  = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Geolocation Fields/'
datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

def save_to_hdf5(outfilename, arraydict, fillvalue=0.0, verbose=False):
    '''
    Takes a bunch of arrays, named in the arraydict parameter, and saves 
    to outfilename as hdf5 using h5py with fillvalue=0, gzip compression
    '''
    if verbose:
        print("saving to "+outfilename)
    with h5py.File(outfilename,"w") as f:
        # attribute creation
        # give the HDF5 root some more attributes
        f.attrs['Filename']        = outfilename.split('/')[-1]
        f.attrs['creator']          = 'fio.py, Jesse Greenslade'
        f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        f.attrs['h5py_version']     = h5py.version.version
        f.attrs['Fill Value']       = fillvalue
        
        if verbose:
            print("Inside fio.save_to_hdf5()")
            print(arraydict.keys())
        
        for name in arraydict.keys():
            # create dataset, using arraydict values
            darr=arraydict[name]
            if verbose:
                print((name, darr.shape, darr.dtype))
            
            # Fill array using darr,
            # this way takes minutes to save, using ~ 500 MB space / avg
            dset=f.create_dataset(name,fillvalue=fillvalue,
                                  data=darr, compression_opts=9,
                                  chunks=True, compression="gzip")
            # for VC items and RSC, note the units in the file.
            if ('VC' in name) or ('RSC' == name) or ('SC' == name) or ('col_uncertainty' in name):
                dset.attrs["Units"] = "Molecules/cm2"
        # force h5py to flush buffers to disk
        f.flush()

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
        return glob('omhchog/OMI-Aura*%4dm%02d%02d*.he5'%(date.year, date.month, date.day))[0]
    if metaData:
        return ('omhchorp/metadata/metadata_%s.he5'%(date.strftime('%Y%m%d')))
    if not (regridded or reprocessed):
        return glob('omhcho/OMI-Aura*%4dm%02d%02d*'%(date.year, date.month, date.day))
    
    # geos chem output created via idl scripts match the following
    if geoschem:
        return ('gchcho/hcho_%4d%2d.he5'%(date.year,date.month))
    
    # reprocessed and regridded match the following:
    avg=['8','1'][oneday] # one or 8 day average when applicable
    typ=['p','g'][regridded] # reprocessed or regridded
    res='%1.2fx%1.2f'%(latres,lonres) # resolution string
    d = 'omhchor'+typ+'/' # directory
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
    filepath='MYD14C8H/MYD14C8H.%4d%03d.h5' % (tt.tm_year, daymatch)
    return read_8dayfire_path(filepath)

def read_8dayfire_path(path):
    '''
    Read fires file using given path
    '''    
    ## Fields to be read:
    # Count of fires in each grid box over 8 days
    corrFirePix = 'CorrFirePix'
    cloudCorrFirePix = 'CloudCorrFirePix'
    
    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        cfp     = in_f[corrFirePix].value
        ccfp    = in_f[cloudCorrFirePix].value
    
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
    lats = np.arange(90,-90,-0.5) - 0.25
    lons = np.arange(-180,180, 0.5) + 0.25
    
    newlats= np.arange(-90,90, latres) + latres/2.0
    newlons= np.arange(-180,180, lonres) + lonres/2.0

    mlons,mlats = np.meshgrid(lons,lats)
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)    
    
    fires = read_8dayfire(date)[0]
    interp = griddata( (mlats.ravel(), mlons.ravel()), fires.ravel(), (mnewlats, mnewlons), method='nearest')
    return interp, newlats, newlons

def read_omhcho(path, szamax=60, screen=[-5e15, 1e17], maxlat=None, verbose=False):
    '''
    Read info from a single swath file
    NANify entries with main quality flag not equal to zero
    NANify entries where xtrackqualityflags aren't zero
    Returns:{'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
            'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
            'qualityflag':qf, 'xtrackflag':xqf,
            'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}
    '''
    
    # Total column amounts are in molecules/cm2
    field_hcho  = datafields+'ColumnAmount' 
    # other useful fields
    field_amf   = datafields+'AirMassFactor'
    field_amfg  = datafields+'AirMassFactorGeometric'
    field_apri  = datafields+'GasProfile'
    field_plevs = datafields+'ClimatologyLevels'
    field_w     = datafields+'ScatteringWeights'
    field_qf    = datafields+'MainDataQualityFlag'
    field_clouds= datafields+'AMFCloudFraction'
    field_xqf   = geofields +'XtrackQualityFlags'
    field_lon   = geofields +'Longitude'
    field_lat   = geofields +'Latitude'
    field_sza   = geofields +'SolarZenithAngle'
    # uncertainty flags
    field_colUnc    = datafields+'ColumnUncertainty' # also molecules/cm2
    field_fitflag   = datafields+'FitConvergenceFlag'
    field_fitRMS    = datafields+'FittingRMS'
    
    
    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        lats    = in_f[field_lat].value     #[ 1644, 60 ]
        lons    = in_f[field_lon].value     #
        hcho    = in_f[field_hcho].value    #
        amf     = in_f[field_amf].value     # 
        amfg    = in_f[field_amfg].value    # geometric amf
        clouds  = in_f[field_clouds].value  # cloud fraction
        qf      = in_f[field_qf].value      #
        xqf     = in_f[field_xqf].value     # cross track flag
        sza     = in_f[field_sza].value     # solar zenith angle
        # uncertainty arrays                #
        cunc    = in_f[field_colUnc].value  # uncertainty
        fcf     = in_f[field_fitflag].value # convergence flag
        frms    = in_f[field_fitRMS].value  # fitting rms
        #                                   # [ 47, 1644, 60 ]
        w       = in_f[field_w].value       # scattering weights
        apri    = in_f[field_apri].value    # apriori
        plevs   = in_f[field_plevs].value   # pressure dim
        
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
    
    #return hcho, lats, lons, amf, amfg, w, apri, plevs
    return {'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
            'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
            'qualityflag':qf, 'xtrackflag':xqf, 'sza':sza,
            'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}

def read_omhcho_day(day=datetime(2005,1,1),verbose=False):
    '''
    Read an entire day of omhcho swaths
    '''
    fnames=determine_filepath(day)
    data=read_omhcho(fnames[0],verbose=verbose) # read first swath
    swths=[]
    for fname in fnames[1:]: # read the rest of the swaths
        swths.append(read_omhcho(fname),verbose=verbose)
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
        

def read_omhchog(date, eightdays=False, verbose=False):
    '''
    Function to read provided lvl 2 gridded product omhchog
    '''
    def getdata(date, verbose=False):
        ## File to be read:
        fname=determine_filepath(date, gridded=True)
        if verbose: print ("reading "+fname)
        
        # Total column amounts are in molecules/cm2
        # total vertical columns
        field_hcho  = datafieldsg+'ColumnAmountHCHO' 
        # other useful fields
        field_amf   = datafieldsg+'AirMassFactor'
        field_qf    = datafieldsg+'MainDataQualityFlag'
        field_xf    = datafieldsg+'' # todo: cross track flags
        field_lon   = datafieldsg+'Longitude'
        field_lat   = datafieldsg+'Latitude'
        
        ## read in file:
        with h5py.File(fname,'r') as in_f:        
            ## get data arrays
            lats    = in_f[field_lat].value
            lons    = in_f[field_lon].value
            hcho    = in_f[field_hcho].value
            amf     = in_f[field_amf].value
            qf      = in_f[field_qf].value
            #xf      = in_f[field_xf].value
        
        ## remove missing values and bad flags: 
        # QF: missing<0, suss=1, bad=2
        suss = qf != 0
        amf[suss] =np.NaN
        hcho[suss]=np.NaN
        lats[suss]=np.NaN
        lons[suss]=np.NaN
        
        # TODO: Remove row anomaly
        # TODO: Cloud frac removal?
        # TODO: Make function to remove post processed AAOD>thresh data
        
        return (hcho, lats, lons, amf)#, xf)
    
    # Return our day if that's all we want
    if not eightdays:
        return getdata(date,verbose=verbose)
    
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    lats=np.arange(-90,90,0.25)+0.25/2.0
    lons=np.arange(-180,180,0.25)+0.25/2.0
    hchos=np.zeros([len(lats),len(lons)])
    amfs=np.zeros([len(lats),len(lons)])
    counts=0
    for day in days8:
        hcho, daylats, daylons, amf = getdata(day,verbose=verbose)
        count=1-np.isnan(hcho)
        counts=np.nansum(count,axis=0)+counts
        hchos=np.nansum(hcho,axis=0) + hchos
        amfs=np.nansum(amf,axis=0)+amfs
    amfs=amfs/counts
    hchos=hchos/counts
    return (hchos, lats, lons, amfs, counts)

def read_omhchorg(date, oneday=False, latres=0.25, lonres=0.3125, keylist=None):
    '''
    Function to read the data from one of my regreeded files
    Inputs:
        date = datetime(y,m,d,0) of desired day file
        oneday  = False : set to True to read a single day, leave as False to read 8-day avg
        latres=0.25 : latitude resolution
        lonres=0.3125 : lon resolution
        keylist=None : list of arrays to return( default is all data )
    '''
    
    if keylist is None:
        keylist= ['AMF_G','AMF','ColumnAmountHCHO','GridEntries','Latitude','Longitude','ScatteringWeight','ShapeFactor','PressureLevels']
    retstruct=dict.fromkeys(keylist)
    fpath=determine_filepath(date, oneday=oneday, latres=latres, lonres=lonres, regridded=True)
    with h5py.File(fpath,'r') as in_f:
        for key in keylist:
            retstruct[key]=in_f[key].value
    return (retstruct)

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
        keylist=['AMF_GC','AMF_GCz','AMF_OMI','SC','VC_GC','VC_OMI','VCC','gridentries',
                 'latitude','longitude','RSC','RSC_latitude','RSC_GC','RSC_region',
                 'col_uncertainty_OMI','fires']
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
    return (retstruct)

def read_gchcho(date):
    '''
    Read the geos chem hcho column data into a class and return the class
    this is actually just a wrapper, class method does all the work.
    '''
    dataset = gchcho()
    dataset.ReadFile(date)
    return(dataset)

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
    

    
