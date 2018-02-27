# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:05 2016

This script is used to take GC output, OMI hcho swathes,
    and OMNO2d gridded fire counts - to combine them into my omhcho dataset

@author: jesse
"""

# modules to read/write data to file
import h5py # for hdf5 reading/writing
import pandas as pd # for the modis csv files

# math module
import numpy as np

# path modules
import os.path
import sys
from glob import glob

import timeit # to look at how long python code takes...

# lets use sexy sexy parallelism
from multiprocessing import Pool

# datetime for nice date handling
from datetime import timedelta, datetime

# plotting :
from mpl_toolkits.basemap import Basemap, interp

###########
# GLOBALS
###########
__VERBOSE__=True # set to true for more print statements
__DEBUG__=True # set to true for even more print statements
_orig_stdout_=sys.stdout

###########
# Helper Functions
###########

def create_lat_lon_grid(latres=0.25,lonres=0.3125):
    '''
    Returns lats, lons, latbounds, lonbounds for grid with input resolution
    '''
    # lat and lon bin boundaries
    lat_bounds=np.arange(-90, 90+latres/2.0, latres)
    lon_bounds=np.arange(-180, 180+lonres/2.0, lonres)
    # lat and lon bin midpoints
    lats=np.arange(-90,90,latres)+latres/2.0
    lons=np.arange(-180,180,lonres)+lonres/2.0

    return(lats,lons,lat_bounds,lon_bounds)

def list_days(day0,dayn=None,month=False):
    '''
        return list of days from day0 to dayn, or just day0
        if month is True, return [day0,...,end_of_month]
    '''
    if month:
        dayn=last_day(day0)
    if dayn is None: return [day0,]
    numdays = (dayn-day0).days + 1 # timedelta
    return [day0 + timedelta(days=x) for x in range(0, numdays)]

def edges_from_mids(x,fix=False):
    '''
        Take a lat or lon vector input and return the edges
        Works for REGULAR grids only
    '''
    assert x[1]-x[0] == x[2]-x[1], "Resolution at edge not representative"
    # replace assert with this if it works, HANDLES GEOS CHEM LATS PROBLEM ONLY
    if x[1]-x[0] != x[2]-x[1]:
        xres=x[2]-x[1]   # Get resolution away from edge
        x[0]=x[1]-xres   # push out the edges
        x[-1]=x[-2]+xres #

    # new vector for array
    newx=np.zeros(len(x)+1)
    # resolution from old vector
    xres=x[1]-x[0]
    # edges will be mids - resolution / 2.0
    newx[0:-1]=np.array(x) - xres/2.0
    # final edge
    newx[-1]=newx[-2]+xres

    # Finally if the ends are outside 90N/S or 180E/W then bring them back
    if fix:
        if newx[-1] >= 90: newx[-1]=89.99
        if newx[0] <= -90: newx[0]=-89.99
        if newx[-1] >= 180: newx[-1]=179.99
        if newx[0] <= -180: newx[0]=-179.99

    return newx

def area_quadrangle(SWNE):
    '''
        Return area of sphere with earths radius bounded by S,W,N,E quadrangle
        units = km^2
    '''
    #Earths Radius
    R=6371.0
    # radians from degrees
    S,W,N,E=SWNE
    Sr,Wr,Nr,Er = np.array(SWNE)*np.pi/180.0
    # perpendicular distance from plane containing line of latitude to the pole
    # (checked with trig)

    h0=R*(1-np.sin(Sr))
    h1=R*(1-np.sin(Nr))

    # Area north of a latitude: (Spherical cap - wikipedia)
    A0= 2*np.pi*R*h0
    A1= 2*np.pi*R*h1
    A_zone= A0-A1 # Area of zone from south to north

    # portion between longitudes
    p=(E-W)/360.0

    # area of quadrangle
    A= A_zone*p
    return A

def area_grid(lats, lons):
    '''
        Area give lats and lons in a grid in km^2
        can do non grid with provided latres, lonres arrays

        Lats and Lons are centres of gridpoints
    '''
    areas=np.zeros([len(lats),len(lons)]) + np.NaN
    latres=np.abs(lats[1]-lats[0])
    lonres=np.abs(lons[1]-lons[0])
    yr,xr=latres/2.0,lonres/2.0

    for yi,y in enumerate(lats):
        for xi, x in enumerate(lons):
            if not np.isfinite(x+y):
                continue
            SWNE=[y-yr, x-xr, y+yr, x+xr]
            areas[yi,xi] = area_quadrangle(SWNE)
    return areas

def regrid_to_lower(data, lats, lons, newlats_e, newlons_e, func=np.nanmean):
    '''
        Regrid data to lower resolution
        using EDGES of new grid and mids of old grid
        apply func to data within each new gridbox (mean by default)
    '''
    ret=np.zeros([len(newlats_e)-1,len(newlons_e)-1])+np.NaN
    for i in range(len(newlats_e)-1):
        for j in range(len(newlons_e)-1):
            lati= (lats >= newlats_e[i]) * (lats < newlats_e[i+1])
            loni= (lons >= newlons_e[j]) * (lons < newlons_e[j+1])

            tmp=data[lati,:]
            tmp=tmp[:,loni]
            ret[i,j]=func(tmp)
    return ret

def set_adjacent_to_true(mask):
    '''
        Take a mask (EG fire mask) and set squares adjacent to true as true
    '''
    mask_copy = np.zeros(mask.shape).astype(bool)
    ny,nx=mask.shape
    for x in range(nx):
        for y in np.arange(1,ny-1): # don't worry about top and bottom row
            mask_copy[y,x] = np.sum(mask[[y-1,y,y+1],[x-1,x,(x+1)%nx]]) > 0
    return mask_copy


###########
## FIO
###########

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

def read_hdf5(filename):
    '''
        Should be able to read hdf5 files created by my method above...
        Returns data dictionary and attributes dictionary
    '''
    if __VERBOSE__:
        print('reading from file '+filename)

    retstruct={}
    retattrs={}
    with h5py.File(filename,'r') as in_f:

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

def read_regridded_swath(date,dateN=None):
    '''
        Read one to N days of regridded swaths and fires
    '''
    # Reading One day only is easy:
    if dateN is None:
        filename=date.strftime('Data/omi_hcho_%Y%m%d.hdf')
        data,attr=read_hdf5(filename)
    # Reading many days is requires combining them:
    else:
        data,attr = {},{}
        dats,atts  = [],[]
        days=list_days(date,dateN,False) # list of days
        # Read each day into a list
        for day in days:
            filename=day.strftime('Data/omi_hcho_%Y%m%d.hdf')
            dat,att=read_hdf5(filename)
            dats.append(dat)
            atts.append(att)
        # Combine list into array for each of the non dimensional data
        for k in dats[0].keys():
            data[k] = np.array([dats[j][k] for j in range(len(days))])
        for k in ['lats','lons']:
            data[k]=dats[0][k]
        # assume all the attributes match:
        attr = atts[0]
    return data,attr

def read_omhcho(path, szamax=60, screen=[-5e15, 1e17], maxlat=60, verbose=False):
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

    datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
    geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

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


def read_MOD14A1(date=datetime(2005,1,1), per_km2=False):
    '''
        Read the modis product of firepix/1000km2/day

        Returns firepix/km2/day or firepix/day
    '''

    # file looks like:
    #'Data/MOD14A1_D_FIRE/2005/MOD14A1_D_FIRE_2005-01-02.CSV'
    fpath='Data/MOD14A1_D_FIRE/'+date.strftime('%Y/MOD14A1_D_FIRE_%Y-%m-%d.CSV')
    fires=pd.read_csv(fpath).values

    fires[fires>9000] = 0. # np.NaN # ocean squares
    fires[fires==0.1] = 0. # land squares but no fire I think...
    assert np.nansum(fires < 0) == 0, "There are negative fire pixels?"

    lats=np.linspace(89.9,-89.9,1799) # lats are from top to bottom when read using pandas
    lons=np.linspace(-180,179.9,3600) # lons are from left to right

    if not per_km2:
        # area per gridbox in km2:
        area=area_grid(lats,lons)
        fires=fires * 1e3 * area # now fire_pixels/day

    return fires,lats,lons

def read_MOD14A1_interpolated(date=datetime(2005,1,1), latres=0.25,lonres=0.3125):
    '''
        Read firepixels/day from MOD14A1 daily gridded product
        returns fires, lats, lons
    '''
    newlats= np.arange(-90,90, latres) + latres/2.0
    newlons= np.arange(-180,180, lonres) + lonres/2.0
    newlats_e=edges_from_mids(newlats)
    newlons_e=edges_from_mids(newlons)
    fires,lats,lons=read_MOD14A1(date,per_km2=False)

    newfires=regrid_to_lower(fires,lats,lons,newlats_e,newlons_e,func=np.nansum)
    return newfires,newlats,newlons

def get_good_pixel_list(date, maxlat=60):
    '''
    Create a long list of 'good' pixels
    '''
    ## 0) setup stuff:
    # list where we keep good ref sector pixels
    # Stuff from OMI
    lats=list()
    lons=list()
    slants=list()       # Slant columns from (molec/cm2)
    RSC_OMI=list()      # corrected vertical columns (molec/cm2) from abad15
    AMFos=list()        # AMFs from OMI
    AMFGs=list()        # Geometric AMFs from OMI
    cloudfracs=list()   # cloud fraction
    track=list()        # track index 0-59, used in refseccorrection
    scan=list()         # scan index
    flags=list()        # main data quality flags
    xflags=list()       # cross track flags (not sure)
    cunc=list()         # Uncertainties (molecs/cm2)
    fcf=list()
    frms=list()


    ## 1) read in the good pixels for a particular date,
    ##
    #

    # grab all swaths for input date:
    files = glob(date.strftime('Data/omhcho/%Y/OMI-Aura_L2-OMHCHO_%Ym%m%d*'))
    assert len(files) > 0, "omhcho data is not at %s"%date.strftime('Data/omhcho/%Y/OMI-Aura_L2-OMHCHO_%Ym%m%d*')

    if __DEBUG__:
        print("%d omhcho files for %s"%(len(files),str(date)))

    # loop through swaths
    for ff in files:
        if __DEBUG__: print("trying to read %s"%ff)
        omiswath = read_omhcho(ff, maxlat=maxlat)
        flat,flon = omiswath['lats'], omiswath['lons']

        # only looking at good pixels
        goods = np.logical_not(np.isnan(flat))
        if __DEBUG__: print("%d good pixels in %s"%(np.sum(goods),ff))

        # some things for later use:
        flats=list(flat[goods])
        flons=list(flon[goods])
        omamfgs=list((omiswath['AMFG'])[goods])

        # We also store the track position for reference sector correction later
        goodwhere=np.where(goods==True)
        swathtrack=goodwhere[1]
        swathscan=goodwhere[0]
        track.extend(list(swathtrack))
        scan.extend(list(swathscan))

        # SCs are VCs * AMFs
        fslants=omiswath['HCHO']*omiswath['AMF'] # In molecules/cm2

        # add this file's lats,lons,SCs,AMFs to our lists
        slants.extend(list(fslants[goods]))
        RSC_OMI.extend(list(omiswath['RSC_OMI'][goods]))
        lats.extend(flats)
        lons.extend(flons)
        AMFos.extend(list((omiswath['AMF'])[goods]))
        AMFGs.extend(omamfgs)
        cloudfracs.extend(list((omiswath['cloudfrac'])[goods]))
        cunc.extend(list((omiswath['coluncertainty'])[goods]))


    # after all the swaths are read in: send the lists back in a single
    # dictionary structure

    return({'lat':lats, 'lon':lons, 'SC':slants,
            'AMF_OMI':AMFos, 'RSC_OMI':RSC_OMI,
            'cloudfrac':cloudfracs, 'track':track, 'scan':scan,
            'qualityflag':flags, 'xtrackflag':xflags,
            'columnuncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms})

def make_gridded_swaths(date, latres=0.25, lonres=0.3125, remove_clouds=True):
    '''
    1) get good pixels list from OMI swath files

    4) place lists neatly into gridded latlon arrays
    5) Save as hdf5 with nice enough attributes
    '''

    ## 1)
    #
    ymdstr=date.strftime("%Y%m%d")

    if __VERBOSE__:
        print("create_omhchorp_1 called for %s"%ymdstr)
    ## set stdout to parent process
    if __DEBUG__:
        #sys.stdout = open("logs/create_omhchorp.%s"%ymdstr, "w")
        print("This file was created by reprocess.create_omhchorp_1(%s) "%str(date))
        print("Turn off verbose and __DEBUG__ to stop creating these files")
        print("Process thread: %s"%str(os.getpid()))

    goodpixels=get_good_pixel_list(date)
    lons_pix=np.array(goodpixels['lon'])
    lats_pix=np.array(goodpixels['lat'])
    # SC UNITS: Molecs/cm2
    SC_pix=np.array(goodpixels['SC'])
    VC_C_pix=np.array(goodpixels['RSC_OMI'])
    AMF_pix=np.array(goodpixels['AMF_OMI'])
    cloud_pix=np.array(goodpixels['cloudfrac'])
    cunc_pix=np.array(goodpixels['columnuncertainty'])

    # how many lats, lons
    lats,lons,lat_bounds,lon_bounds=create_lat_lon_grid(latres=latres,lonres=lonres)
    ny,nx = len(lats), len(lons)

    # reCalculate VCs:
    VC_pix  = SC_pix / AMF_pix

    ## Read fires in
    fire_count,_flats,_flons=read_MOD14A1_interpolated(date,latres=latres,lonres=lonres)

    ## 4)
    # take list and turn into gridded product...
    # majority of processing time in here ( 75 minutes? )


    # Filter for removing cloudy entries
    cloud_filter = cloud_pix < 0.4 # This is a list of booleans for the pixels

    assert all(_flats == lats), "fire interpolation does not match our resolution"

    ## DATA which will be outputted in gridded file
    #SC      = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC      = np.zeros([ny,nx],dtype=np.double)+np.NaN      # Vert col
    VC_C    = np.zeros([ny,nx],dtype=np.double)+np.NaN      # corrected
    cunc    = np.zeros([ny,nx],dtype=np.double)+np.NaN      # uncert
    AMF     = np.zeros([ny,nx],dtype=np.double)+np.NaN      # AMF
    counts  = np.zeros([ny,nx],dtype=np.int)                # pixel count per box

    for i in range(ny):
        for j in range(nx):

            # how many pixels within this grid box
            matches=(lats_pix >= lat_bounds[i]) & (lats_pix < lat_bounds[i+1]) & (lons_pix >= lon_bounds[j]) & (lons_pix < lon_bounds[j+1])

            # remove clouds
            if remove_clouds:
                matches = matches & cloud_filter

            # how many pixels in this gridbox
            counts[i,j]= np.sum(matches)

            # go to next gridbox if no pixels in this one
            if counts[i,j] < 1:
                continue

            # Save the means of each good grid pixel
            VC[i,j]         = np.mean(VC_pix[matches])
            VC_C[i,j]       = np.mean(VC_C_pix[matches])
            cunc[i,j]       = np.mean(cunc_pix[matches])
            AMF[i,j]        = np.mean(AMF_pix[matches])

    ## 5)
    # Save one day averages to file
    # with attributes
    outd=dict()

    outd['VC']              = VC
    outd['VC_C']            = VC_C
    outd['pixels']          = counts
    outd['lats']            = lats
    outd['lons']            = lons
    outd['uncertainty']     = cunc
    outd['AMF']             = AMF
    outd['fires']           = fire_count
    outfilename='Data/omi_hcho_%s.hdf'%ymdstr

    attrs = {
        'pixels':          {'desc':'satellite pixels averaged per gridbox',
                            'units':'N'},
        'VC':              {'units':'molec/cm2',
                            'desc':'regridded OMI VC'},
        'VC_C':            {'units':'molec/cm2',
                            'desc':'regridded OMI VC, Reference sector corrected'},
        'uncertainty':     {'units':'molec/cm2',
                            'desc':'OMI column uncertainty'},
        'AMF':             {'desc':'average AMF for pixels'},
        'fires':           {'desc':"fire pixel count from MOD14A1"},
        'lats':            {'units':'degrees',
                            'desc':'grid box centres: deg north from equator'},
        'lons':            {'units':'degrees',
                            'desc':'grid box centres: deg east'},
        }

    if __VERBOSE__:
        print("sending day average to be saved: "+outfilename)
    if __DEBUG__:
        print(("keys: ",outd.keys()))

    save_to_hdf5(outfilename, outd, attrdicts=attrs, verbose=__DEBUG__)
    if __DEBUG__:
        print("File should be saved now...")



def regrid_N_days(date, latres=0.25, lonres=0.3125, remove_clouds=True,
                     days=8, processes=8):
    '''
    run the one day reprocessing function in parallel using N processes for M days
    '''

    # time how long it all takes
    if __VERBOSE__:
        print('processing %3d days using %2d processes'%(days,processes))
        start_time=timeit.default_timer()

    # our N days in a list
    daysN = [ date + timedelta(days=dd) for dd in range(days)]

    # create pool of M processes
    pool = Pool(processes=processes)
    if __DEBUG__:
        print ("Process pool created ")

    ## function arguments in a list, for each of N days
    # arguments: date,latres,lonres, remove_clouds
    inputs = [(dd, latres, lonres, remove_clouds) for dd in daysN]

    # run all at once since each day can be independantly processed
    results = [ pool.apply_async(make_gridded_swaths, args=inp) for inp in inputs ]

    if __DEBUG__:
        print("apply_async called for each day")
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    if __DEBUG__:
        print("Pool Closed and Joined")
    if __VERBOSE__:
        elapsed = timeit.default_timer() - start_time
        print ("Took %6.2f minutes to reprocess %3d days using %2d processes"%(elapsed/60.0,days,processes))



