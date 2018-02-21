# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:05 2016

This script is used to take GC output, OMI hcho swathes,
    and OMNO2d gridded fire counts - to combine them into my omhcho dataset

@author: jesse
"""

# module to read/write data to file
from utilities import fio
from classes.gchcho import gchcho
import numpy as np
import os.path
import sys

import timeit # to look at how long python code takes...

# lets use sexy sexy parallelograms
from multiprocessing import Pool
#from functools import partial

from datetime import timedelta, datetime
#from glob import glob

# GLOBALS
__VERBOSE__=True # set to true for more print statements
__DEBUG__=False # set to true for even more print statements

# interpolate linearly to 500 points
ref_lat_bins=np.arange(-90,90,0.36)+0.18



def sum_dicts(d1,d2):
    '''
    Add two dictionaries together, only where keys match
    '''
    d3=dict()
    # for each key in both dicts
    for k in set(d1) & set(d2):
        d3[k]=d1[k]+d2[k]
    return d3

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


def get_good_pixel_list(date, getExtras=False, maxlat=60, PalmerAMF=True, verbose=False):
    '''
    Create a long list of 'good' pixels
    Also calculate new AMF for each good pixel
    If getExtras==True then also return q and xtrack flags, along with
        omega and apriori columns, FOR TESTING ONLY

    '''
    ## 0) setup stuff:
    # list where we keep good ref sector pixels
    # Stuff from OMI
    lats=list()
    lons=list()
    slants=list()       # Slant columns from (molec/cm2)
    RSC_OMI=list()      # RSC vertical columns (molec/cm2) from abad15
    ref_column=list()   # Median remote pacific reference radiance column (molec/cm2) # todo:remove- abad says this is not what I thought
    AMFos=list()        # AMFs from OMI
    AMFGs=list()        # Geometric AMFs from OMI
    cloudfracs=list()   # cloud fraction
    track=list()        # track index 0-59, used in refseccorrection
    flags=list()        # main data quality flags
    xflags=list()       # cross track flags from
    apris=None          # aprioris (molecs/cm3)
    ws=None             # omegas
    w_pmids=None        # omega pressure mids (hPa)
    cunc=list()         # Uncertainties (molecs/cm2)
    fcf=list()
    frms=list()
    # specifically for randal martin AMF code:
    sza=list()          # solar zenith angle
    vza=list()          # viewing zenith angle
    scan=list()         # row from swath
    ctp=list()          # cloud top pressure
    AMFpp=list()        # AMFs calculated using lidort and randal martin code
                        # Data/GC_Output/tropchem_geos5_2x25_47L/pp_amf

    # Stuff created using GEOS-Chem info
    sigmas=None
    AMFgcs=list()       # AMFs using S_s
    AMFgczs=list()      # AMFs using S_z

    ## grab our GEOS-Chem apriori info (dimension: [ levs, lats, lons ])
    gcdata = gchcho(date)
    # GCHCHO UNITS ARE GENERALLY m AND hPa

    ## 1) read in the good pixels for a particular date,
    ##  create and store the GEOS-Chem AMF along with the SC
    #

    # loop through swaths
    files = fio.determine_filepath(date)
    if verbose:
        print("%d omhcho files for %s"%(len(files),str(date)))
    for ff in files:
        if verbose: print("trying to read %s"%ff)
        omiswath = fio.read_omhcho(ff, maxlat=maxlat, verbose=verbose)
        flat,flon = omiswath['lats'], omiswath['lons']

        # only looking at good pixels
        goods = np.logical_not(np.isnan(flat))
        if __VERBOSE__: print("%d good pixels in %s"%(np.sum(goods),ff))

        # some things for later use:
        flats=list(flat[goods])
        flons=list(flon[goods])
        omamfgs=list((omiswath['AMFG'])[goods])
        plevs=(omiswath['plevels'])[:,goods]
        omegas=(omiswath['omega'])[:,goods]

        # Each track has a radiance reference column amount for each swath
        # taken from the median over the remote pacific
        f_ref_col=omiswath['rad_ref_col'] # array of 60, units: molec/cm2

        # We also store the track position for reference sector correction later
        goodwhere=np.where(goods==True)
        swathtrack=goodwhere[1]
        swathscan=goodwhere[0]
        track.extend(list(swathtrack))
        scan.extend(list(swathscan))

        # so each pixel has it's tracks radiance reference sector correction:
        ref_column.extend(list(f_ref_col[swathtrack]))

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
        ctp.extend(list((omiswath['ctp'])[goods]))
        sza.extend(list((omiswath['sza'])[goods]))
        vza.extend(list((omiswath['vza'])[goods]))
        cunc.extend(list((omiswath['coluncertainty'])[goods]))

        if getExtras:
            flags.extend(list((omiswath['qualityflag'])[goods])) # should all be zeros
            xflags.extend(list((omiswath['xtrackflag'])[goods])) # also zeros
            fcf.extend(list((omiswath['convergenceflag'])))
            frms.extend(list((omiswath['fittingRMS'])))
            # these are 47x1600x60
            aprioris=(omiswath['apriori'])[:,goods]
            if apris is None:
                apris=aprioris
                ws=omegas
                w_pmids=plevs
            else: # turn these into arrays of 47xentries
                apris=np.append(apris, aprioris,axis=1)
                ws=np.append(ws, omegas,axis=1)
                w_pmids=np.append(w_pmids, plevs, axis=1)

        # Create new AMF for each good entry...
        for i in range(np.sum(goods)):
            newAMF_s, newAMF_z = gcdata.calculate_AMF(omegas[:,i],plevs[:,i],omamfgs[i],flats[i],flons[i])
            AMFgcs.append(newAMF_s)
            AMFgczs.append(newAMF_z)
            # AMF_GCz does not relevel to the highest surface pressure between pixel and gridbox
            # AMF_GC does (using this one)
            # We can do some sort of test here if required.

    # If we want the PP list of AMFs
    AMFpp=np.zeros(len(lats))+np.NaN
    if PalmerAMF:
        pp_inds, pp_amf = fio.read_AMF_pp(date)
        if pp_inds is not None:
            pp_inds=np.array(pp_inds); pp_amf=np.array(pp_amf)
            AMFpp[pp_inds]=pp_amf
    AMFpp=list(AMFpp)

    # after all the swaths are read in: send the lists back in a single
    # dictionary structure

    return({'lat':lats, 'lon':lons, 'SC':slants, 'rad_ref_col':ref_column,
            'AMF_OMI':AMFos, 'AMF_GC':AMFgcs, 'AMF_GCz':AMFgczs, 'AMF_G':AMFGs,
            'RSC_OMI':RSC_OMI, 'AMF_PP':AMFpp,
            'sza':sza, 'vza':vza, 'ctp':ctp,
            'cloudfrac':cloudfracs, 'track':track, 'scan':scan,
            'qualityflag':flags, 'xtrackflag':xflags,
            'omega':ws, 'omega_pmids':w_pmids, 'apriori':apris, 'sigma':sigmas,
            'columnuncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms})

def reference_sector_correction(date, latres=0.25, lonres=0.3125, goodpixels=None):
    '''
    Determine the reference sector correction for a particular day
    Returns reference sector correction array [ 500, 60 ]
        interpolated from 90S to 90N linearly at 500 points(0.36 deg resolution)
        for each of the 60 omi tracks
        Array has units of molecules/cm2
    THIS IS FROM GONZALEZ ABAD 2015 UPDATED SAO PAPER
    '''

    ## Grab GEOS-Chem monthly average:
    #
    gcdata=gchcho(date)
    gc_lons=gcdata.lons
    gc_ref_inds=(gc_lons <= -140) & (gc_lons >= -160)
    gc_lats=gcdata.lats
    # fix endpoints so that lats array is [-90, -88, ..., 88, 90]
    gc_lats[0]=-90
    gc_lats[-1]=90

    # GEOS-CHEM VC HCHO in molecules/m2, convert to molecules/cm2
    gc_VC=gcdata.VC_HCHO * 1e-4
    # pull out reference sector vertical columns
    gc_VC_ref=gc_VC[:,gc_ref_inds]
    # average over the longitude dimension
    gc_VC_ref=np.mean(gc_VC_ref,axis=1)
    # interpolate linearly to 500 points
    ref_lat_mids=np.arange(-90,90,0.36)+0.18
    ref_lat_bounds=np.arange(-90, 90.01, 0.36)
    #gc_VC_ref_interp=np.interp(ref_lat_bins, gc_lats, gc_VC_ref)

    ## for any latitude, return the gc_VC_ref value using linear interpolation
    # over 90S to 90N, in molecules/cm2:
    def gc_VC_ref_func(lat):
        return (np.interp(lat,gc_lats,gc_VC_ref))

    ## NOW we grab the OMI satellite reference pixels
    # if we don't have a list of goodpixels passed in then get them
    if goodpixels is None:
        goodpixels = get_good_pixel_list(date)
    omi_lons=np.array(goodpixels['lon'])
    omi_lats=np.array(goodpixels['lat'])
    #OMI SCs in molecules/cm2
    omi_SC=np.array(goodpixels['SC'])
    omi_track=np.array(goodpixels['track']) # list of track indexes
    omi_AMF=np.array(goodpixels['AMF_OMI']) # Old AMF seen by OMI

    ## Get indices of ref sector pixels
    #
    ref_inds = (omi_lons > -160) & (omi_lons < -140)

    #reference sector slant columns
    ref_lats=omi_lats[ref_inds]
    ref_SC=omi_SC[ref_inds] # molecs/cm2
    ref_track=omi_track[ref_inds]
    ref_amf=omi_AMF[ref_inds]

    ## Reference corrections for each reference sector pixel
    # correction[lats] = OMI_refSC - GEOS_chem_Ref_VCs * AMF_omi
    # NB: ref_SC=molecs/cm2, gc=molecs/cm2 (converted from m2 above)
    # ref_corrections are in molecs/cm2
    ref_corrections = ref_SC - gc_VC_ref_func(ref_lats) * ref_amf

    # use median at each lat bin along each of the 60 tracks
    #
    ref_sec_correction = np.zeros((500,60), dtype=np.double) + np.NAN
    # for each track
    for i in range(60):
        track_inds= ref_track == i
        track_lats=ref_lats[track_inds]
        track_corrections = ref_corrections[track_inds]
        # reference sector is interpolated over 500 latitudes
        for j in range(500):
            lat_low=ref_lat_bounds[j]
            lat_high=ref_lat_bounds[j+1]
            lat_range_inds = (track_lats >= lat_low) & (track_lats <= lat_high)
            # if no entries then skip calculation
            test=np.sum(lat_range_inds)
            if test > 0:
                # grab median of corrections within this lat range
                median_correction=np.median(track_corrections[lat_range_inds])
                ref_sec_correction[j,i]=median_correction

    # ref_sec_correction [500, 60] is done
    return(ref_sec_correction, gc_VC_ref_func(ref_lat_mids))

def create_omhchorp_1(date, latres=0.25, lonres=0.3125, remove_clouds=True, remove_fires=True, verbose=True):
    '''
    1) get good pixels list from OMI swath files
    2) determine reference sector correction
    3) calculate VCs, VCC(ref corrected), Anything else
    4) place lists neatly into gridded latlon arrays
    5) Save as hdf5 with nice enough attributes
    '''

    ## 1)
    #
    ymdstr=date.strftime("%Y%m%d")

    if __VERBOSE__:
        print("create_omhchorp_1 called for %s"%ymdstr)
    ## set stdout to parent process
    if verbose or __DEBUG__:
        sys.stdout = open("logs/create_omhchorp.%s"%ymdstr, "w")
        print("This file was created by reprocess.create_omhchorp_1(%s) "%str(date))
        print("Turn off verbose and __DEBUG__ to stop creating these files")
        print("Process thread: %s"%str(os.getpid()))

    goodpixels=get_good_pixel_list(date, verbose=(verbose or __DEBUG__))
    omi_lons=np.array(goodpixels['lon'])
    omi_lats=np.array(goodpixels['lat'])
    # SC UNITS: Molecs/cm2
    omi_SC=np.array(goodpixels['SC'])
    omi_RSC=np.array(goodpixels['RSC_OMI'])
    omi_AMF_gc=np.array(goodpixels['AMF_GC'])
    omi_AMF_gcz=np.array(goodpixels['AMF_GCz'])
    omi_AMF_omi=np.array(goodpixels['AMF_OMI'])
    omi_AMF_pp=np.array(goodpixels['AMF_PP'])
    omi_tracks=np.array(goodpixels['track'])
    omi_clouds=np.array(goodpixels['cloudfrac'])
    omi_cunc=np.array(goodpixels['columnuncertainty'])

    ## 2)
    # reference sector correction to slant column pixels
    # Correction and GC_ref_sec are both in molecules/cm2
    ref_sec_correction, GC_ref_sec=reference_sector_correction(date,latres=latres, lonres=lonres, goodpixels=goodpixels)

    # Now we turn it into an interpolated function with lat and track the inputs:
    #
    def rsc_function(lats,track):
        track_correction=ref_sec_correction[:,track]

        # fix the NAN values through interpolation
        # [nan, 1, 2, nan, 4, nan, 4] -> [1, 1, 2, 3, 4, 4, 4]
        nans=np.isnan(track_correction)
        latnans=np.isnan(lats)

        if nans.all() or latnans.all():
            return np.repeat(np.NaN,len(lats))

        try:
            track_correction = np.interp(ref_lat_bins, ref_lat_bins[~nans], track_correction[~nans])
        except ValueError as verr:
            print("ERROR:")
            print(verr)
            print("Warning: Continuing on after that ERROR")
            return np.repeat(np.NaN, len(lats))
        return(np.interp(lats, ref_lat_bins, track_correction))

    ## 3)
    # Calculate VCs:
    omi_VC_gc   = omi_SC / omi_AMF_gc
    omi_VC_omi  = omi_SC / omi_AMF_omi
    # omi_VC_pp   = omi_SC / omi_AMF_pp # just look at corrected for now...
    # Calculate GC Ref corrected VC (called VCC by me)
    omi_VCC = np.zeros(omi_VC_gc.shape)+np.NaN
    omi_VCC_pp=np.zeros(omi_VC_gc.shape)+np.NaN
    # TODO: also here calculate the full VCC from RM code where the AMF exists

    for track in range(60):
        track_inds= omi_tracks==track
        # for each track VCC = (SC - correction) / AMF_GC
        # If track has one or less non-nan values then skip
        if np.isnan(omi_lats[track_inds]).all():
            continue
        track_rsc=rsc_function(omi_lats[track_inds],track)
        track_sc=omi_SC[track_inds]
        omi_VCC[track_inds]= (track_sc - track_rsc) / omi_AMF_gc[track_inds]
        omi_VCC_pp[track_inds]=(track_sc - track_rsc) / omi_AMF_pp[track_inds]
    # that should cover all good pixels - except if we had a completely bad track some day
    #assert np.sum(np.isnan(omi_VCC))==0, "VCC not created at every pixel!"
    if __VERBOSE__ and np.isnan(omi_VCC).any():
        vccnans=np.sum(np.isnan(omi_VCC))
        print ("Warning %d nan vcc entries from %d on %s"%(vccnans, len(omi_VCC),ymdstr))

    ## 4)
    # take list and turn into gridded product...
    # majority of processing time in here ( 75 minutes? )

    # how many lats, lons
    lats,lons,lat_bounds,lon_bounds=create_lat_lon_grid(latres=latres,lonres=lonres)
    ny,nx = len(lats), len(lons)

    # Filter for removing cloudy entries
    cloud_filter = omi_clouds < 0.4 # This is a list of booleans for the pixels
    # Filter for removing fire affected squares(from current and prior 8 days)
    # filter is booleans matching lat/lon grid. True for fires
    fire_count, _flats, _flons = fio.read_8dayfire_interpolated(date,latres=latres,lonres=lonres)
    fire_filter_16 = get_16day_fires_mask(date,latres=latres,lonres=lonres)
    fire_filter_8  = get_8day_fires_mask(date,latres=latres,lonres=lonres)

    ## DATA which will be outputted in gridded file
    SC      = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_gc   = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_omi  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC     = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC_pp  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    RSC_OMI = np.zeros([ny,nx],dtype=np.double)+np.NaN
    cunc_omi= np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_gc  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_gcz = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_omi = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_pp  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    counts  = np.zeros([ny,nx],dtype=np.int)
    countspp  = np.zeros([ny,nx],dtype=np.int)
    for i in range(ny):
        for j in range(nx):

            # how many pixels within this grid box
            matches=(omi_lats >= lat_bounds[i]) & (omi_lats < lat_bounds[i+1]) & (omi_lons >= lon_bounds[j]) & (omi_lons < lon_bounds[j+1])

            # remove clouds
            if remove_clouds:
                matches = matches & cloud_filter
            counts[i,j]= np.sum(matches)
            if counts[i,j] < 1:
                continue
            #Different count for PP entries
            countspp[i,j]= np.sum(~np.isnan(omi_VCC_pp[matches]))

            # Save the means of each good grid pixel
            SC[i,j]         = np.mean(omi_SC[matches])
            VC_gc[i,j]      = np.mean(omi_VC_gc[matches])
            VC_omi[i,j]     = np.mean(omi_VC_omi[matches])
            VCC[i,j]        = np.mean(omi_VCC[matches])
            VCC_pp[i,j]     = np.mean(omi_VCC_pp[matches])
            RSC_OMI[i,j]    = np.mean(omi_RSC[matches]) # the correction amount
            # TODO: store analysis data for saving, when we decide what we want analysed
            cunc_omi[i,j]   = np.mean(omi_cunc[matches])
            AMF_gc[i,j]     = np.mean(omi_AMF_gc[matches])
            AMF_gcz[i,j]    = np.mean(omi_AMF_gcz[matches])
            AMF_omi[i,j]    = np.mean(omi_AMF_omi[matches])
            AMF_pp[i,j]     = np.mean(omi_AMF_pp[matches])

    ## 5)
    # Save one day averages to file
    #if ('VC' in name) or ('RSC' == name) or ('SC' == name) or ('col_uncertainty' in name):
    #    dset.attrs["Units"] = "Molecules/cm2"
    #if 'fire_mask_16' == name:
    #    dset.attrs["description"] = "1 if 1 or more fires in this or the 8 adjacent gridboxes over the current or prior 8 day blocks"
    #if 'fire_mask_8' == name:
    #    dset.attrs["description"] = "1 if 1 or more fires in this or the 8 adjacent gridboxes over the current 8 day block"

    outd=dict()
    outd['VC_OMI']              = VC_omi
    outd['VC_GC']               = VC_gc
    outd['SC']                  = SC
    outd['VCC']                 = VCC
    outd['VCC_PP']              = VCC_pp
    outd['VC_OMI_RSC']          = RSC_OMI # omi's RSC column amount
    outd['gridentries']         = counts
    outd['ppentries']           = countspp
    outd['latitude']            = lats
    outd['longitude']           = lons
    outd['RSC']                 = ref_sec_correction
    outd['RSC_latitude']        = ref_lat_bins
    outd['RSC_GC']              = GC_ref_sec
    outd['RSC_region']          = np.array([-90, -160, 90, -140])
    outd['col_uncertainty_OMI'] = cunc_omi
    outd['AMF_GC']              = AMF_gc
    outd['AMF_GCz']             = AMF_gcz
    outd['AMF_OMI']             = AMF_omi
    outd['AMF_PP']              = AMF_pp
    outd['fire_mask_16']        = fire_filter_16.astype(np.int16)
    outd['fire_mask_8']         = fire_filter_8.astype(np.int16)
    outd['fires']               = fire_count.astype(np.int16)
    outfilename=fio.determine_filepath(date,latres=latres,lonres=lonres,reprocessed=True,oneday=True)

    if __VERBOSE__:
        print("sending day average to be saved: "+outfilename)
    if __DEBUG__:
        print(("keys: ",outd.keys()))

    fio.save_to_hdf5(outfilename, outd, attrdicts=fio.__OMHCHORP_ATTRS__, verbose=verbose)
    if __DEBUG__:
        print("File should be saved now...")
    ## 5.1)
    ## TODO: Save analysis metadata like SSDs or other metrics
    #


def create_omhchorp_8(date, latres=0.25, lonres=0.3125):
    '''
    Combine eight omhchorp_1 outputs to create an 8 day average file
    '''
    # check that there are 8 omhchorp_1 files
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    files8= []
    for day in days8:
        #yyyymmdd=date.strftime("%Y%m%d")
        filename=fio.determine_filepath(day,latres=latres,lonres=lonres,oneday=True,reprocessed=True)
        assert os.path.isfile(filename), "ERROR: file not found for averaging : "+filename
        files8.append(filename)

    # normal stuff will be identical between days
    normallist=['latitude', 'longitude', 'RSC_latitude', 'RSC_region',
                'fires', 'fire_mask_8', 'fire_mask_16']
    # list of things we need to add together and average
    sumlist=['AMF_GC', 'AMF_GCz', 'AMF_OMI', 'SC', 'VC_GC', 'VC_OMI','VC_OMI_RSC',
             'VCC', 'col_uncertainty_OMI']
    # other things need to be handled seperately
    otherlist=['gridentries','RSC', 'RSC_GC','ppentries','AMF_PP','VCC_PP']

    # keylist is all keys
    keylist=sumlist.copy()
    keylist.extend(normallist)
    keylist.extend(otherlist)

    # function adds arrays pretending nan entries are zero. two nans add to nan
    def addArraysWithNans(x, y):
        x = np.ma.masked_array(np.nan_to_num(x), mask=np.isnan(x) & np.isnan(y))
        y = np.ma.masked_array(np.nan_to_num(y), mask=x.mask)
        return (x+y).filled(np.nan)

    # initialise sum dictionary with data from first day
    sumdict=dict()
    data=fio.read_omhchorp(date,latres=latres,lonres=lonres,oneday=True)
    for key in keylist:
        sumdict[key] = data[key].astype(np.float64)

    # read in and sum the rest of the 8 reprocessed days
    for day in days8[1:]:
        data=fio.read_omhchorp(day,latres=latres,lonres=lonres,oneday=True)
        daycount=data['gridentries']
        daycountpp=data['ppentries']
        sumdict['gridentries'] = addArraysWithNans(sumdict['gridentries'], daycount)
        sumdict['ppentries'] = addArraysWithNans(sumdict['ppentries'], daycountpp)
        sumdict['RSC'] = addArraysWithNans(sumdict['RSC'], data['RSC'])
        sumdict['RSC_GC'] = sumdict['RSC_GC'] + data['RSC_GC']

        for ppkey in ['AMF_PP','VCC_PP']:
            y=data[ppkey].astype(np.float64) * daycountpp
            sumdict[ppkey] = addArraysWithNans(sumdict[ppkey],y)

        # for each averaged amount we want to sum together the totals
        for key in sumlist:
            # y is today's total amount ( gridcell avgs * gridcell counts )
            y=data[key].astype(np.float64) * daycount
            # add arrays treating nans as zero using masked arrays
            sumdict[key]= addArraysWithNans(sumdict[key],y)

    # take the average and save out to netcdf
    avgdict=dict()
    counts=sumdict['gridentries']
    countspp=sumdict['ppentries']
    for key in sumlist:
        avgdict[key]= sumdict[key] / counts.astype(float)
    for key in normallist:
        avgdict[key]= sumdict[key]
    avgdict['gridentries']=counts.astype(int)
    avgdict['ppentries']=countspp.astype(int)
    avgdict['RSC']=sumdict['RSC']/8.0
    avgdict['RSC_GC']=sumdict['RSC_GC']/8.0
    avgdict['AMF_PP']=sumdict['AMF_PP']/countspp.astype(float)
    avgdict['VCC_PP']=sumdict['VCC_PP']/countspp.astype(float)
    outfilename=fio.determine_filepath(date,latres=latres,lonres=lonres,oneday=False,reprocessed=True)
    fio.save_to_hdf5(outfilename, avgdict, attrdicts=fio.__OMHCHORP_ATTRS__)
    print("File Saved: "+outfilename)

def Reprocess_N_days(date, latres=0.25, lonres=0.3125, days=8, processes=8, remove_clouds=True,remove_fires=True):
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

    #arguments: date,latres,lonres,getsum
    # function arguments in a list, for each of N days
    inputs = [(dd, latres, lonres, remove_clouds, remove_fires) for dd in daysN]

    # run all at once since each day can be independantly processed
    results = [ pool.apply_async(create_omhchorp_1, args=inp) for inp in inputs ]

    if __DEBUG__:
        print("apply_async called for each day")
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    if __DEBUG__:
        print("Pool Closed and Joined")
    if __VERBOSE__:
        elapsed = timeit.default_timer() - start_time
        print ("Took %6.2f minutes to reprocess %3d days using %2d processes"%(elapsed/60.0,days,processes))

def get_8day_fires_mask(date=datetime(2005,1,1), latres=0.25, lonres=0.3125):
    '''
    1) read aqua 8 day fire count
    2) return a mask set to true where fire influence is expected
    '''
    def set_adjacent_to_true(mask):
        mask_copy = np.zeros(mask.shape).astype(bool)
        ny,nx=mask.shape
        for x in range(nx):
            for y in np.arange(1,ny-1): # don't worry about top and bottom row
                mask_copy[y,x] = np.sum(mask[[y-1,y,y+1],[x-1,x,(x+1)%nx]]) > 0
        return mask_copy

    # read day fires
    fires, flats, flons = fio.read_8dayfire_interpolated(date,latres=latres,lonres=lonres)

    # TODO: read night fires:

    # create a mask in squares with fires or adjacent to fires
    mask = fires > 0
    retmask = set_adjacent_to_true(mask)

    return retmask

def get_16day_fires_mask(date, latres=0.25, lonres=0.3125):
    '''
    '''
    # current 8 day fire mask
    mask = get_8day_fires_mask(date, latres, lonres)
    # prior 8 days fire mask:
    pridate=date-timedelta(days=8)
    if pridate >= datetime(2005,1,1):
        maskpri= get_8day_fires_mask(pridate,latres,lonres)
        mask = mask | maskpri
    return mask

