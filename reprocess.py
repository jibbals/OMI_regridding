# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:05 2016

This script is used to take GC output, OMI hcho swathes,
    and OMNO2d gridded fire counts - to combine them into my omhcho dataset

@author: jesse
"""

# module to read/write data to file
from utilities import fio
from utilities import utilities as util
from utilities import GMAO
#from classes.gchcho import gchcho
from classes.GC_class import GC_sat
# Shouldn't use omhchorp while creating omhchorp
#import classes.omhchorp as omhchorp
import numpy as np
import os.path
import sys
import timeit # to look at how long python code takes...
from scipy import interpolate

# lets use sexy sexy parallelograms
from multiprocessing import Pool
#from functools import partial

from datetime import timedelta, datetime
#from glob import glob

# GLOBALS
__VERBOSE__=True # set to true for more print statements
__DEBUG__=True # set to true for even more print statements

# interpolate linearly to 500 points
__RSC_lat_bins__ = np.arange(-90,90,0.36)+0.18
__RSC_region__   = [-90, -160, 90, -140]

# latitudes and longitudes which I will regrid everything to
__latm__, __late__ = GMAO.GMAO_lats()
__lonm__, __lone__ = GMAO.GMAO_lons()
#latres=0.25, lonres=0.3125
__LATRES__ = GMAO.__LATRES__
__LONRES__ = GMAO.__LONRES__






def sum_dicts(d1,d2):
    '''
    Add two dictionaries together, only where keys match
    '''
    d3=dict()
    # for each key in both dicts
    for k in set(d1) & set(d2):
        d3[k]=d1[k]+d2[k]
    return d3


def get_good_pixel_list(date, getExtras=False, maxlat=60, PalmerAMF=True):
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
    VCC_OMI=list()      # RSC vertical columns (molec/cm2) from abad15
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
    #gcdata = gchcho(date)
    # Satellite overpass data: [lats,lons,levs]
    gcdata = GC_sat(date)

    ## 1) read in the good pixels for a particular date,
    ##  create and store the GEOS-Chem AMF along with the SC
    #

    # loop through swaths
    files = fio.determine_filepath(date,omhcho=True)
    if __DEBUG__:
        print("%d omhcho files for %s"%(len(files),str(date)))
    for ff in files:
        if __DEBUG__: print("trying to read %s"%ff)
        omiswath = fio.read_omhcho(ff, maxlat=maxlat)
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
        VCC_OMI.extend(list(omiswath['VCC_OMI'][goods]))
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
            'VCC_OMI':VCC_OMI, 'AMF_PP':AMFpp,
            'sza':sza, 'vza':vza, 'ctp':ctp,
            'cloudfrac':cloudfracs, 'track':track, 'scan':scan,
            'qualityflag':flags, 'xtrackflag':xflags,
            'omega':ws, 'omega_pmids':w_pmids, 'apriori':apris, 'sigma':sigmas,
            'columnuncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms})

def GC_ref_sector(month):
    '''
        Reference sector over remote pacific binned into 500 latitudes
        From GEOS-Chem satellite overpass data averaged monthly
    '''
    # Read O_hcho for the month
    keylist=['IJ-AVG-$_CH2O','TIME-SER_AIRDEN','BXHGHT-$_BXHEIGHT']
    gcdata= GC_sat(util.first_day(month),util.last_day(month),keys=keylist, run='tropchem')

    gc_lons=gcdata.lons
    gc_ref_inds=(gc_lons <= -140) & (gc_lons >= -160)
    gc_lats=gcdata.lats
    # fix endpoints so that lats array is [-90, -88, ..., 88, 90]
    gc_lats[0]=-90
    gc_lats[-1]=90

    # GEOS-CHEM VC HCHO in molecules/cm2
    VC=np.nanmean(gcdata.O_hcho,axis=0) # [lats,lons] - month averaged

    # pull out reference sector vertical columns
    VC_ref=VC[:,gc_ref_inds]
    # average over the longitude dimension
    VC_ref=np.mean(VC_ref,axis=1)

    # interpolate linearly to 500 points
    ref_lat_mids=__RSC_lat_bins__

    gc_VC_ref_interp=np.interp(ref_lat_mids, gc_lats, VC_ref)

    return gc_VC_ref_interp, ref_lat_mids


def reference_sector_correction(date, goodpixels=None):
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
    # Satellite overpass output from normal run:
    # Can do daily RSC... but shouldn't I think
    #gcdata=GC_sat(date,) # arrays like [lats,lons, 47levs]
    #vars(gcdata).keys()
    #   ['N_air', 'temp', '_has_time_dim', 'E_isop_bio_kgs', 'NO2', 'lons_e', 'dstr', 'E_isop_bio', 'nlons', 'OH', 'tplev', 'nlevs', 'ntimes', 'attrs', 'surftemp', 'hcho', 'O_hcho', 'area', 'nlats', 'lats', 'press', 'boxH', 'lats_e', 'dates', 'isop', 'time', 'lons', 'psurf'])

    # GC_ref[500], lats[500]
    gc_VC_ref, gc_ref_lats = GC_ref_sector(date)
    gc_VC_ref_interp = interpolate.interp1d(gc_ref_lats,gc_VC_ref,kind='nearest')
    ref_lat_bounds = util.edges_from_mids(gc_ref_lats)
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
    gc_AMF=np.array(goodpixels['AMF_GC']) # new amf from GC
    pp_AMF=np.array(goodpixels['AMF_PP']) # new amf using GC and PP code

    ## Get indices of ref sector pixels
    #
    ref_inds = (omi_lons > -160) & (omi_lons < -140)

    #reference sector slant columns
    ref_lats=omi_lats[ref_inds]
    ref_SC=omi_SC[ref_inds] # molecs/cm2
    ref_track=omi_track[ref_inds]
    # Ref AMF should be based on which VCC we are trying to determine (AMF_OMI, AMF_GC, AMF_PP )
    ref_amfs=[omi_AMF[ref_inds],gc_AMF[ref_inds],pp_AMF[ref_inds]]


    ## Reference corrections for each reference sector pixel
    # correction[lats] = OMI_refSC - GEOS_chem_Ref_VCs * AMF_omi
    # NB: ref_SC=molecs/cm2, gc=molecs/cm2
    # ref_corrections are in molecs/cm2
    # VCC_x = (SC_omi - ref_SC + VC_gc_0 x AMF_omi ) / AMF_x
    # correction_x = SC_omi_0 - ref_GC*AMF_omi
    ref_corrections = [ref_SC - gc_VC_ref_interp(ref_lats) * ref_amf for ref_amf in ref_amfs]

    # use median at each lat bin along each of the 60 tracks
    #
    ref_sec_correction = np.zeros((500,60,3), dtype=np.double) + np.NAN
    # for each track
    for i in range(60):
        track_inds= ref_track == i
        track_lats= ref_lats[track_inds]
        track_corrections = [ref_correction[track_inds] for ref_correction in ref_corrections]
        # reference sector is interpolated over 500 latitudes
        for j in range(500):
            lat_low=ref_lat_bounds[j]
            lat_high=ref_lat_bounds[j+1]
            lat_range_inds = (track_lats >= lat_low) & (track_lats < lat_high)
            # if no entries then skip calculation
            test=np.sum(lat_range_inds)
            if test > 0:
                # grab median of corrections within this lat range
                median_corrections=[np.median(track_correction[lat_range_inds]) for track_correction in track_corrections]
                for k in range(3):
                    ref_sec_correction[j,i,k]=median_corrections[k]

    # ref_sec_correction [500, 60, 3] and gc_VC_ref[500] are returned
    return(ref_sec_correction, gc_VC_ref)

def create_omhchorp(date):
    '''
    1) get good pixels list from OMI swath files
    2) determine reference sector correction
    3) calculate VCs, VCC(ref corrected), Anything else
    4) place lists neatly into gridded latlon arrays
    5) save as hdf5 with nice enough attributes
        Takes about 50 minutes to run a day (lots of reading and regridding)
    '''

    ## 1)
    #
    ymdstr=date.strftime("%Y%m%d")

    if __VERBOSE__:
        print("create_omhchorp called for %s"%ymdstr)
    ## set stdout to parent process
    #if __DEBUG__:
        # No longer running python threads
        #sys.stdout = open("logs/create_omhchorp.%s"%ymdstr, "w")
        #print("This file was created by reprocess.create_omhchorp(%s) "%str(date))
        #print("Turn off verbose and __DEBUG__ to stop creating these files")
        #print("Process thread: %s"%str(os.getpid()))

    goodpixels=get_good_pixel_list(date)
    omi_lons=np.array(goodpixels['lon'])
    omi_lats=np.array(goodpixels['lat'])
    # SC UNITS: Molecs/cm2
    omi_SC=np.array(goodpixels['SC'])
    omi_VCC_OMI=np.array(goodpixels['VCC_OMI'])
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
    # One correction for each of OMI, GC, PP amf determined
    ref_sec_corrections, GC_ref_sec=reference_sector_correction(date,goodpixels=goodpixels)
    # ref sec correction [latitude, track, k]

    # Now we turn it into an interpolated function with lat and track the inputs:
    #
    def rsc_function(lats,track,k):
        # k=0: OMI RSC
        # k=1: GC RSC
        # k=2: PP RSC
        track_correction=ref_sec_corrections[:,track,k]

        # fix the NAN values through linear interpolation
        # [nan, 1, 2, nan, 4, nan, 4] -> [1, 1, 2, 3, 4, 4, 4]
        nans=np.isnan(track_correction)
        latnans=np.isnan(lats)

        if nans.all() or latnans.all():
            return np.repeat(np.NaN,len(lats))

        try:
            track_correction = np.interp(__RSC_lat_bins__, __RSC_lat_bins__[~nans], track_correction[~nans])
        except ValueError as verr:
            print("ERROR:")
            print(verr)
            print("Warning: Continuing on after that ERROR")
            return np.repeat(np.NaN, len(lats))
        return(np.interp(lats, __RSC_lat_bins__, track_correction))

    ## 3)
    # Calculate VCs:
    omi_VC_gc   = omi_SC / omi_AMF_gc
    omi_VC_omi  = omi_SC / omi_AMF_omi
    omi_VC_pp   = omi_SC / omi_AMF_pp

    # Calculate GC Ref corrected VC (called VCC by me)
    omi_VCC = np.zeros(omi_VC_gc.shape)+np.NaN
    omi_VCC_pp=np.zeros(omi_VC_gc.shape)+np.NaN
    omi_VCC_omi_newrsc=np.zeros(omi_VC_gc.shape)+np.NaN

    # skip when no omhcho data available...
    if len(omi_lats) > 0:
        for track in range(60):
            track_inds= omi_tracks==track
            # for each track VCC = (SC - correction) / AMF_GC
            # If track has one or less non-nan values then skip
            if np.isnan(omi_lats[track_inds]).all():
                continue
            track_rscs=[rsc_function(omi_lats[track_inds],track,k) for k in range(3)]
            #gc_track_rsc=rsc_function(omi_lats[track_inds],track,1)
            #pp_track_rsc=rsc_function(omi_lats[track_inds],track,2)
            track_sc=omi_SC[track_inds]
            # original SC corrected by our new RSC
            omi_VCC_omi_newrsc[track_inds] = (track_sc - track_rscs[0]) / omi_AMF_omi[track_inds]
            # GC based VCC
            omi_VCC[track_inds]     = (track_sc - track_rscs[1]) / omi_AMF_gc[track_inds]
            # may be dividing by nans, ignore warnings
            omi_VCC_pp[track_inds]  = (track_sc - track_rscs[2]) / omi_AMF_pp[track_inds]

    # that should cover all good pixels - except if we had a completely bad track some day
    #assert np.sum(np.isnan(omi_VCC))==0, "VCC not created at every pixel!"
    if __VERBOSE__ and np.isnan(omi_VCC).any():
        vccnans=np.sum(np.isnan(omi_VCC))
        print ("Warning %d nan vcc entries from %d on %s"%(vccnans, len(omi_VCC),ymdstr))

    ## 4)
    # take list and turn into gridded product...
    # majority of processing time in here ( 55 minutes? )

    # how many lats, lons
    lats,lons,lat_bounds,lon_bounds=__latm__,__lonm__,__late__,__lone__
    ny,nx = len(lats), len(lons)

    # Filter for removing cloudy entries
    cloud_filter = omi_clouds < 0.4 # This is a list of booleans for the pixels

    # fire filter can be made from the fire_count
    fire_count,_flats,_flons=fio.read_MOD14A1_interpolated(date)
    assert all(_flats == lats), "fire interpolation does not match our resolution"

    # Smoke filter similarly can be made from AAOD stored each day
    smoke_aaod,_flats,_flons=fio.read_AAOD_interpolated(date)
    assert all(_flats == lats), "smoke aaod interpolation does not match our resolution"

    # masks here made using default values...
    # takes around 5 mins to do anthromask,
    # 10 mins for firemask,
    # 15 seconds for smoke mask
    # Masks now handled seperately
    #    firemask,_fdates,_flats,_flons=fio.make_fire_mask(date)
    #    smokemask,_sdates,_slats,_slons=fio.make_smoke_mask(date)
    #    anthromask,_adates,_alats,_alons=fio.make_anthro_mask(date)

    ## DATA which will be outputted in gridded file
    SC      = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_gc   = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_omi  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_pp   = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC_GC  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC_pp  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC_OMI = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC_OMI_newrsc = np.zeros([ny,nx],dtype=np.double)+np.NaN
    cunc_omi= np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_gc  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_gcz = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_omi = np.zeros([ny,nx],dtype=np.double)+np.NaN
    AMF_pp  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    counts  = np.zeros([ny,nx],dtype=np.int)
    countspp  = np.zeros([ny,nx],dtype=np.int)

    # Skip all this if there is no omhcho data on this day
    if len(omi_lats) > 0:
        for i in range(ny):
            for j in range(nx):

                # how many pixels within this grid box
                matches=(omi_lats >= lat_bounds[i]) & (omi_lats < lat_bounds[i+1]) & (omi_lons >= lon_bounds[j]) & (omi_lons < lon_bounds[j+1])

                # remove clouds
                #if remove_clouds:
                matches = matches & cloud_filter
                ppmatches = matches & ~np.isnan(omi_VCC_pp)
                counts[i,j]= np.sum(matches)
                #Different count for PP entries
                countspp[i,j]= np.sum(ppmatches)

                # Save the means of each good grid pixel
                if counts[i,j] > 0:
                    SC[i,j]         = np.mean(omi_SC[matches])
                    VC_gc[i,j]      = np.mean(omi_VC_gc[matches])
                    VC_omi[i,j]     = np.mean(omi_VC_omi[matches])
                    VCC_GC[i,j]     = np.mean(omi_VCC[matches]) # RSC Corrected VC_GC
                    VCC_OMI[i,j]    = np.mean(omi_VCC_OMI[matches])  # RSC corrected VC_omi
                    VCC_OMI_newrsc[i,j] = np.mean(omi_VCC_omi_newrsc[matches])  # RSC corrected VC_omi
                    cunc_omi[i,j]   = np.mean(omi_cunc[matches])
                    AMF_gc[i,j]     = np.mean(omi_AMF_gc[matches])
                    AMF_gcz[i,j]    = np.mean(omi_AMF_gcz[matches])
                    AMF_omi[i,j]    = np.mean(omi_AMF_omi[matches])

                if countspp[i,j] > 0:

                    VC_pp[i,j]      = np.mean(omi_VC_pp[ppmatches])
                    VCC_pp[i,j]     = np.mean(omi_VCC_pp[ppmatches]) # RSC Corrected VC_PP
                    AMF_pp[i,j]     = np.mean(omi_AMF_pp[ppmatches])
                    assert not np.isnan(VC_pp[i,j]), 'VC_PP created a nan from non-zero pixels!'
                    assert not np.isnan(VCC_pp[i,j]), 'VCC_PP created a nan from non-zero pixels!'

    outd=dict()

    outd['VC_OMI']              = VC_omi
    outd['VC_GC']               = VC_gc
    outd['VC_PP']               = VC_pp
    outd['SC']                  = SC
    outd['VCC_GC']              = VCC_GC
    outd['VCC_PP']              = VCC_pp
    outd['VCC_OMI']             = VCC_OMI # omi's VC (corrected by reference sector)
    outd['VCC_OMI_newrsc']      = VCC_OMI_newrsc
    outd['gridentries']         = counts
    outd['ppentries']           = countspp
    outd['latitude']            = lats
    outd['longitude']           = lons
    outd['RSC']                 = ref_sec_corrections
    outd['RSC_latitude']        = __RSC_lat_bins__
    outd['RSC_GC']              = GC_ref_sec
    outd['RSC_region']          = np.array(__RSC_region__)
    outd['col_uncertainty_OMI'] = cunc_omi
    outd['AMF_GC']              = AMF_gc
    outd['AMF_GCz']             = AMF_gcz
    outd['AMF_OMI']             = AMF_omi
    outd['AMF_PP']              = AMF_pp
    outd['fires']               = fire_count.astype(np.int16)
    outd['AAOD']                = smoke_aaod # omaeruvd aaod 500nm product interpolated
    #outd['firemask']            = np.squeeze(firemask.astype(np.int16))
    #outd['smokemask']           = np.squeeze(smokemask.astype(np.int16))
    #outd['anthromask']          = np.squeeze(anthromask.astype(np.int16))
    # Check we got everything:
    assert all( [ keyname in outd.keys() for keyname in fio.__OMHCHORP_KEYS__] ), 'Missing some keys from OMHCHORP product'
    outfilename=fio.determine_filepath(date,reprocessed=True)

    if __VERBOSE__:
        print("sending day average to be saved: "+outfilename)
    if __DEBUG__:
        print(("keys: ",outd.keys()))

    fio.save_to_hdf5(outfilename, outd, attrdicts=fio.__OMHCHORP_ATTRS__, verbose=__DEBUG__)
    if __DEBUG__:
        print("File should be saved now...")
    ## 5.1)
    ## TODO: Save analysis metadata like SSDs or other metrics
    #

def create_omhchorp_justfires(date):
    '''
        Function to create omhchorp with just fires, this is for fire filtering the first few days in 2005
    '''

    ymdstr=date.strftime("%Y%m%d")

    if __VERBOSE__:
        print("create_omhchorp_justfires called for %s"%ymdstr)

    # how many lats, lons
    lats,lons,lat_bounds,lon_bounds=__latm__,__lonm__,__late__,__lone__
    ny,nx = len(lats), len(lons)

    # fire filter can be made from the fire_count
    fire_count,_flats,_flons=fio.read_MOD14A1_interpolated(date)
    assert all(_flats == lats), "fire interpolation does not match our resolution"

    # Smoke filter similarly can be made from AAOD stored each day
    smoke_aaod,_flats,_flons=fio.read_AAOD_interpolated(date)
    assert all(_flats == lats), "smoke aaod interpolation does not match our resolution"

    # Save fires, smoke out to H5 format, along with dates, lats, lons
    outd=dict()
    outd['latitude']            = lats
    outd['longitude']           = lons
    outd['fires']               = fire_count.astype(np.int16)
    #outd['AAOD']                = smoke_aaod # omaeruvd aaod 500nm product interpolated
    firemask,_fdates,_flats,_flons=fio.make_fire_mask(date)
    outd['firemask']            = np.squeeze(firemask.astype(np.int16))
    outfilename=fio.determine_filepath(date,reprocessed=True)

    if __VERBOSE__:
        print("sending day average to be saved: "+outfilename)
    if __DEBUG__:
        print(("keys: ",outd.keys()))

    fio.save_to_hdf5(outfilename, outd, attrdicts=fio.__OMHCHORP_ATTRS__, verbose=__DEBUG__)
    if __DEBUG__:
        print("File should be saved now...")


def Reprocess_N_days(date, days=8, processes=8):
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
    inputs = [(dd) for dd in daysN]

    # run all at once since each day can be independantly processed
    results = [ pool.apply_async(create_omhchorp, args=inp) for inp in inputs ]

    if __DEBUG__:
        print("apply_async called for each day")
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    if __DEBUG__:
        print("Pool Closed and Joined")
    if __VERBOSE__:
        elapsed = timeit.default_timer() - start_time
        print ("Took %6.2f minutes to reprocess %3d days using %2d processes"%(elapsed/60.0,days,processes))

if __name__=='__main__':
    # reprocess a day as a test of the process
    start=timeit.default_timer()
    create_omhchorp(datetime(2008,1,1))

    print("Took %6.2f minutes to run for 1 day"%((timeit.default_timer()-start)/60.0))

