# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:05 2016

@author: jesse
"""

# module to read/write data to file
import fio
import numpy as np
import os.path

import timeit # to look at how long python code takes...
# lets use sexy sexy parallelograms
#import multiprocessing as mp
from multiprocessing import Pool
#from functools import partial

from datetime import timedelta, datetime
from glob import glob

# GLOBALS
__VERBOSE__=True # set to true for more print statements

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
    

def calculate_amf_sigma(AMF_G, w, S, w_coords, S_coords, plotname=None, plt=None):
    '''
    Determine AMF using AMF_G * \int_0^1 { w(s) S(s) ds }
    using sigma coordinate system
    w and S datapoints should be at w_coords and S_coords respectively
    This is for interpolation of w and S before integration
    scipy.integrate.quad() uses QUADPACK from fortran77 integration library:
        This uses adaptive quadrature and keeps error below (TODO: set a good tolerance)
    Inputs:(AMF_G, w, S, w_coords, S_coords, plotname=None)
        AMF_G   : geometric AMF from OMI data
        w       : array of OMI scattering weights
        S       : apriori ( sigma normalised )
        w_coords: sigma coordinates of w data array
        S_coords: sigma coordinates of S data array
        plotname: Set to something if you want to save a plot output
    Returns: AMF_new
    '''
    
    # linear interpolation, these become functions of w and S
    # w_ = interp1d(w_coords,w,'linear')
    
    # numpy linear interp, by default w_(>xmax) = w_(xmax), and same for negative side
    # numpy interp requires ascending 'from' coordinates
    def w_(x):
        return(np.interp(x, w_coords[::-1], w[::-1]))
    def S_(x):
        return(np.interp(x, S_coords[::-1], S[::-1]))
        
    ## just use sum of w * S / delta s for a bunch of steps...
    
    # 101 equidistant steps:
    #coord101 = np.linspace(1,0,101)
    #int1= np.sum(w_(coord101)*S_(coord101)*(coord101[0]-coord101[1]))
    
    # using geos 72 sigma levels:
    mids= (S_coords[0:-2] + S_coords[1:-1]) / 2.0
    lens= (S_coords[0:-2] - S_coords[1:-1])    
    int3 = np.sum(w_(mids)*S_(mids) * lens)
    
    AMF_new = AMF_G * int3
    if plotname is not None:
        assert plt is not None, 'need to pass plt'
        f=plt.figure(figsize=(10,8))
        plt.plot(w, w_coords, '.', label='orig $\omega$')
        plt.plot(w_(mids),mids, label='interp $\omega$')
        plt.plot(S, S_coords, '.', label='orig S')
        plt.plot(S_(mids),mids,label='interp S')
        plt.ylabel('$\sigma$'); plt.xlabel('unitless')
        mval=max(max(w),max(S),2.0)
        plt.ylim([1.05,-0.05]); plt.xlim([0.,mval])
        plt.legend(); plt.title('amf calculation factors')
        plt.text(.15,.7,'AMF=%5.2f'%AMF_new)
        f.savefig(plotname)
        print('saved '+plotname); plt.close()
    
    return(AMF_new)

def get_good_pixel_list(date, getExtras=False):
    '''
    Create a long list of 'good' pixels
    Also calculate new AMF for each good pixel
    If getExtras==True then also return q and xtrack flags, along with 
        omega and apriori columns, FOR TESTING ONLY
    '''
    ## 0) setup stuff:
    # list where we keep good ref sector pixels
    lats=list()
    lons=list()
    slants=list()
    AMFos=list()
    AMFgcs=list()
    cloudfracs=list()
    track=list() # track index 0-59 (for reference sector correction)
    flags=list()
    xflags=list()
    apris=None # aprioris (molecs/cm3?)
    ws=None # omegas
    sigmas=None
    
    ## grab our GEOS-Chem apriori info (dimension: [ levs, lats, lons ])
    gchcho = fio.read_gchcho(date)
    # GCHCHO UNITS ARE GENERALLY m AND hPa
    
    ## 1) read in the good pixels for a particular date,
    ##  create and store the GEOS-Chem AMF along with the SC
    #
    
    # loop through swaths
    files = fio.determine_filepath(date)
    for ff in files:
        omiswath = fio.read_omhcho(ff)
        flat,flon = omiswath['lats'], omiswath['lons']
        
        # only looking at good pixels
        goods = np.logical_not(np.isnan(flat))
        
        #if __VERBOSE__:
        #    print('%4e good entries in %s'%(np.sum(goods),ff))
        # some things for later use:
        flats=list(flat[goods])
        flons=list(flon[goods])
        omamfgs=list((omiswath['AMFG'])[goods])
        plevs=(omiswath['plevels'])[:,goods]
        omegas=(omiswath['omega'])[:,goods]
        
        # SCs are VCs * AMFs
        fslants=omiswath['HCHO']*omiswath['AMF'] # In molecules/cm2
        
        # add this file's lats,lons,SCs,AMFs to our lists
        slants.extend(list(fslants[goods]))
        lats.extend(flats)
        lons.extend(flons)
        AMFos.extend(list((omiswath['AMF'])[goods]))        
        cloudfracs.extend(list((omiswath['cloudfrac'])[goods]))
        
        # we still need to work out Geos Chem based AMFs for our pixels
        # for which we need pressures, sigmas, and omegas
        #
        
        # Determine sigma coordinates from pressure levels
        # use level 0 as psurf and top level as TOA
        #om_surf_alt = om_plevs[0,:,:] ** 2 / om_plevs[1,:,:]        
        om_sigma = np.zeros(plevs.shape)
        om_toa = plevs[-1, :]
        om_surf = plevs[0,:] 
        om_diff = om_surf-om_toa
        for ss in range(47):
            om_sigma[ss,:] = (plevs[ss,:] - om_toa)/om_diff
        
        if getExtras:
            flags.extend(list((omiswath['qualityflag'])[goods])) # should all be zeros
            xflags.extend(list((omiswath['xtrackflag'])[goods])) # also zeros
            # these are 47x1600x60
            aprioris=(omiswath['apriori'])[:,goods]
            if apris is None:
                apris=aprioris
                ws=omegas
                sigmas=om_sigma
            else: # turn these into arrays of 47xentries
                apris=np.append(apris, aprioris,axis=1)
                ws=np.append(ws, omegas,axis=1)
                sigmas=np.append(sigmas, om_sigma,axis=1)
        
        # Create new AMF for each good entry...
        for i in range(np.sum(goods)):
            gc_shape_i, gc_sigma_i = gchcho.get_single_apriori(flats[i], flons[i])
            NEWAMF = calculate_amf_sigma(omamfgs[i], omegas[:,i], gc_shape_i, om_sigma[:,i], gc_sigma_i)
            AMFgcs.append(NEWAMF)
        
        # We also store the track position for reference sector correction later
        swathtrack=np.where(goods==True)[1]
        track.extend(list(swathtrack))
    
    # after all the swaths are read in: send the lists back in a single 
    # dictionary structure
    return({'lat':lats,'lon':lons,'SC':slants,
            'AMF_OMI':AMFos, 'AMF_GC':AMFgcs, 
            'cloudfrac':cloudfracs, 'track':track,
            'qualityflag':flags,'xtrackflag':xflags,
            'omega':ws,'apriori':apris,'sigma':sigmas})

def reference_sector_correction(date, latres=0.25, lonres=0.3125, goodpixels=None):
    '''
    Determine the reference sector correction for a particular day
    Returns reference sector correction array [ 500, 60 ]
        interpolated from 90S to 90N linearly at 500 points(0.36 deg resolution)
        for each of the 60 omi tracks
        Array has units of molecules/cm2
    '''
    
    ## Grab GEOS-Chem monthly average:
    # 
    gchcho=fio.read_gchcho(date)
    gc_lons=gchcho.lons
    gc_ref_inds=(gc_lons <= -140) & (gc_lons >= -160)
    gc_lats=gchcho.lats
    # fix endpoints so that lats array is [-90, -88, ..., 88, 90]
    gc_lats[0]=-90
    gc_lats[-1]=90
    
    # GEOS-CHEM VC HCHO in molecules/m2, convert to molecules/cm2
    gc_VC=gchcho.VC_HCHO * 1e-4
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
    omi_AMF=np.array(goodpixels['AMF_GC'])
    
    ## Get indices of ref sector pixels
    #
    ref_inds = (omi_lons > -160) & (omi_lons < -140)
    
    #reference sector slant columns
    ref_lats=omi_lats[ref_inds]
    ref_SC=omi_SC[ref_inds] # molecs/cm2
    ref_track=omi_track[ref_inds]
    ref_amf=omi_AMF[ref_inds]
    
    ## Reference corrections for each reference sector pixel
    # correction[lats] = OMI_refSC - GEOS_chem_Ref_VCs * AMF
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
        # for each latitude bin ( PRE ABAD EMAIL )
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

def create_omhchorp_1(date, latres=0.25, lonres=0.3125, remove_clouds=True):
    '''
    1) get good pixels list from OMI swath files
    2) determine reference sector correction
    3) calculate VCs, VCC(ref corrected), Anything else 
    4) place lists neatly into gridded latlon arrays
    5) Save as netcdf
    '''
    ## 1) 
    # 
    goodpixels=get_good_pixel_list(date)
    omi_lons=np.array(goodpixels['lon'])
    omi_lats=np.array(goodpixels['lat'])
    # SC UNITS: Molecs/cm2
    omi_SC=np.array(goodpixels['SC'])
    omi_AMF_gc=np.array(goodpixels['AMF_GC'])
    omi_AMF_omi=np.array(goodpixels['AMF_OMI'])
    omi_tracks=np.array(goodpixels['track'])
    omi_clouds=np.array(goodpixels['cloudfrac'])
    
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
        track_correction = np.interp(ref_lat_bins, ref_lat_bins[~nans], track_correction[~nans])
        return(np.interp(lats, ref_lat_bins, track_correction))
    
    ## 3)
    # Calculate VCs:
    omi_VC_gc   = omi_SC / omi_AMF_gc
    omi_VC_omi  = omi_SC / omi_AMF_omi
    # Calculate GC Ref corrected VC (called VCC by me)
    omi_VCC = np.zeros(omi_VC_gc.shape)+np.NaN
    for track in range(60):
        track_inds= omi_tracks==track
        # for each track VCC = (SC - correction) / AMF_GC
        omi_VCC[track_inds]= (omi_SC[track_inds] - rsc_function(omi_lats[track_inds],track)) / omi_AMF_gc[track_inds]
    # that should cover all good pixels
    assert np.sum(np.isnan(omi_VCC))==0, "VCC not created at every pixel!"
    
    ## 4)
    # take list and turn into gridded product...
    # majority of processing time in here ( 75 minutes? )
    
    # lat and lon bin boundaries
    lat_bounds=np.arange(-90, 90+latres/2.0, latres)
    lon_bounds=np.arange(-180, 180+lonres/2.0, lonres)
    # lat and lon bin midpoints
    lats=np.arange(-90,90,latres)+latres/2.0
    lons=np.arange(-180,180,lonres)+lonres/2.0
    # how many lats, lons
    ny=len(lats)
    nx=len(lons)
    
    # Filter for removing cloudy entries
    cloud_filter= omi_clouds < 0.4
    
    SC      = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_gc   = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VC_omi  = np.zeros([ny,nx],dtype=np.double)+np.NaN
    VCC     = np.zeros([ny,nx],dtype=np.double)+np.NaN
    counts  = np.zeros([ny,nx],dtype=np.int)
    for i in range(ny):
        for j in range(nx):
            matches=(omi_lats >= lat_bounds[i]) & (omi_lats < lat_bounds[i+1]) & (omi_lons >= lon_bounds[j]) & (omi_lons < lon_bounds[j+1])
            if remove_clouds:
                matches = matches & cloud_filter
            # if no pixels in this grid square, continue
            counts[i,j]= np.sum(matches)
            if counts[i,j] < 1:
                continue
            # Save the means of each good grid pixel
            SC[i,j]     =np.mean(omi_SC[matches])
            VC_gc[i,j]  =np.mean(omi_VC_gc[matches])
            VC_omi[i,j] =np.mean(omi_VC_omi[matches])
            VCC[i,j]    =np.mean(omi_VCC[matches])
            #AMF_gc[i,j] =np.mean(omi_AMF_gc[matches])
            #AMF_omi[i,j]=np.mean(omi_AMF_omi[matches])
            # TODO: store analysis data for saving, when we decide what we want analysed
    
    ## 5) 
    # Save one day averages to file
    outd=dict()
    outd['VC_OMI']      =VC_omi
    outd['VC_GC']       =VC_gc
    outd['SC']          =SC
    outd['VCC']         =VCC
    outd['gridentries'] =counts
    outd['latitude']    =lats
    outd['longitude']   =lons
    outd['RSC']         =ref_sec_correction
    outd['RSC_latitude']=ref_lat_bins
    outd['RSC_GC']      =GC_ref_sec
    outd['RSC_region']  =np.array([-90, -160, 90, -140])
    outfilename=fio.determine_filepath(date,latres=latres,lonres=lonres,reprocessed=True,oneday=True)
    
    if __VERBOSE__:
        print("sending day average to be saved:")
        print(outd.keys())
    fio.save_to_hdf5(outfilename, outd)
    
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
        yyyymmdd=date.strftime("%Y%m%d")
        filename=fio.determine_filepath(day,latres=latres,lonres=lonres,oneday=True,reprocessed=True)
        assert os.path.isfile(filename), "ERROR: file not found for averaging : "+filename
        files8.append(filename)
    
    # normal stuff will be identical between days
    normallist=['latitude','longitude','RSC_latitude','RSC_region']
    # list of things we need to add together and average
    sumlist=['AMF_GC','AMF_OMI','SC','VC_GC','VC_OMI','VCC']
    # other things need to be handled seperately
    otherlist=['gridentries','RSC', 'RSC_GC']
    
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
        sumdict['gridentries'] = addArraysWithNans(sumdict['gridentries'], daycount)
        sumdict['RSC'] = addArraysWithNans(sumdict['RSC'], data['RSC'])
        sumdict['RSC_GC'] = sumdict['RSC_GC'] + data['RSC_GC']
        
        # for each averaged amount we want to sum together the totals
        for key in sumlist:
            # y is today's total amount ( gridcell avgs * gridcell counts )
            y=data[key].astype(np.float64) * daycount
            # add arrays treating nans as zero using masked arrays
            sumdict[key]= addArraysWithNans(sumdict[key],y)
    
    # take the average and save out to netcdf
    avgdict=dict()
    counts=sumdict['gridentries']
    for key in sumlist:
        avgdict[key]= sumdict[key] / counts.astype(float)
    for key in normallist:
        avgdict[key]= sumdict[key]
    avgdict['gridentries']=counts.astype(int)
    avgdict['RSC']=sumdict['RSC']/8.0
    avgdict['RSC_GC']=sumdict['RSC_GC']/8.0
    outfilename=fio.determine_filepath(date,latres=latres,lonres=lonres,oneday=False,reprocessed=True)
    fio.save_to_hdf5(outfilename, avgdict)
    print("File Saved: "+outfilename)


def Reprocess_N_days(date, latres=0.25, lonres=0.3125, days=8, processes=8, remove_clouds=True):
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
    if __VERBOSE__:
        print ("Pool Created ")
    
    #arguments: date,latres,lonres,getsum
    # function arguments in a list, for each of N days
    inputs = [(dd, latres, lonres, remove_clouds) for dd in daysN]
    
    # run all at once since each day can be independantly processed
    results = [ pool.apply_async(create_omhchorp_1, args=inp) for inp in inputs ]
    if __VERBOSE__:
        print("apply_async called for each day")
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    if __VERBOSE__:
        print("Close and Join called for Pool")
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
            for y in range(ny-1)+1: # don't worry about top and bottom row
                mask_copy[y,x] = np.sum(mask[[y-1,y,y+1],[x-1,x,(x+1)%ny]]) > 0
        return mask_copy
    
    # read reprocessed:
    
    # read fires
    fires, flats, flons = fio.read_8dayfire_interpolated(date,latres=latres,lonres=lonres)
    # create a mask in squares with fires or adjacent to fires
    mask = fires > 0
    retmask = set_adjacent_to_true(mask)
    
    return retmask

def get_16day_first_mask(date, latres=0.25, lonres=0.3125):
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

