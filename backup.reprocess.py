# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:05 2016

@author: jesse
"""

# module to read/write data to file
import fio

import numpy as np

# my module with class definitions and that jazz
#from classes import gchcho

import timeit # to look at how long python code takes...
# lets use sexy sexy parallelograms
import multiprocessing as mp
from multiprocessing import Pool
#from functools import partial

from datetime import timedelta, datetime
from glob import glob

# GLOBALS
__VERBOSE__=True # set to true for more print statements

def sum_dicts(d1,d2):
    '''
    Add two dictionaries together, only where keys match
    '''
    d3=dict()
    # for each key in both dicts
    for k in set(d1) & set(d2):
        d3[k]=d1[k]+d2[k]
    return d3
    

def calculate_amf_sigma(AMF_G, w, S, w_coords, S_coords, plotname=None):
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
    def w_(x):
        return (np.interp(x, w_coords[::-1], w[::-1]))
    def S_(x):
        return (np.interp(x, S_coords[::-1], S[::-1]))
        
    ## just use sum of w * S / delta s for a bunch of steps...
    
    #print("101 equidistant steps:")
    #coord101 = np.linspace(1,0,101)
    #int1= np.sum(w_(coord101)*S_(coord101)*(coord101[0]-coord101[1]))
    #print(int1)
    #print("using omi 47 sigma levels:")
    #int2 = np.sum(vstep( (w_coords[0:-2] + w_coords[1:-1])/2.0, w_coords[0:-2] - w_coords[1:-1] ))
    #print(int2)
    #print("using geos 72 sigma levels:")
    mids= (S_coords[0:-2] + S_coords[1:-1]) / 2.0
    lens= (S_coords[0:-2] - S_coords[1:-1])    
    int3 = np.sum(w_(mids)*S_(mids) * lens)
    #print(int3)
    
    AMF_new = AMF_G * int3
    return(AMF_new)


def run_omhchorp_1_notshared(date=datetime(2005,1,1,0), latres=0.25, lonres=0.3125):
    '''
    Reprocess omhcho l2 day into reprocessed sum, save to a dictionary of arrays and return them
    Inputs: 
        shd = shared dictionary
        shl = shared lock
        date=day of regridding
        latres=regridded bins latitude resolution
        lonres=regridded bins longitude resolution
    '''
    
    # time how long it all takes
    start_time=timeit.default_timer()
    print("starting omhchorp_1 on day %s"%date.strftime('%Y %m %d'))
    
    # bin swath lat/lons into grid
    # Add extra bin to catch the NANs
    lonbins=np.arange(-180,180+lonres+0.001,lonres)
    latbins=np.arange(-90,90+latres+0.001,latres)
    
    ## set up new latitude longitude grids
    #
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    # bins are left shifted by half a binsize so that pixels are
    # associated with closest lat/lon
    
    # some data can be preloaded
    tempgchcho = fio.read_gchcho(date)
    gc_shape_s, gc_lats, gc_lons, gc_sigma = tempgchcho.get_apriori(latres=latres, lonres=lonres)
    
    sumdict=dict()
    sumdict['AMF_OMI'] =np.zeros([ny,nx])
    sumdict['AMF_GC']  =np.zeros([ny,nx])
    sumdict['AMF_SSD'] =np.zeros([ny,nx])
    sumdict['ColumnAmountHCHO']=np.zeros([ny,nx])
    sumdict['ColumnAmountHCHO_OMI']=np.zeros([ny,nx])
    sumdict['GridEntries']=np.zeros([ny,nx])
    
    ## arrays to hold data
    #
    # molecs/cm2 [ lats, lons ]
    print(sumdict['ColumnAmountHCHO'].shape)
    # amfs [ lats, lons ]
    print(sumdict['AMF_OMI'].shape)
    
    ## grab our GEOS-Chem apriori info dimension: [ levs, lats, lons ]
    gchcho = fio.read_gchcho(date)
    gc_shape_s, gc_lats, gc_lons, gc_sigma = gchcho.get_apriori(latres=latres, lonres=lonres)
    
    
    # Metadata for analysis:
    # disable if space is too much of an issue and in a rush
    saveMetaData=True
    if saveMetaData:
        locd=dict()
        locd['ColumnAmountHCHO_OMI']=np.zeros([ny,nx])
        locd['ColumnAmount_SSD']=np.zeros([ny,nx])
        locd['PressureLevels_OMI']=np.zeros([ny,nx,47])
        locd['ScatteringWeight']=np.zeros([ny,nx,47])
        locd['ShapeFactor_OMI']=np.zeros([ny,nx,47]) 
        locd['Sigma_GC']=gc_sigma
        locd['Sigma_OMI']=np.zeros([ny,nx,47])
    
    ## determine which files we want using the date input
    #
    folder='omhcho/'
    mask=folder+date.strftime('*%Ym%m%d*.he5')
    print("reading files that match "+mask)
    files = glob(mask)
    files.sort() # ALPHABETICALLY Ordered ( chronologically also )
    
    ## Read 1 day of data by looping over 14 - 15 swath files
    ## read in each swath, sorting it into the lat/lon bins
    #
    for ff in files:
        # print each file name as we start processing it
        print ("Processing "+ff)
        
        # read in each swathe using fio.py method
        # hcho, lats, lons, amf, amfg, w, apri, plevs [(levels,)lons, lats, candidates]
        omhcho, flat, flon, omamf, omamfg, om_w, om_shape_z, om_plevs = fio.read_omhcho(ff,remove_clouds=True)
        latinds = np.digitize(flat,latbins)-1
        loninds = np.digitize(flon,lonbins)-1
        
        # Determine sigma coordinates from pressure levels
        om_sigma = np.zeros(om_plevs.shape)
        om_toa = om_plevs[-1, :, :]
        
        ## Surface pressure and sigma levels calculations from OMI satellite plevels
        # use level 0 as psurf although it is actually pressure mid of bottom level.. (I THINK)
        # I could calculate psurf from geometric midpoint estimate:
        #   sqrt(plev[1]*psurf)=plev[0] -> psurf=plev[0] ^ 2 / plev[1]
        om_surf_alt = om_plevs[0,:,:] ** 2 / om_plevs[1,:,:]
        om_surf = om_plevs[0,:,:] 
        om_diff = om_surf-om_toa
        for ss in range(47):
            om_sigma[ss,:,:] = (om_plevs[ss,:,:] - om_toa)/om_diff
        
        
        # lats[latinds[latinds<ny+1]] matches swath latitudes to gridded latitude
        # latinds[loninds==0] gives latitude indexes where lons are -179.875
        
        # for each latitude index yi, longitude index xi
        for yi in range(ny):
            for xi in range(nx):
                
                # Any swath pixels land in this bin?
                match = np.logical_and(loninds==xi, latinds==yi)
                
                # how many entries at this point from this file?
                fcount = np.sum(match)
                
                # if we have at least one entry, add them to binned set
                if fcount > 0:
                    ## slant columns are vertical columns x amf
                    slants = omhcho[match] * omamf[match]
                    #print ('slants shapes: %s'%str(slants.shape))
                    
                    ## alter the AMF to recalculate the vertical columns
                    ## loop over each scene recalculating AMF 
                    for scene in range(fcount):
                        ## new amfs: AMF_n = AMF_G * \int_0^1 om_w(s) * S_s(s) ds
                        slant= slants[scene]
                        om_sigmai = (om_sigma[:, match])[:,scene]
                        om_wi     = (om_w[:, match])[:,scene]
                        om_amfgi = omamfg[match][scene]
                        AMF_new = calculate_amf_sigma(om_amfgi, om_wi, gc_shape_s[:,yi,xi], 
                                                      om_sigmai, gc_sigma[:, yi, xi] )
                        OMHCHORPi = slant/AMF_new
                        
                        # store sum of new vertical columns
                        #omhchorp[yi,xi] = omhchorp[yi,xi]+OMHCHORPi
                        sumdict['ColumnAmountHCHO'][yi,xi] = sumdict['ColumnAmountHCHO'][yi,xi]+OMHCHORPi
                        # store sum of new AMFs in this column
                        sumdict['AMF_GC'][yi,xi] = sumdict['AMF_GC'][yi,xi]+AMF_new
                        
                        # store sum of squared differences
                        squarediff = (omamf[match][scene] - AMF_new)**2
                        sumdict['AMF_SSD'][yi,xi] = sumdict['AMF_SSD'][yi,xi]+squarediff
                        
                        if saveMetaData:
                            squarediff = (omhcho[match][scene] - OMHCHORPi)**2
                            locd['ColumnAmount_SSD'][yi,xi] = locd['ColumnAmount_SSD'][yi,xi]+squarediff
                        
                    
                    if saveMetaData:
                        # OMI dimensions are 47 levels x N rows x 60 pixels x c entries
                        # store sum of scattering weights
                        locd['ScatteringWeight'][yi,xi,:] = locd['ScatteringWeight'][yi,xi,:] + np.nansum(om_w[:, match ], axis=1)
                        
                        # store sum of aprioris
                        locd['ShapeFactor_OMI'][yi,xi,:] = locd['ShapeFactor_OMI'][yi,xi,:] + np.nansum(om_shape_z[:, match ], axis=1)
                        
                        # store sum of vertical columns
                        locd['ColumnAmountHCHO_OMI'][yi,xi] = locd['ColumnAmountHCHO_OMI'][yi,xi] + np.nansum(omhcho[match])
                        sumdict['ColumnAmountHCHO_OMI'][yi,xi] = sumdict['ColumnAmountHCHO_OMI'][yi,xi] + np.nansum(omhcho[match])
                        
                        # store sum of omi sigma levels and pressure levels
                        locd['Sigma_OMI'][yi,xi,:] = locd['Sigma_OMI'][yi,xi,:] + np.nansum(om_sigma[:, match ], axis=1)
                        locd['PressureLevels_OMI'][yi,xi,:] = locd['PressureLevels_OMI'][yi,xi,:] + np.nansum(om_plevs[:, match ], axis=1)
                        
                    # store sum of AMFs
                    sumdict['AMF_OMI'][yi,xi] = sumdict['AMF_OMI'][yi,xi] + np.nansum(omamf[ match ])
                    
                    # Keep count of scenes added to this grid box
                    #count[yi,xi] = count[yi,xi] + fcount 
                    sumdict['GridEntries'][yi,xi] = sumdict['GridEntries'][yi,xi] + fcount
                    
                    
        # END OF yi, xi loop
    # END OF ff file loop ( swath files )
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to reprocess one day")  
    if saveMetaData:
        outfilename=fio.determine_filepath(date, metaData=True)
        fio.save_to_hdf5(outfilename, locd)
    return sumdict

def create_omhchorp_8_notshared(date, latres=0.25, lonres=0.3125):
    '''
    Create reprocessed data, using manager shared dictionary
    '''
    
    # time how long it all takes
    if __VERBOSE__:
        print('starting 8 day reprocessing function')
    start_time=timeit.default_timer()
    
    # first create our arrays with which we cumulate 8 days of data
    # number of lats and lons determined by binsize(regular grid)    
    ## set up new latitude longitude grids
    #
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    
    # create pool of 8 processes, one for each day ( run with 8 or more cpus optimally )
    pool = Pool(processes=8)
    
    #arguments: date,latres,lonres,getsum
    # function arguments in a list, for each of 8 days
    inputs = [(dd, latres, lonres) for dd in days8]
    
    # run all at once (since processing a day is independant for each day)
    results = [ pool.apply_async(run_omhchorp_1_notshared, args=inp) for inp in inputs ]
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    
    # grab results, add them all together
    eightdaysum=results[0].get()
    for di in np.arange(7)+1:
        daysum = results[di].get()
        eightdaysum=sum_dicts(eightdaysum, daysum)
    
    # add some usefull info:
    eightdaysum['Latitude']=lats
    eightdaysum['Longitude']=lons
    ## grab our GEOS-Chem apriori info dimension: [ levs, lats, lons ]
    gchcho = fio.read_gchcho(date)
    gc_shape_s, _gc_lats, _gc_lons, _gc_sigma = gchcho.get_apriori(latres=latres, lonres=lonres)
    eightdaysum['ShapeFactor_GC']=gc_shape_s
    
    if __VERBOSE__:
        print ('8 Processes complete for '+date.strftime('%Y %m %d'))
        print('list of [key, value.shape], for shared dictionary')
        print([(key, eightdaysum[key].shape) for key in eightdaysum.keys()])
    
    # eightdaysum is now our summed 8 days of data, we need to save the AVERAGE
    
    ## change the dictionary to represent 8-day means 
    # Divide sums by count to get averaged gridded data
    # divide by zero gives nan and runtime warning, which we ignore
    with np.errstate(divide='ignore', invalid='ignore'):
        # lists of things to average:
        meanlist=['AMF_OMI','AMF_GC','ColumnAmountHCHO_OMI','ColumnAmountHCHO']
        omilist=['ScatteringWeight','ShapeFactor_OMI', 'Sigma_OMI']
        count=eightdaysum['GridEntries']
        for key in eightdaysum.keys():
            if key in meanlist:
                eightdaysum[key]=np.true_divide(eightdaysum[key],count)
            elif key in omilist:
                # take average on each OMI level
                temparr=np.zeros([ny,nx,47])
                for ll in range(47):
                    temparr[:,:,ll] = np.true_divide(eightdaysum[key][:,:,ll],count)
                assert eightdaysum[key].shape == temparr.shape, "omi temparr shape is wrong"
                eightdaysum[key]=temparr
            else:
                if __VERBOSE__:
                    print ("key unaveraged:%s"%key)
    print('max of count: %d'%np.nanmax(count))
    # save out to he5 file
    outfilename=fio.determine_filepath(date, latres, lonres, reprocessed=True, oneday=False)
    fio.save_to_hdf5(outfilename, eightdaysum)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to reprocess eight days")
    

# SEEMS NOT TO WORK!??!?!?!?
def create_omhchorp_8(date, latres=0.25, lonres=0.3125):
    '''
    Create reprocessed data, using manager shared dictionary
    '''
    
    # time how long it all takes
    if __VERBOSE__:
        print('starting 8 day reprocessing function')
    start_time=timeit.default_timer()
    
    # first create our arrays with which we cumulate 8 days of data
    # number of lats and lons determined by binsize(regular grid)    
    ## set up new latitude longitude grids
    #
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    
    # some data can be preloaded
    tempgchcho = fio.read_gchcho(date)
    gc_shape_s, gc_lats, gc_lons, gc_sigma = tempgchcho.get_apriori(latres=latres, lonres=lonres)
    
    # manager which has shared data
    mgr = mp.Manager()
    # SET UP SHARED DICTIONARY 
    #
    shd = mgr.dict()
    shd['AMF_OMI'] =np.zeros([ny,nx])
    shd['AMF_GC']  =np.zeros([ny,nx])
    shd['AMF_SSD'] =np.zeros([ny,nx])
    shd['ColumnAmountHCHO']=np.zeros([ny,nx])
    shd['GridEntries']=np.zeros([ny,nx])
    shd['Latitude']=lats
    shd['Longitude']=lons
    shd['ShapeFactor_GC']=gc_shape_s
    # Extra stuff will be saved in daily metadata files...
    
    # The Shared Lock, used before writing to shd
    shl = mgr.Lock()
    
    # NOW WE DEFINE OUR 1 DAY CREATION FUNCTION
    # DO THIS HERE SO THAT SHARED DICTIONARY IS AVAILABLE
    def run_omhchorp_1(day):
        '''
        Reprocess omhcho l2 day into reprocessed sum, saved in a shared array 
            USING SHARED DATA
        Inputs: 
            day=day of regridding
        '''
        
        if __VERBOSE__:
            print("starting omhchorp_1 on day %s"%day.strftime('%Y %m %d'))
            # time how long it all takes
            one_day_time=timeit.default_timer()
        
        ## grab our GEOS-Chem apriori info dimension: [ levs, lats, lons ]
        gchcho = fio.read_gchcho(day)
        gc_shape_s, gc_lats, gc_lons, gc_sigma = gchcho.get_apriori(latres=latres, lonres=lonres)
        
        # bin swath lat/lons into grid
        # Add extra bin to catch the NANs
        lonbins=np.arange(-180,180+lonres+0.001,lonres)
        latbins=np.arange(-90,90+latres+0.001,latres)
        
        # Metadata for analysis:
        # disable if space is too much of an issue and in a rush
        saveMetaData=True
        if saveMetaData:
            locd=dict()
            locd['ColumnAmountHCHO']=np.zeros([ny,nx])
            locd['ColumnAmountHCHO_OMI']=np.zeros([ny,nx])
            locd['ColumnAmount_SSD']=np.zeros([ny,nx])
            locd['PressureLevels_OMI']=np.zeros([ny,nx,47])
            locd['ScatteringWeight']=np.zeros([ny,nx,47])
            locd['ShapeFactor_OMI']=np.zeros([ny,nx,47]) 
            locd['Sigma_GC']=gc_sigma
            locd['Sigma_OMI']=np.zeros([ny,nx,47])
        
        ## determine which files we want using the day input
        #
        folder='omhcho/'
        mask=folder+day.strftime('*%Ym%m%d*.he5')
        print("reading files that match "+mask)
        files = glob(mask)
        files.sort() # ALPHABETICALLY Ordered ( chronologically also )
        
        ## Read 1 day of data by looping over 14 - 15 swath files
        ## read in each swath, sorting it into the lat/lon bins
        #
        for ff in files:
            # print each file name as we start processing it
            print ("Processing "+ff)
            
            # read in each swathe using fio.py method
            # hcho, lats, lons, amf, amfg, w, apri, plevs [(levels,)lons, lats, candidates]
            omhcho, flat, flon, omamf, omamfg, om_w, om_shape_z, om_plevs = fio.read_omhcho(ff)
            latinds = np.digitize(flat,latbins)-1
            loninds = np.digitize(flon,lonbins)-1
            
            # Determine sigma coordinates from pressure levels
            om_sigma = np.zeros(om_plevs.shape)
            om_toa = om_plevs[-1, :, :]
            
            ## Surface pressure and sigma levels calculations from OMI satellite plevels
            # use level 0 as psurf although it is actually pressure mid of bottom level.. (I THINK)
            # I could calculate psurf from geometric midpoint estimate:
            #   sqrt(plev[1]*psurf)=plev[0] -> psurf=plev[0] ^ 2 / plev[1]
            om_surf_alt = om_plevs[0,:,:] ** 2 / om_plevs[1,:,:]
            om_surf = om_plevs[0,:,:] 
            om_diff = om_surf-om_toa
            for ss in range(47):
                om_sigma[ss,:,:] = (om_plevs[ss,:,:] - om_toa)/om_diff
            
            
            # lats[latinds[latinds<ny+1]] matches swath latitudes to gridded latitude
            # latinds[loninds==0] gives latitude indexes where lons are -179.875
            
            # for each latitude index yi, longitude index xi
            for yi in range(ny):
                for xi in range(nx):
                    
                    # Any swath pixels land in this bin?
                    match = np.logical_and(loninds==xi, latinds==yi)
                    
                    # how many entries at this point from this file?
                    fcount = np.sum(match)
                    
                    # if we have at least one entry, add them to binned set
                    if fcount > 0:
                        ## slant columns are vertical columns x amf
                        slants = omhcho[match] * omamf[match]
                        #print ('slants shapes: %s'%str(slants.shape))
                        
                        ## alter the AMF to recalculate the vertical columns
                        ## loop over each scene recalculating AMF 
                        for scene in range(fcount):
                            ## new amfs: AMF_n = AMF_G * \int_0^1 om_w(s) * S_s(s) ds
                            slant= slants[scene]
                            om_sigmai = (om_sigma[:, match])[:,scene]
                            om_wi     = (om_w[:, match])[:,scene]
                            om_amfgi = omamfg[match][scene]
                            AMF_new = calculate_amf_sigma(om_amfgi, om_wi, gc_shape_s[:,yi,xi], 
                                                          om_sigmai, gc_sigma[:, yi, xi] )
                            OMHCHORPi = slant/AMF_new
                            
                            # LOCK WHILE WRITING TO SHARED DATA
                            shl.acquire()
                            
                            # store sum of new vertical columns
                            #omhchorp[yi,xi] = omhchorp[yi,xi]+OMHCHORPi
                            shd['ColumnAmountHCHO'][yi,xi] = shd['ColumnAmountHCHO'][yi,xi]+OMHCHORPi
                            # store sum of new AMFs in this column
                            shd['AMF_GC'][yi,xi] = shd['AMF_GC'][yi,xi]+AMF_new
                            
                            # store sum of squared differences
                            squarediff = (omamf[match][scene] - AMF_new)**2
                            shd['AMF_SSD'][yi,xi] = shd['AMF_SSD'][yi,xi]+squarediff
                            
                            if saveMetaData:
                                locd['ColumnAmountHCHO'][yi,xi]=locd['ColumnAmountHCHO'][yi,xi]+OMHCHORPi
                                squarediff = (omhcho[match][scene] - OMHCHORPi)**2
                                locd['ColumnAmount_SSD'][yi,xi] = locd['ColumnAmount_SSD'][yi,xi]+squarediff
                                
                            # RELEASE LOCK
                            shl.release()
                            
                        # ACQUIRE LOCK AGAIN FOR WRITES
                        shl.acquire()
                        
                        if saveMetaData:
                            # OMI dimensions are 47 levels x N rows x 60 pixels x c entries
                            # store sum of scattering weights
                            locd['ScatteringWeight'][yi,xi,:] = locd['ScatteringWeight'][yi,xi,:] + np.nansum(om_w[:, match ], axis=1)
                            
                            # store sum of aprioris
                            locd['ShapeFactor_OMI'][yi,xi,:] = locd['ShapeFactor_OMI'][yi,xi,:] + np.nansum(om_shape_z[:, match ], axis=1)
                            
                            # store sum of vertical columns
                            locd['ColumnAmountHCHO_OMI'][yi,xi] = locd['ColumnAmountHCHO_OMI'][yi,xi] + np.nansum(omhcho[match])
                            
                            # store sum of omi sigma levels and pressure levels
                            locd['Sigma_OMI'][yi,xi,:] = locd['Sigma_OMI'][yi,xi,:] + np.nansum(om_sigma[:, match ], axis=1)
                            locd['PressureLevels_OMI'][yi,xi,:] = locd['PressureLevels_OMI'][yi,xi,:] + np.nansum(om_plevs[:, match ], axis=1)
                            
                        # store sum of AMFs
                        shd['AMF_OMI'][yi,xi] = shd['AMF_OMI'][yi,xi] + np.nansum(omamf[ match ])
                        
                        # Keep count of scenes added to this grid box
                        #count[yi,xi] = count[yi,xi] + fcount 
                        shd['GridEntries'][yi,xi] = shd['GridEntries'][yi,xi] + fcount
                        
                        # RELEASE LOCK
                        shl.release()
                        
            # END OF yi, xi loop
        # END OF ff file loop ( swath files )
        
        # Save daily metadata for analysis
        if saveMetaData:
            outfilename=fio.determine_filepath(day, metaData=True)
            fio.save_to_hdf5(outfilename, locd)
            # free the memory used by locd
            del locd
        
        if __VERBOSE__:
            # print out how long whole process took
            one_day_elapsed = timeit.default_timer() - one_day_time
            print ("Took " + str(one_day_elapsed/60.0)+ " minutes to reprocess one day")
        return 0
    # END OF ONE DAY REPROCESS FUNCTION
    # 
    
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    
    ## create pool of 8 processes, one for each day ( run with 8 or more cpus optimally )
    #pool = Pool(processes=8)
    #
    ##arguments: date,save,latres,lonres,getsum,reprocessed
    ## function arguments in a list, for each of 8 days
    #inputs = [(dd, savedaily, latres, lonres,True) for dd in days8]
    #
    ## run all at once (since processing a day is independant for each day)
    #results = [ pool.apply_async(omhcho_1_day_reprocess, args=inp) for inp in inputs ]
    #pool.close()
    #pool.join() # wait until all jobs have finished(~ 70 minutes)
    #
    ## grab results, add them all together
    ##output = [p.get() for p in results] # crashes due to memory issue...
    #for di in range(8):
    #    daysum = results[di].get()
    
    #arguments: (shd, shl, date=datetime(2005,1,1,0), latres=0.25, lonres=0.3125):
    #inputs = [ (shd, shl, dd, latres, lonres) for dd in days8 ]
    inputs = [ (dd, ) for dd in days8 ]
    
    # run all at once (since processing a day is independant for each day)
    procs=[]
    
    #results.append( pool.apply_async(run_omhchorp_1, args=inp) )
    # create processes and start them
    procs = [mp.Process(target=run_omhchorp_1, args=inp) for inp in inputs]
    
    # start each process ( asynchronous ):
    for p in procs: p.start()
    
    # wait for each process to finish:
    for proc in procs: proc.join()
    
    if __VERBOSE__:
        print ('8 Processes complete for '+date.strftime('%Y %m %d'))
        print('list of [key, value.shape], for shared dictionary')
        print([(key, shd[key].shape) for key in shd.keys()])
    
    # shd is now our summed 8 days of data, we need to save the AVERAGE
    
    ## change the dictionary to represent 8-day means 
    # Divide sums by count to get averaged gridded data
    # divide by zero gives nan and runtime warning, which we ignore
    with np.errstate(divide='ignore', invalid='ignore'):
        # lists of things to average:
        meanlist=['AMF_OMI','AMF_GC','ColumnAmountHCHO_OMI','ColumnAmountHCHO']
        omilist=['ScatteringWeight','ShapeFactor_OMI', 'Sigma_OMI']
        count=shd['GridEntries']
        for key in shd.keys():
            if key in meanlist:
                shd[key]=np.true_divide(shd[key],count)
            elif key in omilist:
                # take average on each OMI level
                temparr=np.zeros([ny,nx,47])
                for ll in range(47):
                    temparr[:,:,ll] = np.true_divide(shd[key][:,:,ll],count)
                assert shd[key].shape == temparr.shape, "omi temparr shape is wrong"
                shd[key]=temparr
            else:
                if __VERBOSE__:
                    print ("key unaveraged:%s"%key)
    print('max of count: %d'%np.nanmax(count))
    # save out to he5 file
    outfilename=fio.determine_filepath(date, latres, lonres, reprocessed=True, oneday=False)
    fio.save_to_hdf5(outfilename, shd)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to reprocess eight days")
    
    
def omhcho_1_day_reprocess(date=datetime(2005,1,1,0), save=False, latres=0.25, lonres=0.3125, getsum=False):
    '''
    Turn L2 OMHCHO swath files into a gridded dataset
    Inputs: 
        date=day of regridding
        save=save to hdf
        latres=regridded bins latitude resolution
        lonres=regridded bins longitude resolution
        getsum= True to return sum of entries rather than average ( eg for longer averaging )
    Outputs: (  )
    '''
    
    # time how long it all takes
    start_time=timeit.default_timer()
    
    # bin swath lat/lons into grid
    # Add extra bin to catch the NANs
    lonbins=np.arange(-180,180+lonres+0.001,lonres)
    latbins=np.arange(-90,90+latres+0.001,latres)
    
    ## set up new latitude longitude grids
    #
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    # after binned, shift these bins left by half a binsize so that pixels are
    # associated with closest lat/lon
    
    ## arrays to hold data
    #
    # molecs/cm2 [ lats, lons ]
    omhchorg=np.zeros([ny,nx]) # regridded
    omhchorp=np.zeros([ny,nx]) # regridded and reprocessed
    omhchorp_sum_square_diffs=np.zeros([ny,nx]) # VC sum of squared change
    
    # amfs [ lats, lons ]
    amf = np.zeros([ny,nx])
    amf_gc = np.zeros([ny,nx])
    amf_sum_square_diffs = np.zeros([ny,nx])
    
    # scattering weights [ lats, lons, levels ]
    omega=np.zeros([ny,nx,47])
    
    # a-priori shape factor
    shape=np.zeros([ny,nx,47])
    
    # omi sigma levels and pressure levels
    sigma_omi = np.zeros([ny,nx,47])
    plevels_omi=np.zeros([ny,nx,47])
    
    # candidate count [ lats, lons ]
    count=np.zeros([ny,nx],dtype=np.int)
    
    ## grab our GEOS-Chem apriori info dimension: [ levs, lats, lons ]
    gchcho = fio.read_gchcho(date)
    gc_shape_s, gc_lats, gc_lons, gc_sigma = gchcho.get_apriori(latres=latres, lonres=lonres)
    
    ## determine which files we want using the date input
    #
    folder='omhcho/'
    mask=folder+date.strftime('*%Ym%m%d*.he5')
    print("reading files that match "+mask)
    files = glob(mask)
    files.sort() # ALPHABETICALLY Ordered ( chronologically also )
    
    ## Read 1 day of data by looping over 14 - 15 swath files
    ## read in each swath, sorting it into the lat/lon bins
    #
    for ff in files:
        # print each file name as we start processing it
        print ("Processing "+ff)
        
        # hcho, lats, lons, amf, amfg, w, apri, plevs [(levels,)lons, lats, candidates]
        omhcho, flat, flon, omamf, omamfg, om_w, om_shape_z, om_plevs = fio.read_omhcho(ff)
        latinds = np.digitize(flat,latbins)-1
        loninds = np.digitize(flon,lonbins)-1
        
        # Determine sigma coordinates from pressure levels
        om_sigma = np.zeros(om_plevs.shape)
        om_toa = om_plevs[-1, :, :]
        ## Surface pressure and sigma levels calculations from OMI satellite plevels
        # use level 0 as psurf although it is actually pressure mid of bottom level.. (I THINK)
        # I could calculate psurf from geometric midpoint estimate:
        #   sqrt(plev[1]*psurf)=plev[0] -> psurf=plev[0] ^ 2 / plev[1]
        om_surf_alt = om_plevs[0,:,:] ** 2 / om_plevs[1,:,:]
        om_surf = om_plevs[0,:,:] 
        om_diff = om_surf-om_toa
        for ss in range(47):
            om_sigma[ss,:,:] = (om_plevs[ss,:,:] - om_toa)/om_diff
        #print('sigma eg:')
        #print(om_sigma[:,10,10])
        
        # set all the top sigma's to zero...
        #om_sigma[-1,:,:] = 0
        
        # lats[latinds[latinds<ny+1]] matches swath latitudes to gridded latitude
        # latinds[loninds==0] gives latitude indexes where lons are -179.875
        #tested=False
        # for each latitude index yi, longitude index xi
        for yi in range(ny):
            for xi in range(nx):
                
                # Any swath pixels land in this bin?
                match = np.logical_and(loninds==xi, latinds==yi)
                
                # how many entries at this point from this file?
                fcount = np.sum(match)
                
                # if we have at least one entry, add them to binned set
                if fcount > 0:
                    ## slant columns are vertical columns x amf
                    slants = omhcho[match] * omamf[match]
                    #print ('slants shapes: %s'%str(slants.shape))
                    
                    ## alter the AMF to recalculate the vertical columns
                    ## loop over each scene recalculating AMF 
                    for scene in range(fcount):
                        ## new amfs: AMF_n = AMF_G * \int_0^1 om_w(s) * S_s(s) ds
                        slant= slants[scene]
                        om_sigmai = (om_sigma[:, match])[:,scene]
                        om_wi     = (om_w[:, match])[:,scene]
                        om_amfgi = omamfg[match][scene]
                        AMF_new = calculate_amf_sigma(om_amfgi, om_wi, gc_shape_s[:,yi,xi], 
                                                      om_sigmai, gc_sigma[:, yi, xi] )
                        OMHCHORPi = slant/AMF_new
                        
                        # store sum of new vertical columns
                        omhchorp[yi,xi] = omhchorp[yi,xi]+OMHCHORPi
                        
                        # store sum of new AMFs in this column
                        amf_gc[yi,xi] = amf_gc[yi,xi] + AMF_new
                        
                        # store sum of squared differences
                        squarediff = (omamf[match][scene] - AMF_new)**2
                        amf_sum_square_diffs[yi,xi] = amf_sum_square_diffs[yi,xi] + squarediff
                        squarediff = (omhcho[match][scene] - OMHCHORPi)**2
                        omhchorp_sum_square_diffs[yi,xi] =omhchorp_sum_square_diffs[yi,xi] + squarediff
                        
                        # For testing let's look at one of the columns in amf calculation
                        #if not tested:
                        #    AMF_new_test = calculate_amf_sigma( om_amfgi, om_wi,
                        #        gc_shape_s[:,yi,xi], om_sigmai, gc_sigma[:,yi,xi],
                        #        plotname= "AMF_TEST_PLOT_%s.png"%date.strftime("%Y%m%d"))
                        #    tested=True
                        
                        
                    # OMI dimensions are 47 levels x N rows x 60 pixels x c entries
                    # store sum of scattering weights
                    omega[yi, xi, :] = np.nansum(om_w[:, match ], axis=1)
                    
                    # store sum of aprioris
                    shape[yi, xi, :] = np.nansum(om_shape_z[:, match ], axis=1)
                    
                    # store sum of vertical columns
                    omhchorg[yi,xi] = np.nansum(omhcho[ match ])
                    
                    # store sum of omi sigma levels and pressure levels
                    sigma_omi[yi,xi,:] = np.nansum(om_sigma[:, match ], axis=1)
                    plevels_omi[yi,xi,:] = np.nansum(om_plevs[:, match ], axis=1)
                    
                    # store sum of AMFs
                    amf[yi,xi] = np.nansum(omamf[ match ])
                    
                    # Keep count of scenes added to this grid box
                    count[yi,xi] = count[yi,xi] + fcount 
                    
        # END OF yi, xi loop
    # END OF ff file loop ( swath files )
    
    # Shortcut for returning sums without saving means:
    sumdict={'AMF_OMI':amf,'AMF_GC':amf_gc,'AMF_SSD':amf_sum_square_diffs,
           'ColumnAmountHCHO_OMI':omhchorg, 'ColumnAmountHCHO':omhchorp,
           'ColumnAmount_SSD':omhchorp_sum_square_diffs,
           'GridEntries':count, 'Sigma_GC':gc_sigma, 'Sigma_OMI':sigma_omi,
           'PressureLevels_OMI':plevels_omi,
           'Latitude':lats,'Longitude':lons,'ScatteringWeight':omega,
           'ShapeFactor_OMI':shape, 'ShapeFactor_GC':gc_shape_s}
    if getsum and not save:
        # print out how long whole process took and return
        elapsed = timeit.default_timer() - start_time
        print ("Took " + str(elapsed/60.0)+ " minutes to reprocess one day")
        return sumdict
    
    # Divide sums by count to get daily averaged gridded data
    # divide by zero gives nan and runtime warning, lets ignore those
    with np.errstate(divide='ignore', invalid='ignore'):
        momhchorg=np.true_divide(omhchorg,count)
        momhchorp=np.true_divide(omhchorp,count)
        mamf=np.true_divide(amf,count)
        mamf_gc=np.true_divide(amf_gc,count)
        # (x,y,z) / (x,y)  cannot be broadcast, (z,y,x) / (y,x) can. This avoids a loop
        momega=np.transpose( np.true_divide(omega.transpose(), count.transpose()) )
        mshape=np.transpose( np.true_divide(shape.transpose(), count.transpose()) )
        msigma_omi=np.transpose( np.true_divide(sigma_omi.transpose(), count.transpose()) )
        mplevels_omi=np.transpose( np.true_divide(plevels_omi.transpose(), count.transpose()) )
    
    # can double check that worked by comparing level zero
    mo0f = np.isfinite(momega[:,:,0])
    if ~ np.array_equal(momega[ mo0f,0] * count[mo0f], omega[ mo0f,0 ]):
        print((" mean*count ~= total, total abs dif = ", np.sum(np.abs(momega[mo0f,0] * count[mo0f] - omega[ mo0f,0 ]))))
    
    #Make a dictionary structure with all the stuff we want to store
    meandict={'AMF_OMI':mamf,'AMF_GC':mamf_gc,'AMF_SSD':amf_sum_square_diffs,
               'ColumnAmountHCHO_OMI':momhchorg, 'ColumnAmountHCHO':momhchorp,
               'ColumnAmount_SSD':omhchorp_sum_square_diffs,
               'GridEntries':count, 'Sigma_GC':gc_sigma, 'Sigma_OMI':msigma_omi,
               'PressureLevels_OMI':mplevels_omi,
               'Latitude':lats,'Longitude':lons,'ScatteringWeight':momega,
               'ShapeFactor_OMI':mshape, 'ShapeFactor_GC':gc_shape_s}
    
    # save out to he5 file        
    if save:
        #outfilename='omhcho_1p%1.2fx%1.2f_%4d%02d%02d.he5'%(latres, lonres, date.year,date.month,date.day)
        outfilename=fio.determine_filepath(date, latres=latres, lonres=lonres, reprocessed=True, oneday=True)
        
        #start_time=timeit.default_timer()
        fio.save_to_hdf5(outfilename, meandict)
        #elapsed = timeit.default_timer() - start_time
        #print ("Took " + str(elapsed/60.0)+ " minutes to save ")
        print("Saved: %s"%outfilename)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to reprocess one day")
    
    # return the sum or mean
    if getsum:
        return sumdict
    
    return meandict

def create_omhcho_8_day_reprocessed(date=datetime(2005,1,1,0), latres=0.25,
                                lonres=0.3125, savedaily=False):
    '''
    Take one day gridded data and put into x by x degree 8 day avg hdf5 files
    Uses my own hdf one day gridded stuff
    Inputs:defaults
        date= datetime(2005,1,1,0) : start day of 8 day average
        latres= 0.25    : Must divide wholy into 180
        lonres= 0.3125  : Must divide wholy into 360
        parallel= 0     : how many processors to use, Zero for no parallelism
        savedaily=False : save each daily gridded dataset to hdf output file
    '''

    # time how long it all takes
    start_time=timeit.default_timer()
    
    # first create our arrays with which we cumulate 8 days of data
    # number of lats and lons determined by binsize(regular grid)    
    ny=int(180/latres)
    nx=int(360/lonres)
    
    sumDictionary = 0
    meanDictionary = 0
    
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    # create pool of 8 processes, one for each day ( run with 8 or more cpus optimally )
    pool = Pool(processes=8)
    
    #arguments: date,save,latres,lonres,getsum,reprocessed
    # function arguments in a list, for each of 8 days
    inputs = [(dd, savedaily, latres, lonres,True) for dd in days8]
    
    # run all at once (since processing a day is independant for each day)
    results = [ pool.apply_async(omhcho_1_day_reprocess, args=inp) for inp in inputs ]
    pool.close()
    pool.join() # wait until all jobs have finished(~ 70 minutes)
    
    # grab results, add them all together
    #output = [p.get() for p in results] # crashes due to memory issue...
    for di in range(8):
        daysum = results[di].get()
        if di == 0:
            sumDictionary= daysum
        else:
            sumDictionary=sum_dicts(sumDictionary, daysum)
    
    #print(sumDictionary.keys())
    #sumdict={'AMF_OMI':amf,'AMF_GC':amf_gc,'AMF_SSD':amf_sum_square_diffs,
    #       'ColumnAmountHCHO_OMI':omhchorg, 'ColumnAmountHCHO':omhchorp,
    #       'ColumnAmount_SSD':omhchorp_sum_square_diffs,
    #       'GridEntries':count, 'Sigma_GC':gc_sigma,
    #       'Latitude':lats,'Longitude':lons,'ScatteringWeight':omega,
    #       'ShapeFactor_OMI':shape, 'ShapeFactor_GC':gc_shape_s}
    #TODO Updaate this comment
    
    ## change the dictionary to represent 8-day means 
    # Divide sums by count to get averaged gridded data
    # divide by zero gives nan and runtime warning
    meanDictionary=sumDictionary
    with np.errstate(divide='ignore', invalid='ignore'):
        # lists of things to average:
        meanlist=['AMF_OMI','AMF_GC','ColumnAmountHCHO_OMI','ColumnAmountHCHO']
        omilist=['ScatteringWeight','ShapeFactor_OMI', 'Sigma_OMI']
        gclist=['ShapeFactor_GC',]
        count=sumDictionary['GridEntries']
        for key in sumDictionary.keys():
            if key in meanlist:
                meanDictionary[key]=np.true_divide(meanDictionary[key],count)
            elif key in omilist:
                # take average on each OMI level
                temparr=np.zeros([ny,nx,47])
                for ll in range(47):
                    temparr[:,:,ll] = np.true_divide(meanDictionary[key][:,:,ll],count)
                assert meanDictionary[key].shape == temparr.shape, "omi temparr shape is wrong"
                meanDictionary[key]=temparr
            elif key in gclist:
                # average each geos chem level for some thingies
                temparr=np.zeros([72,ny,nx])
                for ll in range(72):
                    temparr[ll,:,:] = np.true_divide(meanDictionary[key][ll,:,:],count)
                assert meanDictionary[key].shape == temparr.shape, "gc temparr shape is wrong"
                meanDictionary[key]=temparr
            else:
                print ("key unaveraged:%s"%key)
    
    # save out to he5 file
    outfilename=fio.determine_filepath(date, latres, lonres, reprocessed=True, oneday=False)
    fio.save_to_hdf5(outfilename, meanDictionary)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to create 8-day gridded average ")


def omhcho_1_day_regrid(date=datetime(2005,1,1,0),save=False,
                        latres=0.25,lonres=0.3125):
    '''
    Turn some L2 omhcho swaths into gridded jaz hands
    THIS METHOD TAKES AROUND 1 HOUR
    Inputs:
        date=day of regridding
        save=save to hdf (saves 1 day average)
        latres=regridded bins latitude resolution
        lonres=regridded bins longitude resolution
    Returns: (  )
        
    '''
    # bin swath lat/lons into grid
    # Add extra bin to catch the NANs
    lonbins=np.arange(-180,180+lonres+0.001,lonres)
    latbins=np.arange(-90,90+latres+0.001,latres)
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    # after binned, shift these bins left by half a binsize so that pixels are
    # associated with closest lat/lon
    
    # molecs/cm2 [ lats, lons ]
    data=np.zeros([ny,nx])
    # amfs [ lats, lons ]
    amf = np.zeros([ny,nx])
    amfg= np.zeros([ny,nx])
    # scattering weights [ lats, lons, levels ]
    omega=np.zeros([ny,nx,47])
    # pressure levels
    plevs=np.zeros([ny,nx,47])
    # a-priori shape factor
    shape=np.zeros([ny,nx,47])
    # candidate count [ lats, lons ]
    count=np.zeros([ny,nx],dtype=np.int)
    
    ## Read 1 day of data ( 14 - 15 swath files )
    #
    files = fio.determine_filepath(date)
    
    # read in each swath, sorting it into the lat/lon bins    
    for ff in files:
        print ("Going through "+ff)
        # hcho, lats, lons, amf, amfg, w, apri, plevs
        fhcho, flat, flon, famf, famfg, fw, fs, fplevs = fio.read_omhcho(ff)
        latinds = np.digitize(flat,latbins)-1
        loninds = np.digitize(flon,lonbins)-1
        
        # lats[latinds[latinds<ny+1]] matches swath latitudes to gridded latitude
        # latinds[loninds==0] gives latitude indexes where lons are -179.875
        
        # for each latitude index yi, longitude index xi
        for yi in range(ny):
            for xi in range(nx):
                
                # Any swath pixels land in this bin?
                match = np.logical_and(loninds==xi, latinds==yi)
                
                # how many entries at this point from this file?
                fcount = np.sum(match)
                
                # if we have at least one entry, add them to binned set
                if fcount > 0:
                    
                    # Weights and Shape are 47 levels x N rows x 60 pixels x c entries
                    # store sum of scattering weights
                    omega[yi, xi, :] = np.nansum(fw[:, match ], axis=1)
                    # store sum of plevels
                    plevs[yi, xi, :] = np.nansum(fplevs[:, match], axis=1)
                    # store sum of aprioris
                    shape[yi, xi, :] = np.nansum(fs[:, match ], axis=1)
                    
                    # store sum of vertical columns
                    data[yi,xi] = np.nansum(fhcho[ match ])
                    # store sum of AMFs
                    amf[yi,xi] = np.nansum(famf[ match ])
                    amfg[yi,xi]= np.nansum(famfg[ match ])
                    
                    # add match count to local entry count
                    count[yi,xi] = count[yi,xi] + fcount 
        # END OF yi, xi loop
    # END OF ff file loop ( swaths in a day )
    
    # save out to he5 file        
    if save:
        outfilename=fio.determine_filepath(date,latres=latres,lonres=lonres,regridded=True,oneday=True)
        
        # Divide sums by count to get daily averaged gridded data
        # divide by zero gives nan and runtime warning
        with np.errstate(divide='ignore', invalid='ignore'):
            mdata=np.true_divide(data,count)
            mamf=np.true_divide(amf,count)
            mamfg=np.true_divide(amfg,count)
            # (x,y,z) / (x,y)  cannot be broadcast, (z,y,x) / (y,x) can. This avoids a loop
            momega=np.transpose( np.true_divide(omega.transpose(), count.transpose()) )
            mshape=np.transpose( np.true_divide(shape.transpose(), count.transpose()) )
            mplevs=np.transpose( np.true_divide(plevs.transpose(), count.transpose()) )
        
        # can double check that worked by comparing level zero
        mo0f = np.isfinite(momega[:,:,0])
        if ~ np.array_equal(momega[ mo0f,0] * count[mo0f], omega[ mo0f,0 ]):
            print((" mean*count ~= total, total abs dif = ", np.sum(np.abs(momega[ mo0f,0] * count[mo0f] - omega[ mo0f,0 ]))))
        
        #Make a dictionary structure with all the stuff we want to store
        arraydict={'ColumnAmountHCHO':mdata,'GridEntries':count, 
                   'Latitude':lats,'Longitude':lons,'ScatteringWeight':momega,
                   'ShapeFactor':mshape, 'PressureLevels':mplevs,
                   'AMF':mamf, 'AMF_G':mamfg}
        
        fio.save_to_hdf5(outfilename, arraydict)
    
    # return sum of days' dataset
    arraydict={'ColumnAmountHCHO':data,'GridEntries':count, 
               'Latitude':lats,'Longitude':lons,'ScatteringWeight':omega,
               'ShapeFactor':shape, 'PressureLevels':plevs,
               'AMF':amf,'AMF_G':amfg}
    return arraydict

def create_omhcho_8_day_gridded(date=datetime(2005,1,1,0), latres=0.25,
                                lonres=0.3125, savedaily=False):
    '''
    Take one day gridded data and put into x by x degree 8 day avg hdf5 files
    Uses my own hdf one day gridded stuff
    Inputs:defaults
        date= datetime(2005,1,1,0) : start day of 8 day average
        latres= 0.25    : Must divide wholy into 180
        lonres= 0.3125  : Must divide wholy into 360
        parallel= 0     : how many processors to use, Zero for no parallelism
        savedaily=False : save each daily gridded dataset to hdf output file
    '''

    # time how long it all takes
    start_time=timeit.default_timer()
    
    # first create our arrays with which we cumulate 8 days of data
    # number of lats and lons determined by binsize(regular grid)    
    ny=int(180/latres)
    nx=int(360/lonres)
    
    
    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    
    # create pool of N processes
    pool = Pool(processes=8)
    
    #arguments: date,save,latres,lonres,getsum,reprocessed
    # function arguments in a list, for each of 8 days
    inputs = [(dd, savedaily, latres, lonres) for dd in days8]
    
    # run all at once (since days are fully independant)
    results = [ pool.apply_async(omhcho_1_day_regrid, args=inp) for inp in inputs ]
    print((len(results), "before join"))
    pool.close()
    pool.join()
    print((len(results), "after join"))
    # grab results, add them all together
    output = [p.get() for p in results]
    print((len(output), " p.get() results"))
    #returned dictionary: 
    #{'AMF','ColumnAmountHCHO','GridEntries','Latitude','Longitude','ScatteringWeight','ShapeFactor','PressureLevels'}
    for di in range(8):
        daysum = output[di]
        if di == 0:
            sumDictionary= daysum
        else:
            sumDictionary=sum_dicts(sumDictionary, daysum)
    print(sumDictionary.keys())
            
    ## change the dictionary to represent 8-day means 
    # Divide sums by count to get averaged gridded data
    # divide by zero gives nan and runtime warning
    meanDictionary=sumDictionary
    with np.errstate(divide='ignore', invalid='ignore'):
        # lists of things to average:
        meanlist=['AMF','AMF_G','ColumnAmountHCHO']
        vectorlist=['ScatteringWeight','ShapeFactor','PressureLevels'] # todo: add sigmas
        count=sumDictionary['GridEntries']
        for key in sumDictionary.keys():
            if key in meanlist:
                meanDictionary[key]=np.true_divide(meanDictionary[key],count)
            elif key in vectorlist:
                # take average on each OMI level
                temparr=np.zeros([ny,nx,47])
                for ll in range(47):
                    temparr[:,:,ll] = np.true_divide(meanDictionary[key][:,:,ll],count)
                assert meanDictionary[key].shape == temparr.shape, "omi temparr shape is wrong"
                meanDictionary[key]=temparr
            else:
                print ("key unaveraged:%s"%key)
    
    # save out to he5 file
    outfilename=fio.determine_filepath(date, latres, lonres, regridded=True, oneday=False)
    fio.save_to_hdf5(outfilename, meanDictionary)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to create 8-day gridded average ")
    



def apply_fires_mask(date=datetime(2005,1,1), latres=0.25, lonres=0.3125):
    '''
    1) Read reprocessed data
    2) read aqua 8 day fire count, for current and prior 8 days
    3a) return a mask showing where fire influence is expected
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
    firescur, flats, flons = fio.read_8dayfire_interpolated(date,latres=latres,lonres=lonres)
    # create a mask in squares with fires or adjacent to fires
    maskcur = firescur > 0
    retmask = set_adjacent_to_true(maskcur)
    
    # read prior 8 days fires:
    pridate=date-timedelta(days=8)
    if pridate > datetime(2005,1,1):
        firespri, flats, flons = fio.read_8dayfire_interpolated(pridate, latres=latres,lonres=lonres)
        maskpri=firespri > 0
        retmask = retmask | set_adjacent_to_true(maskpri)
    return retmask
