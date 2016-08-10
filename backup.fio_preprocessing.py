# -*- coding: utf-8 -*-

# Python script reading hdf5 OMI Vert Column dataset

## Modules
# module for hdf eos 5
import h5py 
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from scipy.interpolate import griddata

import timeit # to look at how long python code takes...
# lets use sexy sexy parallelograms
from multiprocessing import Pool

#just_global_things, good hashtag
datafieldsg = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/'
geofieldsg  = 'HDFEOS/GRIDS/OMI Total Column Amount HCHO/Geolocation Fields/'
datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

def save_to_hdf5(outfilename, arraydict, fillvalue=0.0):
    '''
    Takes a bunch of arrays, named in the arraydict parameter, and saves 
    to outfilename as hdf5 using h5py with fillvalue=0, gzip compression
    '''
    print("saving to "+outfilename)
    with h5py.File(outfilename,"w") as f:
        #grp=f.create_group('grid')
        for name in arraydict.keys():
            # create dataset, using arraydict values
            darr=arraydict[name]
            print (name, darr.shape, darr.dtype)
            
            # Fill array using darr,
            # this way takes 1.5 minutes to save, using 401 MB space
            dset=f.create_dataset(name,fillvalue=fillvalue,
                                  data=darr, compression_opts=9,
                                  chunks=True, compression="gzip")
            
        print ("Saved : ", f.keys())
    
    # TODO: some sort of units attribute creation
    #f.attrs.create('Units',)

def read_omhcho(path):
    '''
    Read info from a single swath file
    Returns:
        columnAmount, lats, lons, w(z), z, 
    '''
    
    # Total column amounts are in molecules/cm2
    # total vertical columns (TODO: Maybe use ref sector corrected VC????)
    field_hcho  = datafields+'ColumnAmount' 
    # other useful fields
    field_amf   = datafields+'AirMassFactor'
    field_apri   = datafields+'GasProfile'
    field_w     = datafields+'ScatteringWeights'
    field_qf    = datafields+'MainDataQualityFlag'
    field_xqf   = geofields +'XtrackQualityFlags'
    field_lon   = geofields +'Longitude'
    field_lat   = geofields +'Latitude'
    
    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        lats    = in_f[field_lat].value
        lons    = in_f[field_lon].value
        hcho    = in_f[field_hcho].value
        w       = in_f[field_w].value
        qf      = in_f[field_qf].value
        amf     = in_f[field_amf].value
        apri    = in_f[field_apri].value
        ## remove missing values and bad flags: 
        # QF: missing<0, suss=1, bad=2
        suss = qf != 0
        hcho[suss]=np.NaN
        lats[suss]=np.NaN
        lons[suss]=np.NaN    
        amf[suss] =np.NaN
    
    return hcho, lats, lons, w, amf, apri
    
def read_hchog(filepath):
    '''
    Function to read provided lvl 2 gridded product omhchog
    '''
    ## File to be read:
    fname=filepath
    #files=glob(folder+'/*.he5')

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
    in_f=h5py.File(fname,'r')
    
    ## get data arrays
    lats    = in_f[field_lat].value
    lons    = in_f[field_lon].value
    hcho    = in_f[field_hcho].value
    amf     = in_f[field_amf].value
    qf      = in_f[field_qf].value
    xf      = in_f[field_xf].value
    
    ## remove missing values and bad flags: 
    # QF: missing<0, suss=1, bad=2
    suss = qf != 0
    hcho[suss]=np.NaN
    lats[suss]=np.NaN
    lons[suss]=np.NaN
    
    return (hcho, lats, lons, amf, xf)

def read_regridded(date, oneday=False, latres=0.25, lonres=0.3125):
    '''
    Function to read the data from one of my regreeded files
    Inputs:
        date = datetime(y,m,d,0) of desired day file
        oneday  = False : set to True to read a single day, leave as False to read 8-day avg
        binsize = 0.25 : binsize
    '''
    # filepath based on date and day and binsize
    sbin= '%1.2fx%1.2f'% latres,lonres
    sday= ['8g','1g'][oneday]
    ftype= sday+sbin
    
    fpath = glob("omhcho_%s_%4d%02d%02d.he5" % ftype,date.year,date.month,date.day)
    with h5py.File(fpath,'r') as in_f:
        data    =in_f['ColumnAmountHCHO'].value
        amf     =in_f['AMF'].value
        count   =in_f['GridEntries'].value
        lats    =in_f['Latitude'].value
        lons    =in_f['Longitude'].value
        omega   =in_f['ScatteringWeight'].value
        shape   =in_f['ShapeFactor'].value
    
    return (data, lats, lons, count, amf, omega, shape)

def read_gchcho(date):
    '''
    Read the geos chem hcho column data into a structure
    '''
    
    fpath=glob('gchcho/hcho_%4d%02d.he5' % ( date.year, date.month ) )[0]
    with h5py.File(fpath, 'r') as in_f:
        dsetname='GC_UCX_HCHOColumns'
        dset=in_f[dsetname]
        #('TOTALCOLUMN' '<f8', (91, 144)), ('NUMBERDENSITYHCHO', '<f8', (72, 91, 144)), ('NORMALIZED', '<f8', (72, 91, 144)), ('LATITUDE', '<f4', (91,)), ('LONGITUDE', '<f4', (144,)), ('PEDGES', '<f8', (73, 91, 144)), ('PMIDS', '<f8', (72, 91, 144))])
        VC = dset['TOTALCOLUMN'].squeeze() # molecs/cm2
        eta = dset['NUMBERDENSITYHCHO'].squeeze() # number density profile
        S_z = dset['NORMALIZED'].squeeze() # normalized number density profile
        pmids= dset['PMIDS'].squeeze() # pressure mids (geometric) hPa
        pedges= dset['PMIDS'].squeeze() # pressure edges hPa
        lons= dset['LONGITUDE'].squeeze() # longitude and latitude midpoints
        lats= dset['LATITUDE'].squeeze()
    # create structure and return it
    return {'VC':VC,'eta':eta,'S_z':S_z,'lats':lats,'lons':lons,'pmids':pmids,'pedges':pedges}

def read_apriori(date, latres=0.25, lonres=0.3125):
    '''
    Read GC HCHO column and regrid to lat/lon res. temporal resolution is one month
    inputs:
        date= datetime
        latres, lonres for resolution of GC 2x2.5 hcho columns to be regridded onto
    '''
    gchcho=read_gchcho(date)
    lats=gchcho['lats']
    lons=gchcho['lons']
    S_z=gchcho['S_z']
    newlats= np.arange(-90,90, latres) + latres/2.0
    newlons= np.arange(-180,180, lonres) + lonres/2.0
    
    mlons,mlats = np.meshgrid(lons,lats) # this order keeps the lats/lons dimension order
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)    
    interp = np.zeros([72,len(newlats),len(newlons)])
    
    # interpolate at each pressure level...
    for ii in range(72):
        interp[ii,:,:] = griddata( (mlats.ravel(), mlons.ravel()), 
                                   S_z[ii,:,:].ravel(), 
                                   (mnewlats, mnewlons),
                                   method='nearest')
    # scipy griddata function used to interpolate
    # newlats newlons need to be [72,blah,blah]
    # Also old lats lons need to be [72, blah,blah]
    #vmlats=np.tile(mlats,[72,1,1])
    #vmlons=np.tile(mlons,[72,1,1])
    #vmnewlats=np.tile(mnewlats,[72,1,1])
    #vmnewlons=np.tile(mnewlons,[72,1,1])
    #print (vmlats.shape)
    #print (vmlons.shape)
    #print (S_z.shape)
    #print (vmnewlats.shape)
    #print (vmnewlons.shape)
    #interp = griddata( (vmlats.ravel(),vmlons.ravel( )), S_z.ravel(), (vmnewlats, vmnewlons), method='nearest')
    
    # return the normalisec density profile we need to recalculate AMF 
    return interp,newlats,newlons



def omhcho_1_day_gridded_test(date=datetime(2005,1,1,0),save=False,
                         latres=0.25,lonres=0.25, getsum=False, test=False):
  
    lons=np.arange(-180,180,lonres) + lonres/2.0
    lats=np.arange(-90,90,latres) + latres/2.0
    ny=len(lats)
    nx=len(lons)
    # after binned, shift these bins left by half a binsize so that pixels are
    # associated with closest lat/lon
    
    # molecs/cm2 [ lats, lons, candidates ]
    data=np.zeros([ny,nx])
    # amfs [ lats, lons, candidates ]    
    amf = np.zeros([ny,nx])
    # scattering weights [ lats, lons, levels, candidates ]
    omega=np.zeros([ny,nx,47])
    # a-priori shape factor
    shape=np.zeros([ny,nx,47])
    # candidate count [ lats, lons ]
    count=np.zeros([ny,nx],dtype=np.int)
    return (data,lats,lons, omega, amf, shape, count)

def omhcho_1_day_gridded(date=datetime(2005,1,1,0),save=False,
                         latres=0.25,lonres=0.25, getsum=False, test=False,
                         reprocess=False):
    '''
    Turn some L2 omhcho swaths into gridded jaz hands
    Inputs:
        date=day of regridding
        save=save to hdf
        latres=regridded bins latitude resolution
        lonres=regridded bins longitude resolution
        getsum= True to return sum of entries rather than average ( eg for longer averaging )
        test= just do one swath to see how long everything will take..
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
    
    # molecs/cm2 [ lats, lons, candidates ]
    data=np.zeros([ny,nx])
    # amfs [ lats, lons, candidates ]    
    amf = np.zeros([ny,nx])
    # scattering weights [ lats, lons, levels, candidates ]
    omega=np.zeros([ny,nx,47])
    # a-priori shape factor
    shape=np.zeros([ny,nx,47])
    # candidate count [ lats, lons ]
    count=np.zeros([ny,nx],dtype=np.int)
    
    ## Read 1 day of data ( 14 - 15 swath files )
    #
    folder='omhcho/'
    mask=folder + '*' + str(date.year) + 'm' + '%02d'%date.month + '%02d'%date.day + '*.he5'
    files = glob(mask)
    files.sort() # ALPHABETICAL PLEASE
    
    # read in each swath, sorting it into the lat/lon bins    
    if test:
        files=[files[0],]
    for ff in files:
        #start_time = timeit.default_timer()
        print ("Going through "+ff)
        fhcho, flat, flon, fw, famf, fs = read_omhcho(ff)
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
                    # eventually this is where we optionally alter the AMF and recalculate the vertical columns
                    # for each entry loop over recalculating AMF
                    
                    # Weights and Shape are 47 levels x N rows x 60 pixels x c entries
                    # store sum of scattering weights
                    omega[yi, xi, :] = np.nansum(fw[:, match ], axis=1)
                    # store sum of aprioris
                    shape[yi, xi, :] = np.nansum(fs[:, match ], axis=1)
                    
                    # store sum of vertical columns
                    data[yi,xi] = np.nansum(fhcho[ match ])
                    # store sum of AMFs
                    amf[yi,xi] = np.nansum(famf[ match ])
                    # add match count to local entry count
                    count[yi,xi] = count[yi,xi] + fcount 
        # END OF yi, xi loop
    # END OF ff file loop ( swaths in a day )
    
    # Divide sums by count to get daily averaged gridded data
    # divide by zero gives nan and runtime warning
    with np.errstate(divide='ignore', invalid='ignore'):
        mdata=np.true_divide(data,count)
        mamf=np.true_divide(amf,count)
        # (x,y,z) / (x,y)  cannot be broadcast, (z,y,x) / (y,x) can. This avoids a loop
        momega=np.transpose( np.true_divide(omega.transpose(), count.transpose()) )
        mshape=np.transpose( np.true_divide(shape.transpose(), count.transpose()) )
    
    # can double check that worked by comparing level zero
    mo0f = np.isfinite(momega[:,:,0])
    if ~ np.array_equal(momega[ mo0f,0] * count[mo0f], omega[ mo0f,0 ]):
        print (" mean*count ~= total, total abs dif = ", np.sum(np.abs(momega[ mo0f,0] * count[mo0f] - omega[ mo0f,0 ])))
    
    # save out to he5 file        
    if save:
        outfilename='omhcho_1g%1.2fx%1.2f_%4d%02d%02d.he5'%(latres, lonres, date.year,date.month,date.day)
        if test:
            outfilename='test_'+outfilename
            
        #Make a dictionary structure with all the stuff we want to store
        arraydict={'AMF':mamf,'ColumnAmountHCHO':mdata,'GridEntries':count, 
                   'Latitude':lats,'Longitude':lons,'ScatteringWeight':momega,
                   'ShapeFactor':mshape}
        start_time=timeit.default_timer()
        save_to_hdf5(outfilename, arraydict)
        elapsed = timeit.default_timer() - start_time
        print ("Took " + str(elapsed/60.0)+ " minutes to save ")
    
    # return the sum or the mean of the day
    if getsum:
        return ( data, lats, lons, omega, amf, shape, count )
    return (mdata, lats, lons, momega, mamf, mshape, count)

def create_omhcho_8_day_gridded(date=datetime(2005,1,1,0), latres=0.25,
                                lonres=0.25, parallel=0, savedaily=False):
    '''
    Take one day gridded data and put into x by x degree 8 day avg hdf5 files
    Uses my own hdf one day gridded stuff
    Inputs: (parameter=default)
        date= datetime(2005,1,1,0): start day of 8 day average
        latres= 0.25    : lat bins size, can be any integer multiple of 0.125
        lonres= 0.25    : as latres
        parallel= 0     : how many processors to use, Zero for no parallelism
        savedaily=False : save each daily gridded dataset to hdf output file
    '''

    # time how long it all takes
    start_time=timeit.default_timer()
    
    # first create our arrays with which we cumulate 8 days of data
    # number of lats and lons determined by binsize(regular grid)    
    ny=int(180/latres)
    nx=int(360/lonres)
    
    # molecs/cm2 [ lats, lons]
    data=np.zeros([ny,nx])
    # amfs [ lats, lons]
    amf = np.zeros([ny,nx])
    # scattering weights [ lats, lons, levels]
    omega=np.zeros([ny,nx,47])
    # a-priori shape factor
    shape=np.zeros([ny,nx,47])
    # candidate count [ lats, lons]
    count=np.zeros([ny,nx],dtype=np.int)

    # our 8 days in a list
    days8 = [ date + timedelta(days=dd) for dd in range(8)]
    if parallel<2:
        # Add everythign together for all 8 days
        for day, di in zip(days8, range(8)):
            ddata, lats, lons, domega, damf, dshape, dcount = \
                omhcho_1_day_gridded(date=day, save=savedaily,
                                     latres=latres, lonres=lonres, getsum=True)
            data = data + ddata
            omega = omega + domega
            amf= amf + damf
            shape=shape + dshape
            count=count+dcount
    else:
        # create pool of N processes
        pool = Pool(processes=parallel)
        
        #arguments: date,save,latres,lonres,getsum,reprocessed
        # function arguments in a list, for each of 8 days
        inputs = [(dd, savedaily, latres, lonres,True) for dd in days8]
        
        # run all at once (since days are fully independant)
        results = [ pool.apply_async(omhcho_1_day_gridded, args=inp) for inp in inputs ]
        print len(results), "before join"
        pool.close()
        pool.join()
        print len(results), "after join"
        # grab results, add them all together
        output = [p.get() for p in results]
        print (len(output), " p.get() results")
        print (len(output[0]), "arrays in each result (should be 7)")
        for di in range(8):
            ddata, lats, lons, domega, damf, dshape, dcount = output[di]
            data = data + ddata
            omega = omega + domega
            amf= amf + damf
            shape=shape + dshape
            count=count+dcount
            
    ## Take the average
    # Divide sums by count to get averaged gridded data
    # divide by zero gives nan and runtime warning
    with np.errstate(divide='ignore', invalid='ignore'):
        mdata=np.true_divide(data,count)
        mamf=np.true_divide(amf,count)
        momega=np.transpose( np.true_divide(omega.transpose(), count.transpose()) )
        mshape=np.transpose( np.true_divide(shape.transpose(), count.transpose()) )
    
    # can double check that worked by comparing level zero
    mo0f = np.isfinite(momega[:,:,0])
    if ~ np.array_equal(momega[ mo0f,0] * count[mo0f], omega[ mo0f,0 ]):
        print (" mean*count ~= total, total abs dif = ", np.sum(np.abs(momega[ mo0f,0] * count[mo0f] - omega[ mo0f,0 ])))
    
    # save out to he5 file        
    outfilename='omhcho_8g%1.2fx%1.2f_%4d%02d%02d.he5'%(latres,lonres,date.year,date.month,date.day)
    #Make a dictionary structure with all the stuff we want to store
    arraydict={'AMF':mamf,'ColumnAmountHCHO':mdata,'GridEntries':count, 
               'Latitude':lats,'Longitude':lons,'ScatteringWeight':momega,
               'ShapeFactor':mshape}
    
    # save the resulting 8-day gridded dataset
    save_to_hdf5(outfilename, arraydict)
    
    # print out how long whole process took
    elapsed = timeit.default_timer() - start_time
    print ("Took " + str(elapsed/60.0)+ " minutes to create 8-day gridded average ")
    
    # return nothing
    #return ( data, lats, lons, omega, amf, shape, count )



# original h5py saving method(replaced with better function which also saves space)
#with h5py.File(outfilename,"w") as f:
    #grp=f.create_group('grid')
    #f.create_dataset('AMF',data=amf, fillvalue=0.0, chunks=True, compression="gzip")
    #f.create_dataset('ColumnAmountHCHO',data=data, fillvalue=0.0, chunks=True, compression="gzip")
    #f.create_dataset('GridEntries',data=count, fillvalue=0.0, chunks=True, compression="gzip")
    #f.create_dataset('Latitude',data=lats, chunks=True, compression="gzip")
    #f.create_dataset('Longitude',data=lons, chunks=True, compression="gzip")
    #f.create_dataset('ScatteringWeight',data=omega, chunks=True, compression="gzip")
    #f.create_dataset('ShapeFactor', data=shape, chunks=True, compression="gzip")

##
# In order to test stuff it's easiest to run file in debug with breakpoints,
# Debug -> run while in spyder, anything uncommented past here will run

#results = omhcho_1_day_gridded(test=True)
#create_omhcho_8_day_gridded(parallel=2)
#res1=omhcho_1_day_gridded(date=datetime(2005,1,1,0),save=False,latres=0.25,lonres=0.3125, getsum=False, test=True)
#res2=omhcho_1_day_gridded(date=datetime(2005,1,1,0),save=False,latres=0.25,lonres=0.3125, getsum=True , test=True)

#print ("Result shapes from non summed 1day test:")
#for r in res1:
#    print(r.shape)
#print ("Result shapes from summed 1day test:")
#for r in res2:
#    print(r.shape)
