# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# module for hdf eos 5
#import h5py 
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import matplotlib.pyplot as plt

# local imports:
import plotting as pp

##################
#####GLOBALS######
##################
run_number={"tropchem":0,"UCX":1}
runs=["geos5_2x25_tropchem","UCX_geos5_2x25"]
paths=["/home/574/jwg574/OMI_regridding/Data/GC_Output/%s/trac_avg"%rstr for rstr in runs]
tropchem_dims=['Tau', 'Pressure', 'latitude', 'longitude']
tropchem_keys=tropchem_dims+['ANTHSRCENO', 'ANTHSRCECO', 'ANTHSRCEALK4', 
    'ANTHSRCEACET', 'ANTHSRCEMEK', 'ANTHSRCEALD2', 'ANTHSRCEPRPE', 'ANTHSRCEC3H8',
    'ANTHSRCECH2O', 'ANTHSRCEC2H6', 'BIOBSRCENO', 'BIOBSRCECO', 'BIOBSRCEALK4',
    'BIOBSRCEACET', 'BIOBSRCEMEK', 'BIOBSRCEALD2', 'BIOBSRCEPRPE', 'BIOBSRCEC3H8',
    'BIOBSRCECH2O', 'BIOBSRCEC2H6', 'BIOBSRCESO2', 'BIOBSRCENH3', 'BIOBSRCEBC',
    'BIOBSRCEOC', 'BIOFSRCENO', 'BIOFSRCECO', 'BIOFSRCEALK4', 'BIOFSRCEACET', 
    'BIOFSRCEMEK', 'BIOFSRCEALD2', 'BIOFSRCEPRPE', 'BIOFSRCEC3H8', 'BIOFSRCECH2O', 
    'BIOFSRCEC2H6', 'BIOFSRCESO2', 'BIOFSRCENH3', 'BIOGSRCEISOP', 'BIOGSRCEACET', 
    'BIOGSRCEPRPE', 'BIOGSRCEMONX', 'BIOGSRCEMBOX', 'BIOGSRCEAPIN', 'BIOGSRCEBPIN', 
    'BIOGSRCELIMO', 'BIOGSRCESABI', 'BIOGSRCEMYRC', 'BIOGSRCECARE', 'BIOGSRCEOCIM', 
    'BIOGSRCEHCOOH', 'BIOGSRCEACTA', 'BIOGSRCEALD2', 'BIOGSRCEOMON', 'BIOGSRCEMOHX', 
    'BIOGSRCEETOH', 'BIOGSRCEFARN', 'BIOGSRCEBCAR', 'BIOGSRCEOSQT', 'BIOGSRCECHBr3', 
    'BIOGSRCECH2Br2', 'BIOGSRCESSBr2', 'BXHGHT-$BXHEIGHT', 'BXHGHT-$AD', 
    'BXHGHT-$AVGW', 'BXHGHT-$N(AIR)', 'CH4-EMISCH4-TOT', 'CH4-EMISCH4-GAO', 
    'CH4-EMISCH4-COL', 'CH4-EMISCH4-LIV', 'CH4-EMISCH4-WST', 'CH4-EMISCH4-BFL', 
    'CH4-EMISCH4-RIC', 'CH4-EMISCH4-OTA', 'CH4-EMISCH4-BBN', 'CH4-EMISCH4-WTL',
    'CH4-EMISCH4-SAB', 'CH4-EMISCH4-OTN', 'DRYD-FLXO3df', 'DRYD-FLXCH2Odf', 
    'DRYD-VELO3dv', 'DRYD-VELCH2Odv', 'DXYPDXYP', 'IJ-AVG-$NO', 'IJ-AVG-$O3', 
    'IJ-AVG-$ISOP', 'IJ-AVG-$MVK', 'IJ-AVG-$MACR', 'IJ-AVG-$CH2O', 'IJ-AVG-$ISOPN', 
    'IJ-AVG-$RIP', 'IJ-AVG-$IEPOX', 'IJ-AVG-$NO2', 'IJ-AVG-$NO3', 'JV-MAP-$JHNO3', 
    'JV-MAP-$', 'PBLDEPTHPBL-M', 'PBLDEPTHPBL-L', 'PEDGE-$PSURF', 'PORL-L=$POX', 
    'PORL-L=$LOX', 'TR-PAUSETP-LEVEL', 'TR-PAUSETP-HGHT', 'TR-PAUSETP-PRESS' ]
tropchem_HCHO_keys=tropchem_dims+['BXHGHT-$BXHEIGHT', 'BXHGHT-$AD', 'BXHGHT-$AVGW', 
    'BXHGHT-$N(AIR)', 'DXYPDXYP', 'IJ-AVG-$ISOP', 'IJ-AVG-$CH2O', 'PEDGE-$PSURF', 
    'TR-PAUSETP-LEVEL']
# SOME KEYS DIDN'T GET THE 1E9 SCALING !?
tropchem_scaled_keys=['IJ-AVG-$ISOP','IJ-AVG-$CH2O']

UCX_dims=['time','lev','lat','lon']
UCX_keys=UCX_dims+['IJ_AVG_S__O3','IJ_AVG_S__ISOP','IJ_AVG_S__MVK','IJ_AVG_S__MACR',
    'IJ_AVG_S__CH2O','PEDGE_S__PSURF','CHEM_L_S__LBRO2H','PORL_L_S__POX',
    'BXHGHT_S__BXHEIGHT','BXHGHT_S__AD','BXHGHT_S__AVGW','BXHGHT_S__N_AIR_', 
    'DXYP__','TR_PAUSE__TP_LEVEL','TR_PAUSE__TP_HGHT',
    'TR_PAUSE__TP_PRESS','CH4_LOSS__']
UCX_HCHO_keys=UCX_dims+['IJ_AVG_S__ISOP','IJ_AVG_S__CH2O','PEDGE_S__PSURF',
    'BXHGHT_S__BXHEIGHT','BXHGHT_S__AD','BXHGHT_S__AVGW','BXHGHT_S__N_AIR_',
    'DXYP__','TR_PAUSE__TP_LEVEL']
UCX_scaled_keys=[]
__VERBOSE__=True


################
###FUNCTIONS####
################
def lat_lon_range(lats,lons,region):
    '''
    returns indices of lats, lons which are within region: list=[S,W,N,E]
    '''
    S,W,N,E = region
    lats,lons=np.array(lats),np.array(lons)
    latinds1=np.where(lats>=S)[0]
    latinds2=np.where(lats<=N)[0]
    latinds=np.intersect1d(latinds1,latinds2, assume_unique=True)
    loninds1=np.where(lons>=W)[0]
    loninds2=np.where(lons<=E)[0]
    loninds=np.intersect1d(loninds1,loninds2, assume_unique=True)
    return latinds, loninds

def findrange(data,lats,lons,region):
    '''
    return vmin, vmax of data[lats,lons] within region: list=[SWNE]
    '''
    latinds,loninds=lat_lon_range(lats,lons,region)
    data2=data[latinds,:]
    data2=data2[:,loninds]
    vmin=np.nanmin(data2)
    vmax=np.nanmax(data2)
    return vmin,vmax

def date_from_gregorian(greg):
    ''' 
        gregorian = "hours since 1985-1-1 00:00:0.0"
        Returns datetime object or list of datetimes
    '''
    d0=datetime(1985,1,1,0,0,0)
    if isinstance(greg, (list, tuple, np.ndarray)):
        return([d0+timedelta(hours==hr) for hr in greg])
    return (d0+timedelta(hours==greg))
def index_from_gregorian(gregs, date):
    '''
        Return index of date within gregs array
    '''
    greg =(date-datetime(1985,1,1,0,0,0)).days * 24.0
    if __VERBOSE__:
        print("gregorian %s = %.2e"%(date.strftime("%Y%m%d"),greg))
        print(gregs)
    ind=np.where(gregs==greg)[0]
    return (ind)

def read_UCX():
    ''' Read the UCX netcdf file: '''
    fi=run_number['UCX']
    filename='trac_avg_UCX.nc'
    fullpath="%s/%s"%(paths[fi], filename)
    print("Reading %s"%fullpath)
    ucxfile=nc.Dataset(fullpath,'r')
    return(ucxfile)
 
def read_tropchem(date=datetime(2005,1,1)):
    ''' read output file'''
    dstr=date.strftime("%Y%m%d")
    fi=run_number['tropchem']
    filename='trac_avg.geos5_2x25_tropchem.%s0000.nc'%dstr
    fullpath="%s/%s"%(paths[fi],filename)
    print("Reading %s"%fullpath)
    tropfile=nc.Dataset(fullpath,'r')
    return(tropfile)

def get_tropchem_data(date=datetime(2005,1,1), keys=tropchem_HCHO_keys, 
    monthmean=True, surface=False):
    ''' return a subset of the tropchem data '''
    tf=read_tropchem(date)
    # read dimensions:
    #alt     = tf.variables['altitude'][:]
    #time    = tf.variables['time'][:]
    # lat/lons are grid midpoints
    lat     = tf.variables['latitude'][:]
    lon     = tf.variables['longitude'][:]
    Tau     = tf.variables['Tau'][:] # Tau(time): hours since 19850101:0000
    Press   = tf.variables['Pressure'][:] # Pressure(altitude): hPa, midpoints
    
    # get the subset of data: (most are [[lev],lat,lon,time])
    data={}
    
    for key in keys:
        scale=[1.0, 1e9][key in tropchem_scaled_keys] # scale if needed
        tmp=tf.variables[key][:]*scale
        if __VERBOSE__:
            print("%s originally [%s], with min, mean, max = (%.2e, %.2e %.2e)"%
                (key, str(tmp.shape), np.nanmin(tmp), np.nanmean(tmp), np.nanmax(tmp)))
        # take the monthmean
        if monthmean:
            dims=np.shape(tmp)
            Tdim=len(dims)-1 # time is final dimension
            if dims[Tdim] == len(Tau): # check we have the time dimension
                tmp=np.nanmean(tmp,axis=Tdim)
                if __VERBOSE__:
                    print("%s averaged over time dimension(%2d)"%(key,Tdim))
            else:
                if __VERBOSE__:
                    print ("%s has no time dimension"%(key))
        if surface:
            dims=np.shape(tmp)
            if (len(dims) > 0) and (dims[0] == 47):
                tmp=tmp[0] # take surface slice
                if __VERBOSE__:
                    print("%s had surface slice taken"%(key))
            else:
                if __VERBOSE__:
                    print ("%s has no level dimension"%(key))
        data[key]=tmp
        if __VERBOSE__:
            print("%s has shape: %s"%(key,str(np.shape(data[key]))))
        
    # return the data we want
    tf.close()
    return(data)

def get_UCX_data(date=datetime(2005,1,1), keys=UCX_HCHO_keys, surface=False):
    ''' get a month of UCX output '''
    dstr=date.strftime("%Y%m%d")
    uf=read_UCX()
    # read dimensions:
    # lat/lons are grid midpoints
    lat     = uf.variables['lat'][:]
    lon     = uf.variables['lon'][:]
    Tau     = uf.variables['time'][:] # Tau(time): hours since 19850101:0000
    Press   = uf.variables['lev'][:] # Pressure(altitude): hPa, midpoints
    
    # get the subset of data: (most are [time,[lev],lat,lon])
    data={}
    
    # find the month index:
    #dates=date_from_gregorian(Tau)
    di=index_from_gregorian(Tau,date)
    if __VERBOSE__:
        print ("date index = ")
        print (di)
    for key in keys:
        scale=[1.0, 1e9][key in UCX_scaled_keys] # scale if needed
        tmp=uf.variables[key][:]*scale
        if __VERBOSE__:
            print("%s originally [%s], with min, mean, max = (%.2e, %.2e %.2e)"%
                (key, str(tmp.shape), np.nanmin(tmp), np.nanmean(tmp), np.nanmax(tmp)))
        dims=np.shape(tmp)
        Tdim=0 # time is first dimension
        
        if dims[Tdim] == len(Tau): # check we have the time dimension
            tmp=np.squeeze(tmp[di])
            if __VERBOSE__:
                print("%s date extracted(%s), now shape=%s"%(key,dstr,str(tmp.shape)))
        else:
            if __VERBOSE__:
                print ("%s has no time dimension"%(key))
        # Take surface level if desired
        if surface:
            dims=np.shape(tmp)
            if (len(dims) > 0) and (dims[0] == len(Press)):
                tmp=np.squeeze(tmp[0]) # take surface slice
                if __VERBOSE__:
                    print("%s had surface slice taken"%(key))
            else:
                if __VERBOSE__:
                    print ("%s has no level dimension"%(key))
        data[key]=tmp
        if __VERBOSE__:
            print("%s now has shape: %s"%(key,str(np.shape(data[key]))))
        
    # return the data we want
    uf.close()
    return(data)

def compare_GC_plots(date=datetime(2005,1,1)):
    ''' maps of UCX and tropchem surface HCHO'''
    ausregion=pp.__AUSREGION__ # [S W N E]
    dstr=date.strftime("%Y%m%d")
    
    # First get tropchem data:
    #
    tdat=get_tropchem_data(date=date,monthmean=True,surface=True)
    thcho=tdat['IJ-AVG-$CH2O']
    tlat=tdat['latitude']
    tlon=tdat['longitude']
    # determine min and max:
    tvmin,tvmax = np.nanmin(thcho), np.nanmax(thcho)
    print("Global tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))
    tvmin,tvmax = findrange(thcho,tlat,tlon, ausregion)
    print("Aus tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))
    
    # Then get UCX data:
    #
    udat=get_UCX_data(date=date,surface=True)
    print(val for val in udat)
    uhcho=udat['IJ_AVG_S__CH2O']
    print (np.shape(uhcho))
    ulat=udat['lat']
    ulon=udat['lon']
    assert (np.array_equal(ulat,tlat)) and (np.array_equal(ulon,tlon)), "LATS AND LONS DIFFER"
    
    # determine min and max:
    uvmin,uvmax = np.nanmin(uhcho), np.nanmax(uhcho)
    print("Global UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    uvmin,uvmax = findrange(uhcho,ulat,ulon, ausregion)
    print("Aus UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    vmin,vmax=np.min([uvmin,tvmin]),np.max([uvmax,tvmax])
    
    # Figures with 4 subplots
    f,axes=plt.subplots(2,2,figsize=(14,14))
    kwargs={'vmin':vmin,'vmax':vmax,'linear':True}
    # first is tropchem
    plt.sca(axes[0,0]) 
    m,cs,cb=pp.ausmap(thcho,tlat,tlon, **kwargs)
    plt.title('tropchem surface')
    cb.set_label('ppbv')
    
    # second is UCX
    plt.sca(axes[0,1])
    m,cs,cb=pp.ausmap(uhcho,ulat,ulon, **kwargs)
    plt.title('UCX surface')
    cb.set_label('ppbv')
    
    # Third is diffs:
    plt.sca(axes[1,0])
    m,cs,cb = pp.ausmap(uhcho-thcho, tlat, tlon, **kwargs)
    plt.title('UCX - tropchem')
    cb.set_label('ppbv')

    # Fourth is rel diff:
    plt.sca(axes[1,1])
    kwargs['vmin']=-10; kwargs['vmax']=10
    m,cs,cb = pp.ausmap((uhcho-thcho)/thcho*100, tlat, tlon, **kwargs)
    plt.title('100*(UCX - tropchem)/tropchem')
    cb.set_label('% difference')
    
    
    pname='Figs/GC/tropchem_hcho_%s.png'%dstr
    plt.suptitle('HCHO %s'%dstr)
    plt.savefig(pname)
    print("%s saved"%pname)

__VERBOSE__=False
pp.InitMatplotlib()
compare_GC_plots()

#get_UCX_data(surface=True)

