# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules

# module for hdf eos 5
#import h5py 
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from glob import glob # for file pattern reading

##################
#####GLOBALS######
##################
run_number={"tropchem":0,"UCX":1}
runs=["geos5_2x25_tropchem","UCX_geos5_2x25"]
paths=["/home/574/jwg574/OMI_regridding/Data/GC_Output/%s/trac_avg"%rstr for rstr in runs]
hemcodiag_path="/home/574/jwg574/OMI_regridding/Data/GC_Output/%s/hemco_diags"%runs[run_number['tropchem']]

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
tropchem_Isop_keys=tropchem_HCHO_keys+['IJ-AVG-$ISOP', 'IJ-AVG-$MVK', 'IJ-AVG-$MACR',
    'IJ-AVG-$NO2']
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
UCX_Isop_keys=UCX_HCHO_keys+['IJ_AVG_S__ISOP','IJ_AVG_S__MVK','IJ_AVG_S__MACR']
    
UCX_scaled_keys=[]
__VERBOSE__=False

################
###FUNCTIONS####
################

def date_from_gregorian(greg):
    ''' 
        gregorian = "hours since 1985-1-1 00:00:0.0"
        Returns list of datetimes
    '''
    d0=datetime(1985,1,1,0,0,0)
    greg=np.array(greg)
    #if isinstance(greg, (list, tuple, np.ndarray)):
    return([d0+timedelta(seconds=int(hr*3600)) for hr in greg])
    
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

def ppbv_to_molecs_per_cm2(ppbv, pedges):
    '''
    Inputs:
        ppbv[levs,lats,lons]
        pedges[levs]
    USING http://www.acd.ucar.edu.au/mopitt/avg_krnls_app.pdf
    NOTE: in pdf the midpoints are used rather than the edges
        due to how the AK is defined
    '''
    dims=np.shape(ppbv)
    N = dims[0]
    #P[i] - P[i+1] = pressure differences
    inds=np.arange(N)
    #THERE should be N+1 pressure edges since we have the ppbv for each slab
    diffs= pedges[inds] - pedges[inds+1]
    t = (2.12e13)*diffs # multiplication factor
    out=np.zeros(dims)
    # there's probably a good way to vectorise this, but not worth the time
    for x in range(dims[1]):
        for y in range(dims[2]):
            out[:,x,y] = t*ppbv[:,x,y]
    return out

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
    monthavg=True, surface=False):
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
        # take the monthavg
        if monthavg:
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

def read_HEMCO_diag(date=datetime(2005,1,1)):
    ''' 
        Read the HEMCO_diagnostics BIOGENIC ISOP EMISSIONS data 
        This is for tropchem run only, with output in units 
        float ISOP_BIOG(time, lat, lon) ;
            ISOP_BIOG:long_name = "ISOP_BIOG" ;
            ISOP_BIOG:units = "kg/m2/s" ;
            ISOP_BIOG:averaging_method = "mean" ;
            ISOP_BIOG:_FillValue = -1.e-31f ;
        netcdf hemco_diags.200502
        dimensions:
        	lat = 91 ;
        	lon = 144 ;
        	time = UNLIMITED ; // (672 currently)
        	lev = 47 ;
        variables:
        	float ISOP_BIOG(time, lat, lon) ;
        		ISOP_BIOG:long_name = "ISOP_BIOG" ;
        		ISOP_BIOG:units = "kg/m2/s" ;
        		ISOP_BIOG:averaging_method = "mean" ;
        		ISOP_BIOG:_FillValue = -1.e-31f ;
        	float time(time) ;
        		time:long_name = "Time" ;
        		time:units = "hours since 1985-01-01 00:00:00" ;
                
            
    '''
    # looks like hemco_diags/hemco_diags.200502.nc 
    dstr=date.strftime("%Y%m")
    fname="%s/hemco_diags.%s.nc"%(hemcodiag_path,dstr)
    print("Reading %s"%fname)
    tropfile=nc.Dataset(fname,'r')
    return(tropfile)

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

def determine_trop_column(ppbv, N_air, boxH, tplev):
    '''
        Inputs:
            ppbv[lev,lat,lon]: ppbv of chemical we want the trop column of
            N_air[lev,lat,lon]: number density of air (molec/m3)
            boxH[lev,lat,lon]: level heights (m)
            tplev[lat,lon]: where is tropopause
        Outputs:
            tropcol[lat,lon]: tropospheric column (molec/cm2)
    '''
    dims=np.shape(ppbv)
    
    # (molec_x/1e9 molec_air) * 1e9 * molec_air/m3 * m * m2/cm2
    X = ppbv * 1e-9 * N_air * boxH * 1e-4 # molec/cm2
        
    out=np.zeros([dims[1],dims[2]])
    for lat in range(dims[1]):
        for lon in range(dims[2]):
            trop=int(np.floor(tplev[lat,lon]))
            extra=tplev[lat,lon] - trop
            out[lat,lon]= np.sum(X[0:trop,lat,lon]) + extra*X[trop,lat,lon]
    return out

    
def _test_trop_column_calc():
    '''
    This tests the trop column calculation with a trivial case
    '''
    # lets check for a few dimensions:
    for dims in [ [10,1,1], [100,100,100]]:
        ppbv=np.zeros(dims)+100.    # molec/1e9molecA
        N_air=np.zeros(dims)+1e13   # molecA/m3
        boxH=np.zeros(dims)+1.      # m
        tplev=np.zeros([dims[1],dims[2]]) + 4.4
        # should give 100molec/cm2 per level : tropcol = 440molec/cm2
        out= determine_trop_column(ppbv, N_air, boxH, tplev)
        assert out.shape==(dims[1],dims[2]), 'trop column calc shape is wrong'
        assert np.isclose(out[0,0],440.0), 'trop column calc value is wrong, out=%f'%out[0,0]
        assert np.isclose(np.min(out), np.max(out)), 'Trop column calc has some problem'


if __name__=='__main__':
    _test_trop_column_calc()
