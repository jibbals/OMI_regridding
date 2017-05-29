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
__VERBOSE__=True

def read_UCX():
    ''' Read the UCX netcdf file: '''
    fi=run_number['UCX']
    filename='trac_avg_UCX.nc'
    fullpath="%s/%s"%(paths[fi], filename)
    print("Reading %s"%fullpath)
    #ucxfile=h5py.File(fullpath,'r')
    ucxfile=nc.Dataset(fullpath,'r')
    #print(ucxfile)
    #h5py.Close(ucxfile)
    #ucxfile.close()
    return(ucxfile)
 
def read_tropchem(date=datetime(2005,1,1)):
    ''' read output file'''
    dstr=date.strftime("%Y%m%d")
    fi=run_number['tropchem']
    filename='trac_avg.geos5_2x25_tropchem.%s0000.nc'%dstr
    fullpath="%s/%s"%(paths[fi],filename)
    print("Reading %s"%fullpath)
    #tropfile=h5py.File(fullpath,'r')
    tropfile=nc.Dataset(fullpath,'r')
    #print(tropfile.data_model)
    #h5py.close(tropfile)
    #tropfile.close()
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
    return(data)

def compare_GC_plots(date=datetime(2005,1,1)):
    ''' maps of UCX and tropchem surface HCHO'''
    dstr=date.strftime("%Y%m%d")
    tdat=get_tropchem_data(date=date,monthmean=True,surface=True)
    hcho=tdat['IJ-AVG-$CH2O']
    lat=tdat['latitude']
    lon=tdat['longitude']
    vmin,vmax = np.nanmin(hcho), np.nanmax(hcho)
    print((vmin,vmax))
    vmin,vmax=0.01,4.5
    m,cs,cb=pp.ausmap(hcho,lat,lon, vmin=vmin, vmax=vmax,linear=True)
    plt.title('tropchem surface HCHO (%s)'%dstr)
    cb.set_label('ppbv')
    pname='Figs/GC/tropchem_hcho_%s.png'%dstr
    plt.savefig(pname)
    print("%s saved"%pname)

__VERBOSE__=False
pp.InitMatplotlib()
compare_GC_plots()

__VERBOSE__=True
#get_tropchem_data()
#tfile=read_tropchem()
#print([var for var in tfile.variables])
#tfile.close()
#ufile=read_UCX()

