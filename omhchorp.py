import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
from datetime import datetime
from glob import glob
from scipy.interpolate import RectBivariateSpline as RBS
import h5py

# my file reading library
import fio


class omhchorp:
    '''
    Class for holding OMI regridded, reprocessed dataset
    Structure containing 
        double AMF_GC(lats, lons)       #
        double AMF_GCz(lats,lons)       # AMF using non rejigged lowest levels
        double AMF_OMI(lats, lons)      # 
        double AMF_SSD(lats, lons)
        double VC_GC(lats, lons)
        double VC_OMI(lats, lons) 
        double VCC(lats, lons) 
        int64 gridentries(lats, lons)   # how many entries in each gridbox
        double latitude(lats)
        double longitude(lons)
        double RSC(RSC_lats,60)
        double RSC_GC(RSC_lats)         # GEOS_Chem reference sector values (molecs/cm2)
        double RSC_latitude(RSC_lats)
        double RSC_region(4)            # [lat,lon, lat,lon]
        int64 fires
    '''
    def __init__(self, date, oneday=False, latres=0.25, lonres=0.3125, keylist=None, filename=None):
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
        fullkeylist=fio._omhchorp_keylist
        # read in all the requested information:
        if keylist is None:
            keylist=fullkeylist
        struct=dict.fromkeys(fullkeylist)
        if filename is None:
            fpath=fio.determine_filepath(date,oneday=oneday,latres=latres,lonres=lonres,reprocessed=True)
        else:
            fpath=filename
        self.fpath=fpath
        with h5py.File(fpath,'r') as in_f:
            #print('reading from file '+fpath)
            for key in keylist:
                try:
                    struct[key]=in_f[key].value
                except KeyError as ke: # if there is no matching key then print an error and continue
                    print("Key Error in %s"%fpath)
                    print(ke)
        # date and dimensions
        self.date=date
        self.latitude=struct['latitude']
        self.longitude=struct['longitude']
        self.gridentries=struct['gridentries']
        
        # Reference Sector Correction stuff
        self.RSC_latitude=struct['RSC_latitude']
        self.RSC_region=struct['RSC_region']
        self.RSC_GC=struct['RSC_GC'] 
        # [rsc_lats, 60]  - the rsc for this time period
        self.RSC=struct['RSC']
        
        # Arrays [ lats, lons ]
        self.AMF_GC=struct['AMF_GC']
        self.AMF_OMI=struct['AMF_OMI']
        self.AMF_GCz=struct['AMF_GCz']
        self.SC=struct['SC']
        self.VC_GC=struct['VC_GC']
        self.VC_OMI=struct['VC_OMI']
        self.VCC=struct['VCC']
        self.col_uncertainty_OMI=struct['col_uncertainty_OMI']
        self.fires=struct['fires']
        self.fire_mask_8=struct['fire_mask_8']
        self.fire_mask_16=struct['fire_mask_16']
    
    def apply_fire_mask(self, use_8day_mask=False):
        ''' nanify arrays which are fire affected. '''
        mask = [self.fire_mask_16, self.fire_mask_8][use_8day_mask]
        for arr in [self.AMF_GC,self.AMF_OMI,self.AMF_GCz,self.SC,self.VC_GC,self.VC_OMI,self.VCC,self.col_uncertainty_OMI]:
            arr[mask]=np.NaN
        
    
