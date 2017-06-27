#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
from mpl_toolkits.basemap import maskoceans #Basemap, maskoceans
#from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
#from datetime import datetime
#from glob import glob
#from scipy.interpolate import RectBivariateSpline as RBS
#import h5py

# my file reading library
import fio

__DEBUG__=False

#_keynames=["AMF_GC","AMF_GCz","AMF_OMI",...]

# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Changed to -175 to avoid crossing the 179 -> -179 boundary
__REMOTEPACIFIC__=[-15, -175, 15, -120]

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
        struct=fio.read_omhchorp(date, oneday=oneday, latres=latres, lonres=lonres, keylist=keylist, filename=filename)

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
        # The vertical column corrected using the RSC
        self.VCC=struct['VCC']
        self.VCC_PP=struct['VCC_PP'] # Corrected Paul Palmer VC

        # Arrays [ lats, lons ]
        self.AMF_GC=struct['AMF_GC']
        self.AMF_OMI=struct['AMF_OMI']
        self.AMF_GCz=struct['AMF_GCz']
        self.AMF_PP=struct['AMF_PP'] # AMF calculated using Paul palmers code
        # remove small and negative AMFs
        print("Removing %d AMF_PP's less than 0.1"%np.nansum(self.AMF_PP<0.1))
        self.AMF_PP[self.AMF_PP < 0.1]=np.NaN
        screen=[-5e15,1e17]
        screened=(self.VCC_PP<screen[0]) + (self.VCC_PP>screen[1])
        print("Removing %d VCC_PP's outside [-5e15,1e17]"%(np.sum(screened)))
        self.VCC_PP[screened]=np.NaN

        self.SC=struct['SC']
        self.VC_GC=struct['VC_GC']
        self.VC_OMI=struct['VC_OMI']
        self.VC_OMI_RSC=struct['VC_OMI_RSC']
        self.col_uncertainty_OMI=struct['col_uncertainty_OMI']
        self.fires=struct['fires']
        self.fire_mask_8=struct['fire_mask_8']      # true where fires occurred over last 8 days
        self.fire_mask_16=struct['fire_mask_16']    # true where fires occurred over last 16 days
        mlons,mlats=np.meshgrid(self.longitude,self.latitude)
        self.oceanmask=maskoceans(mlons,mlats,self.AMF_OMI,inlands=False).mask

    def apply_fire_mask(self, use_8day_mask=False):
        ''' nanify arrays which are fire affected. '''
        mask = [self.fire_mask_16, self.fire_mask_8][use_8day_mask]
        for arr in [self.AMF_GC,self.AMF_OMI,self.AMF_GCz,self.SC,self.VC_GC,self.VC_OMI,self.VC_OMI_RSC,self.VCC,self.col_uncertainty_OMI]:
            arr[mask]=np.NaN

    def inds_subset(self, lat0=None,lat1=None,lon0=None,lon1=None, maskocean=False, maskland=False):
        ''' return indices of lat,lon arrays within input box '''
        inds=~np.isnan(self.AMF_OMI) # only want non nans
        mlons,mlats=np.meshgrid(self.longitude,self.latitude)
        with np.errstate(invalid='ignore'): # ignore comparisons with NaNs
            if lat0 is not None:
                inds = inds * (mlats >= lat0)
            if lon0 is not None:
                inds = inds * (mlons >= lon0)
            if lat1 is not None:
                inds = inds * (mlats <= lat1)
            if lon1 is not None:
                inds = inds * (mlons <= lon1)

        # true over ocean squares
        oceanmask=maskoceans(mlons,mlats,self.AMF_OMI,inlands=False).mask

        landmask = (~oceanmask)

        # mask ocean if flag is set
        if maskocean:
            inds *= (~oceanmask)
            if __DEBUG__:
                print("oceanmask:")
                print((type(oceanmask),oceanmask.shape))
                print( (inds * (~oceanmask)).shape )
                print((np.sum(oceanmask),np.sum(~oceanmask))) # true for ocean squares!

        if maskland:
            inds *= (~landmask)

        return inds

    def region_subset(self, region, maskocean=False, maskland=False):
        '''
            Return true where lats and lons are within region
            Can also mask ocean or land squares
            region=[S,W,N,E]
        '''
        return self.inds_subset(lat0=region[0],lat1=region[2],
                                lon0=region[1],lon1=region[3],
                                maskocean=maskocean, maskland=maskland)

    def latlon_bounds(self):
        ''' Return latedges and lonedges arrays '''
        dy=self.latitude[1]-self.latitude[0]
        dx=self.longitude[1]-self.longitude[0]
        y0=self.latitude[0]-dy/2.0
        x0=self.longitude[0]-dx/2.0
        y1=self.latitude[-1]+dy/2.0
        x1=self.longitude[-1]+dx/2.0
        y=np.arange(y0,y1+0.00001,dy)
        x=np.arange(x0,x1+0.00001,dx)
        if y[0]<-90: y[0]=-89.999
        if y[-1]>90: y[-1]=89.999
        return y,x

    def inds_aus(self, maskocean=True,maskland=False):
        ''' return indices of Australia, with or without(default) ocean squares '''
        return self.inds_subset(lat0=-57,lat1=-6,lon0=101,lon1=160,maskocean=maskocean,maskland=maskland)

    def background_HCHO(self):
        ''' return average HCHO over a specific region '''
        #TODO: implement
        # find the average HCHO column over the __REMOTEPACIFIC__
        inds = self.region_subset(__REMOTEPACIFIC__,
                                  maskocean=False,maskland=False)

        BG=np.nanmean(self.VCC[inds])
        return BG

    def lower_resolution(self, key='VCC', factor=8):
        ''' return data with resolution lowered by a factor of 8 (or input any integer)'''
        # this can convert from 0.25 x 0.3125 to 2 x 2.5 resolutions
        data=getattr(self,key)
        ni = len(self.latitude)
        nj = len(self.longitude)
        counts=self.gridentries
        dsum=data*counts
        new_ni, new_nj = int(ni/factor),int(nj/factor)
        newarr=np.zeros([new_ni,new_nj])+ np.NaN
        newcounts=np.zeros([new_ni, new_nj])
        for i in range(new_ni):
            ir = np.arange(i*factor,i*factor+factor)
            for j in range(new_nj):
                jr = np.arange(j*factor, j*factor+factor)
                newcounts[i,j] = np.nansum(counts[ir,jr])
                newarr[i,j] = np.nansum(dsum[ir,jr])
        newarr = newarr/newcounts
        lats=self.latitude[0::factor]
        lons=self.longitude[0::factor]
        return {key:newarr, 'counts':newcounts, 'lats':lats, 'lons':lons}


if __name__=='__main__':

    from datetime import datetime

    om=omhchorp(datetime(2005,1,1))
    landinds=om.inds_subset(maskocean=True)
    ausinds=om.inds_aus(maskocean=True)

    print("%d elements"%(720*1152))
    print("%d land elements"%np.sum(landinds))
    print("%d australia land elements"%np.sum(ausinds))

