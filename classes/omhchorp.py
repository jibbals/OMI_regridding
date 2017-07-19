#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:07:55 2017

Class for omhchorp analysis

@author: jesse
"""

### LIBRARIES/MODULES ###

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from mpl_toolkits.basemap import maskoceans #Basemap, maskoceans
#from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
from datetime import datetime, timedelta

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

# my file reading library
from utilities import fio
import utilities.utilities as util
sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=True
__DEBUG__=False

# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Change to -175 to avoid crossing the 179 -> -179 boundary?
__REMOTEPACIFIC__=[-15, -180, 15, -120]

########################################################################
########################  OMHCHORP CLASS ###############################
########################################################################
class omhchorp:
    '''
    Class for holding OMI regridded, reprocessed dataset
    generally time, latitude, longitude
    '''
    def __init__(self, day0, dayn=None, latres=0.25, lonres=0.3125, keylist=None, ignorePP=False):
        '''
        Read reprocessed OMI files, one month or longer
        Inputs:
            day0 = datetime(y,m,d) of first day to read in
            dayn = final day or None if just reading one day
            latres=0.25
            lonres=0.3125
            keylist=None : read these keys from the files, otherwise read all data
        Output:
            Structure containing omhchorp dataset
        '''
        # Read the days we want to analyse:
        daylist = util.list_days(day0, dayn) # includes last day.
        struct = []
        for day in daylist:
            struct.append(fio.read_omhchorp(date=day, oneday=True,
                                            latres=latres, lonres=lonres,
                                            keylist=keylist))
        # dates and dimensions
        self.dates=daylist
        self.lats=struct[0]['latitude']
        self.lons=struct[0]['longitude']
        self.lats_e = util.edges_from_mids(self.lats)
        self.lons_e = util.edges_from_mids(self.lons)

        nt,self.n_lats,self.n_lons=len(daylist), len(self.lats), len(self.lons)
        self.n_times=nt

        # Set all the data arrays in the same way, [[time],lat,lon]
        for k in fio.__OMHCHORP_KEYS__:
            if dayn is None: # one day only, no time dimension
                setattr(self, k, np.array(struct[0][k]))
            else:
                setattr(self, k, np.array([struct[j][k] for j in range(nt)]))

        # Reference Sector Correction latitudes don't change with time
        self.lats_RSC=struct[0]['RSC_latitude'] # rsc latitude bins
        self.RSC_region=struct[0]['RSC_region']


        # remove small and negative AMFs
        if not ignorePP:
            print("Removing %d AMF_PP's less than 0.1"%np.nansum(self.AMF_PP<0.1))
            self.AMF_PP[self.AMF_PP < 0.1]=np.NaN
            screen=[-5e15,1e17]
            screened=(self.VCC_PP<screen[0]) + (self.VCC_PP>screen[1])
            print("Removing %d VCC_PP's outside [-5e15,1e17]"%(np.sum(screened)))
            self.VCC_PP[screened]=np.NaN

        mlons,mlats=np.meshgrid(self.lons,self.lats)

        # True over ocean squares:
        check_arr=[self.VCC[0],self.VCC][dayn is None] # array with no time dim
        self.oceanmask=maskoceans(mlons,mlats,check_arr,inlands=False).mask

    def apply_fire_mask(self, use_8day_mask=False):
        ''' nanify arrays which are fire affected. '''
        mask = [self.fire_mask_16, self.fire_mask_8][use_8day_mask]
        for arr in [self.AMF_GC,self.AMF_OMI,self.AMF_GCz,self.SC,self.VC_GC,
                    self.VC_OMI,self.VC_OMI_RSC,self.VCC,
                    self.col_uncertainty_OMI]:
            arr[mask]=np.NaN

    def inds_subset(self, lat0=None,lat1=None,lon0=None,lon1=None, maskocean=False, maskland=False):
        ''' return indices of lat,lon arrays within input box '''
        inds=~np.isnan(self.AMF_OMI) # only want non nans
        mlons,mlats=np.meshgrid(self.lons,self.lats)
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
        oceanmask=self.oceanmask
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
        dy=self.lats[1]-self.lats[0]
        dx=self.lons[1]-self.lons[0]
        y0=self.lats[0]-dy/2.0
        x0=self.lons[0]-dx/2.0
        y1=self.lats[-1]+dy/2.0
        x1=self.lons[-1]+dx/2.0
        y=np.arange(y0,y1+0.00001,dy)
        x=np.arange(x0,x1+0.00001,dx)
        if y[0]<-90: y[0]=-89.999
        if y[-1]>90: y[-1]=89.999
        return y,x

    def inds_aus(self, maskocean=True,maskland=False):
        ''' return indices of Australia, with or without(default) ocean squares '''
        return self.inds_subset(lat0=-57,lat1=-6,lon0=101,lon1=160,maskocean=maskocean,maskland=maskland)

    def background_HCHO(self, lats=None):
        ''' return average HCHO over a specific region '''
        region=__REMOTEPACIFIC__
        if lats is not None:
            region[0]=lats[0]
            region[2]=lats[1]
        # find the average HCHO column over the __REMOTEPACIFIC__
        inds = self.region_subset(region, maskocean=False, maskland=False)

        BG=np.nanmean(self.VCC[inds])
        return BG

    def lower_resolution(self, key='VCC', factor=8, dates=None):
        '''
            return data with resolution lowered by a factor of 8 (or input any integer)
            This function averages out the time dimension from dates[0] to dates[1]
        '''
        # this can convert from 0.25 x 0.3125 to 2 x 2.5 resolutions
        data=getattr(self,key)
        counts=self.gridentries
        dsum=data*counts
        # Average over dates.
        if dates is not None:
            d=np.array(self.dates)
            ti = (d >= dates[0]) * (d < dates[1])
            if __VERBOSE__:
                print("omhchorp.lower_resolution averaging %d days"%np.sum(ti))
            data=np.nanmean(data[ti],axis=0)
            counts=np.nansum(counts[ti],axis=0)
            dsum=np.nansum(dsum[ti],axis=0)
        elif len(self.daylist > 1):
            data=np.nanmean(data,axis=0)
            counts=np.nansum(counts,axis=0)
            dsum=np.nansum(dsum,axis=0)

        ni = len(self.lats)
        nj = len(self.lons)

        new_ni, new_nj = int(ni/factor),int(nj/factor)
        newarr=np.zeros([new_ni,new_nj])+ np.NaN
        newcounts=np.zeros([new_ni, new_nj])
        for i in range(new_ni):
            ir = np.arange(i*factor,i*factor+factor)
            for j in range(new_nj):
                jr = np.arange(j*factor, j*factor+factor)
                newcounts[i,j] = np.nansum(counts[ir,jr])
                newarr[i,j] = np.nansum(dsum[ir,jr])
        # Sum divided by entry count, ignore div by zero warning
        with np.errstate(divide='ignore'):
            newarr = newarr/newcounts
            newarr[newcounts==0.0]=np.NaN
        lats=self.lats[0::factor]
        lons=self.lons[0::factor]

        lats_e=util.edges_from_mids(lats)
        lons_e=util.edges_from_mids(lons)
        return {key:newarr, 'counts':newcounts, 'lats':lats, 'lons':lons,
                'lats_e':lats_e, 'lons_e':lons_e}

    def time_averaged(self, day0, dayn=None, keys=['VCC'], month=False):
        '''
            Return keys averaged over the time dimension
                Where date >= day0 and date < dayn
                or whole month if month==True
        '''
        ret={}
        dates=np.array(self.dates)

        # option to do whole month:
        if month:
            dayn=util.last_day(day0) + timedelta(days=1)
        if dayn is None:
            dayn=dayn+timedelta(days=1)

        dinds = (dates >= day0) * (dates < dayn)
        if __VERBOSE__:
            print("omhchorp.time_averaged() averaging %d days"%np.sum(dinds))
            print("from %s to %s"%(day0.strftime('%Y%m%d'),dayn.strftime('%Y%m%d')))

        for key in keys:
            ret[key]=np.nanmean(getattr(self,key)[dinds], axis=0)
        return ret

if __name__=='__main__':

    om=omhchorp(datetime(2005,1,1))
    print("One day data shape: %s"%str(om.VCC.shape))
    om=omhchorp(datetime(2005,1,1), dayn=datetime(2005,1,4))
    print("4 day data shape: %s"%str(om.VCC.shape))

    landinds=om.inds_subset(maskocean=True)
    ausinds=om.inds_aus(maskocean=True)

    print("%d land elements"%np.sum(landinds))
    print("%d australia land elements"%np.sum(ausinds))

