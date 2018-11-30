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

# my file reading library
from utilities import fio
import utilities.utilities as util
from utilities import plotting as pp

###############
### GLOBALS ###
###############

__VERBOSE__=True
__DEBUG__=False


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


        data=fio.read_omhchorp(day0,dayn,keylist=keylist,latres=latres,lonres=lonres)

        # Make all the data attributes of this class
        for k in data.keys():
            setattr(self, k, np.squeeze(np.array(data[k])))

        self.n_times = len(util.list_days(day0,dayn))

        self.oceanmask=util.oceanmask(self.lats,self.lons)


    def inds_subset(self, lat0=None, lat1=None, lon0=None, lon1=None, maskocean=False, maskland=False):
        ''' return indices of lat, lon arrays within input box '''
        inds=np.ones([self.n_lats,self.n_lons]).astype(np.bool) # array of trues

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

    def background_averaged(self, key='VCC_GC', lats=None):
        '''
            return average HCHO over a specific region
            In same units as VCC [ Molecules/cm2 ]
        '''
        region=util.__REMOTEPACIFIC__
        if lats is not None:
            region[0]=lats[0]
            region[2]=lats[1]
        # find the average HCHO column over the __REMOTEPACIFIC__
        inds = self.region_subset(region, maskocean=False, maskland=False)
        attr=getattr(self,key)
        BG=np.nanmean(attr[inds])
        return BG

    def get_background_array(self, key='VCC_GC', lats=None, lons=None):
        '''
            Return background for key based on average over pacific region
            for given latitudes (can be gridded with longitudes)

            Whatever time dimension we have gets averaged over for
                pacific background returned on lats,lons grid
        '''
        if __VERBOSE__:
            print("In omhchorp.get_background_array()")
        # Average pacific ocean longitudinally
        lon0=util.__REMOTEPACIFIC__[1] # left lon
        lon1=util.__REMOTEPACIFIC__[3] # right lon

        if lats is None:
            lats=self.lats
        if lons is None:
            lons=self.lons

        # VCC Data looks like [[time,] lat, lon]

        # grab VCC over the ocean.
        oceanlons= (self.lons >= lon0) * (self.lons <= lon1)
        Data=getattr(self, key)

        if __VERBOSE__:
            print(key,".shape:")
            print (Data.shape)
        if self.n_times>1:
            # average this stuff over the time dim
            if __VERBOSE__:
                dstrs=tuple([self.dates[0].strftime('%Y%m%d'),self.dates[-1].strftime('%Y%m%d')])
                print("omhchorp.get_background_array() averaging over %s-%s"%dstrs)
            oceanVCC=Data[:,:,oceanlons]
            oceanVCC=np.nanmean(oceanVCC,axis=0)
        else:
            oceanVCC=Data[:,oceanlons]

        if __VERBOSE__:
            print("oceanVCC.shape")
            print(oceanVCC.shape)

        # Average the pacific strip longitudinally
        pacific_strip=np.nanmean(oceanVCC,axis=1)
        if __VERBOSE__:
            print("Pacific strip array shape:")
            print(pacific_strip.shape)
        # get background interpolated to whatever latitude
        background=np.interp(lats,self.lats,pacific_strip,left=np.NaN,right=np.NaN)
        # grid it to longitude so we have the background on our grid (longitudinally equal)
        mbackground=background.repeat(len(lons)).reshape([len(lats),len(lons)])

        return mbackground

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
        elif len(self.dates) > 1:
            data=np.nanmean(data,axis=0)
            counts=np.nansum(counts,axis=0)
            dsum=np.nansum(dsum,axis=0)

        ni = len(self.lats)
        nj = len(self.lons)

        new_ni, new_nj = int(ni/factor),int(nj/factor)
        newarr=np.zeros([new_ni,new_nj])+ np.NaN
        newcounts=np.zeros([new_ni, new_nj])
        surface_area=np.zeros([new_ni,new_nj])
        for i in range(new_ni):
            ir = np.arange(i*factor,i*factor+factor)
            for j in range(new_nj):
                jr = np.arange(j*factor, j*factor+factor)
                newcounts[i,j] = np.nansum(counts[ir,jr])
                newarr[i,j] = np.nansum(dsum[ir,jr])
                surface_area[i,j] = np.sum(self.surface_areas[ir,jr])
        # Sum divided by entry count, ignore div by zero warning
        with np.errstate(divide='ignore'):
            newarr = newarr/newcounts
            newarr[newcounts==0.0]=np.NaN
        lats=self.lats[0::factor]
        lons=self.lons[0::factor]

        lats_e=util.edges_from_mids(lats)
        lons_e=util.edges_from_mids(lons)
        return {key:newarr, 'counts':newcounts,
                'surface_areas':surface_area,
                'lats':lats, 'lons':lons,
                'lats_e':lats_e, 'lons_e':lons_e}

    def time_averaged(self, day0, dayn=None, keys=['VCC'],
                      month=False, weighted=True):
        '''
            Return keys averaged over the time dimension
                Where date >= day0 and date <= dayn
                or whole month if month==True
            Just pass in day0 for one day of data
            If weighted then weight the average by gridentries
        '''
        ret={}
        dates=np.array(self.dates)

        if len(dates)==1:
            # one day only, no time dim
            for key in keys:
                ret[key]=getattr(self,key)
            ret['gridentries']=self.gridentries
            if hasattr(self,'ppentries'):
                ret['ppentries']=self.ppentries
            return ret

        # option to do whole month:
        if month:
            dayn=util.last_day(day0)
        if dayn is None:
            dayn=day0


        dinds = (dates >= day0) * (dates <= dayn)
        assert np.sum(dinds)>0, "omhchorp.time_averaged() averaged zero days!"
        if __VERBOSE__:
            print("omhchorp.time_averaged() averaging %d days"%np.sum(dinds))
            print("from %s to %s"%(day0.strftime('%Y%m%d'),dayn.strftime('%Y%m%d')))

        entries=self.gridentries[dinds] # entries for each day
        totentries=np.nansum(entries,axis=0) # total entries over time dim
        ret['gridentries']=totentries
        if hasattr(self,'ppentries'):
            ppentries=self.ppentries[dinds] # entries for each day
            pptotentries=np.nansum(ppentries,axis=0) # total entries over time dim
            ret['ppentries']=pptotentries

        actual={}
        flat={}
        for key in keys:
            data=getattr(self,key)[dinds]
            with np.errstate(divide='ignore', invalid='ignore'):
                actual[key]=np.nansum(data*entries, axis=0)/totentries
                if key=='VCC_PP':
                    actual[key]=np.nansum(data*ppentries, axis=0)/pptotentries
            flat[key]=np.nanmean(data, axis=0)
            ret[key]=[flat[key],actual[key]][weighted]
            #TODO: remove once checked
            if __VERBOSE__:
                print("time_average... Flat:%.2e  VS  actual: %.2e"%(np.nanmean(flat[key]),np.nanmean(actual[key])))

        return ret

    def plot_map(self,key='VCC',day0=None,dayn=None,region=pp.__AUSREGION__,**cmargs):
        '''
            plot key over region averaged on day [to dayn]
        '''

        if day0 is None:
            day0=self.dates[0]
        if dayn is None:
            dayn=day0
        data=self.time_averaged(day0,dayn,keys=[key,],)[key]
        lati,loni=util.lat_lon_range(self.lats,self.lons,region=region)
        data=data[lati,:]
        data=data[:,loni]
        lats=self.lats[lati]
        lons=self.lons[loni]

        return pp.createmap(data, lats, lons, make_edges=False, latlon=True,
              region=region, linear=True, **cmargs)


if __name__=='__main__':

    om=omhchorp(datetime(2005,1,1))
    print("One day data shape: %s"%str(om.VCC.shape))
    om.plot_map(**{'pname':'map_test_1.png'})
    om=omhchorp(datetime(2005,1,1), dayn=datetime(2005,1,4))
    print("4 day data shape: %s"%str(om.VCC.shape))
    om.plot_map(**{'pname':'map_test_2.png'})

    landinds=om.inds_subset(maskocean=True)
    ausinds=om.inds_aus(maskocean=True)

    print("%d land elements"%np.sum(landinds))
    print("%d australia land elements"%np.sum(ausinds))

    om=omhchorp(datetime(2005,1,1),datetime(2005,1,31))
    om.plot_map(**{'pname':'map_test_3.png'})
