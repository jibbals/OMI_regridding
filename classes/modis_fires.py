#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 2018

Class for reading fire masks from modis

@author: jesse
"""

### LIBRARIES/MODULES ###

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

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


########################################################################
########################  CLASS          ###############################
########################################################################
class fires:
    '''
    Read and arrange fire mask data from myd14a1 (modis aqua fire daily)
    '''
    def __init__(self, day0, dayn=None):
        '''
        Read reprocessed OMI files, one month or longer
        Inputs:
            day0 = datetime(y,m,d) of first day to read in
            dayn = final day or None if just reading one day
        '''
        # Read the days we want to analyse:
        # Files are saved in 8 day chunks...
        daylist = util.list_days(day0, dayn) # includes last day.
        struct = []

        floc  = 'Data/MOD14A1_D_FIRE/%4d/MOD14A1_D_FIRE_%s.CSV'
        flocs = [ floc%(d.year,d.strftime('%Y-%m-%d')) for d in daylist ]


        for filename in flocs:
            fire=fio.read_csv_p(filename)

            struct.append(fio.read_omhchorp(date=day, oneday=True,
                                            latres=latres, lonres=lonres,
                                            keylist=keylist))
        # dates and dimensions
        self.dates=daylist
        self.lats=struct[0]['latitude']
        self.lons=struct[0]['longitude']
        self.lat_res=self.lats[1]-self.lats[0]
        self.lon_res=self.lons[1]-self.lons[0]
        self.lats_e = util.edges_from_mids(self.lats)
        self.lons_e = util.edges_from_mids(self.lons)
        self.surface_areas=util.area_grid(self.lats,self.lons,self.lat_res,self.lon_res)
        nt,self.n_lats,self.n_lons=len(daylist), len(self.lats), len(self.lons)
        self.n_times=nt

        # Set all the data arrays in the same way, [[time],lat,lon]
        ret_keylist=struct[0].keys()
        for k in ret_keylist:
            if nt ==1: # one day only, no time dimension
                setattr(self, k, np.squeeze(np.array(struct[0][k])))
            else:
                setattr(self, k, np.array([struct[j][k] for j in range(nt)]))
            if __VERBOSE__:
                print("Read from omhchorp: ",k, getattr(self,k).shape)

        # Reference Sector Correction latitudes don't change with time
        if 'RSC_latitude' in ret_keylist:
            self.lats_RSC=struct[0]['RSC_latitude'] # rsc latitude bins
        if 'RSC_region' in ret_keylist:
            self.RSC_region=struct[0]['RSC_region']

        # remove small and negative AMFs
        print("Removing %d AMF_PP's less than 0.1"%np.nansum(self.AMF_PP<0.1))
        self.AMF_PP[self.AMF_PP < 0.1]=np.NaN
        screen=[-5e15,1e17]
        screened=(self.VCC_PP<screen[0]) + (self.VCC_PP>screen[1])
        print("Removing %d VCC_PP's outside [-5e15,1e17]"%(np.sum(screened)))
        self.VCC_PP[screened]=np.NaN

        mlons,mlats=np.meshgrid(self.lons,self.lats)

        self.oceanmask=maskoceans(mlons,mlats,mlons,inlands=0).mask
        #if 'VCC' in ret_keylist:
        #    self.background=self.get_background_array()
        #self.apply_fire_mask()

    def apply_fire_mask(self, key='VCC', use_8day_mask=True):
        ''' nanify arrays which are fire affected. '''
        mask = [self.fire_mask_16, self.fire_mask_8][use_8day_mask]
        print ("fire mask:",mask.shape,np.nansum(mask))
        print(np.nansum(mask>0))

        data=getattr(self,key)

        print(key, data.shape, ' before firemask:',np.nanmean(data))

        #if len(data.shape) == 3:
        #    for i in range(data.shape[0]): # loop over time dim
        data[mask>0]=np.NaN
        print('     after firemask:',np.nanmean(data))

        #for arr in [self.AMF_GC,self.AMF_OMI,self.AMF_GCz,self.SC,self.VC_GC,
        #            self.VC_OMI,self.VC_OMI_RSC,self.VCC,
        #            self.col_uncertainty_OMI]:
        #    arr[mask]=np.NaN

    def inds_subset(self, lat0=None, lat1=None, lon0=None, lon1=None, maskocean=False, maskland=False):
        ''' return indices of lat, lon arrays within input box '''
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

    def background_VCC_averaged(self, lats=None):
        '''
            return average HCHO over a specific region
            In same units as VCC [ Molecules/cm2 ]
        '''
        region=__REMOTEPACIFIC__
        if lats is not None:
            region[0]=lats[0]
            region[2]=lats[1]
        # find the average HCHO column over the __REMOTEPACIFIC__
        inds = self.region_subset(region, maskocean=False, maskland=False)

        BG=np.nanmean(self.VCC[inds])
        return BG

    def get_background_array(self, lats=None, lons=None):
        '''
            Return background HCHO based on average over pacific region
            for given latitudes (can be gridded with longitudes)

            Whatever time dimension we have gets averaged over for
                pacific background returned on lats,lons grid
        '''
        if __VERBOSE__:
            print("In omhchorp.get_background_array()")
        # Average pacific ocean longitudinally
        lon0=__REMOTEPACIFIC__[1] # left lon
        lon1=__REMOTEPACIFIC__[3] # right lon

        if lats is None:
            lats=self.lats
        if lons is None:
            lons=self.lons

        # VCC Data looks like [[time,] lat, lon]

        # grab VCC over the ocean.
        oceanlons= (self.lons >= lon0) * (self.lons <= lon1)
        if __VERBOSE__:
            print("VCC.shape:")
            print (self.VCC.shape)
        if self.n_times>1:
            # average this stuff over the time dim
            if __VERBOSE__:
                dstrs=tuple([self.dates[0].strftime('%Y%m%d'),self.dates[-1].strftime('%Y%m%d')])
                print("omhchorp.get_background_array() averaging over %s-%s"%dstrs)
            oceanVCC=self.VCC[:,:,oceanlons]
            oceanVCC=np.nanmean(oceanVCC,axis=0)
        else:
            oceanVCC=self.VCC[:,oceanlons]

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
        actual={}
        flat={}
        for key in keys:
            data=getattr(self,key)[dinds]
            with np.errstate(divide='ignore', invalid='ignore'):
                actual[key]=np.nansum(data*entries, axis=0)/totentries
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
            dayn=day
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
