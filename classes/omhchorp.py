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
from utilities import plotting as pp

sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=True
__DEBUG__=False

# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Change to -175 to avoid crossing the 179 -> -179 boundary?
__REMOTEPACIFIC__=[-15, -180, 15, -120]


# Coords for omhchorp:
__OMHCHORP_COORDS__=[
                     'latitude','longitude',
                     ]

# Keys for omhchorp:
__OMHCHORP_KEYS__ = [
    'gridentries',   # how many satellite pixels make up the pixel
    'ppentries',     # how many pixels we got the PP_AMF for
    'RSC',           # The reference sector correction [rsc_lats, 60]
    'RSC_latitude',  # latitudes of RSC
    'RSC_region',    # RSC region [S,W,N,E]
    'RSC_GC',        # GEOS-Chem RSC [RSC_latitude] (molec/cm2)
    'VCC',           # The vertical column corrected using the RSC
    'VCC_PP',        # Corrected Paul Palmer VC
    'AMF_GC',        # AMF calculated using by GEOS-Chem
    'AMF_GCz',       # secondary way of calculating AMF with GC
    'AMF_OMI',       # AMF from OMI swaths
    'AMF_PP',        # AMF calculated using Paul palmers code
    'SC',            # Slant Columns
    'VC_GC',         # GEOS-Chem Vertical Columns
    'VC_OMI',        # OMI VCs
    'VC_OMI_RSC',    # OMI VCs with Reference sector correction? TODO: check
    'col_uncertainty_OMI',
    'fires',         # Fire count
    'AAOD',          # Smoke AAOD_500nm interpolated from OMAERUVd
    ]
    #'fire_mask_8',   # true where fires occurred over last 8 days
    #'fire_mask_16' ] # true where fires occurred over last 16 days

# attributes for omhchorp
__OMHCHORP_ATTRS__ = {
    'gridentries':          {'desc':'satellite pixels averaged per gridbox'},
    'ppentries':            {'desc':'PP_AMF values averaged per gridbox'},
    'VC_OMI':               {'units':'molec/cm2',
                             'desc':'regridded OMI swathe VC'},
    'VC_GC':                {'units':'molec/cm2',
                             'desc':'regridded VC, using OMI SC recalculated using GEOSChem shape factor'},
    'SC':                   {'units':'molec/cm2',
                             'desc':'OMI slant colums'},
    'VCC':                  {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using GEOS-Chem shape factor and reference sector correction'},
    'VCC_PP':               {'units':'molec/cm2',
                             'desc':'Corrected OMI columns using PPalmer and LSurl\'s lidort/GEOS-Chem based AMF'},
    'VC_OMI_RSC':           {'units':'molec/cm2',
                             'desc':'OMI\'s RSC corrected VC '},
    'RSC':                  {'units':'molec/cm2',
                             'desc':'GEOS-Chem/OMI based Reference Sector Correction: is applied to pixels based on latitude and track number'},
    'RSC_latitude':         {'units':'degrees',
                             'desc':'latitude centres for RSC'},
    'RSC_GC':               {'units':'molec/cm2',
                             'desc':'GEOS-Chem HCHO over reference sector'},
    'col_uncertainty_OMI':  {'units':'molec/cm2',
                             'desc':'OMI\'s column uncertainty'},
    'AMF_GC':               {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_OMI':              {'desc':'AMF based on GC recalculation of shape factor'},
    'AMF_PP':               {'desc':'AMF based on PPalmer code using OMI and GEOS-Chem'},
    #'fire_mask_16':         {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current or prior 8 day block"},
    #'fire_mask_8':          {'desc':"1 if 1 or more fires in this or the 8 adjacent gridboxes over the current 8 day block"},
    'fires':                {'desc':"daily gridded fire count from AQUA/TERRA"},
    'AAOD':                 {'desc':'daily smoke AAOD_500nm from AURA (OMAERUVd)'},
    }

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
        if keylist is None:
            keylist=__OMHCHORP_KEYS__

        keylist=list(set(keylist+__OMHCHORP_COORDS__)) # make sure coords are included

        for day in daylist:
            struct.append(fio.read_omhchorp(date=day,
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
        self.surface_areas=util.area_grid(self.lats,self.lons)
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
        if hasattr(self,'AMF_PP'):
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

    def make_fire_mask(self, d0, dN=None, days_masked=1, fire_thresh=1, adjacent=True):
        '''
            Return fire mask with dimensions [len(d0-dN), n_lats, n_lons]
            looks at fires between [d0-days_masked+1, dN], for each day in d0 to dN

            If days_masked extends before contained days then the fire mask is read from omhchorp files

            mask is true where more than fire_thresh fire pixels exist.

        '''
        # first day of filtering
        daylist=util.list_days(d0,dN)
        first_day=daylist[0]-timedelta(days=days_masked-1)
        last_day=daylist[-1]

        if __VERBOSE__:
            print ("make_fire_mask will return rolling %d day fire masks between "%days_masked, d0, '-', last_day)
            print ("They will filter gridsquares with more than %d fire pixels detected"%fire_thresh)

        # check if we already have the required days
        dates=np.array(self.dates)
        if (first_day >= dates[0]) and (last_day <= dates[-1]) and hasattr(self, 'fires'):
            if __VERBOSE__:
                print("fire mask will be made using already read data")
            i = (dates <= last_day) * (dates >= first_day)
            fires=self.fires[i,:,:]

        else:
            if __VERBOSE__:
                print("fire mask will be read from omhchorp now...")
            om=omhchorp(first_day,last_day,keylist=['fires'])
            fires=om.fires

        # mask squares with more fire pixels than allowed
        mask = fires>fire_thresh
        retmask= np.zeros([len(daylist),self.n_lats,self.n_lons]).astype(np.bool)

        # actual mask is made up of sums of daily masks over prior days_masked

        if days_masked>1:
            # from end back to first day in daylist
            for i in -np.arange(0,len(daylist)):
                tempmask=mask[i-days_masked:] # look at last days_masked
                if i < 0:
                    tempmask=tempmask[:i] # remove days past the 'current'

                # mask is made up from prior N days, working backwards
                retmask[i-1]=np.sum(tempmask,axis=0)

        else:
            retmask = mask
        assert retmask.shape[0]==len(daylist), 'return mask is wrong!'

        # mask adjacent squares also (if desired)
        if adjacent:
            for i in range(len(daylist)):
                retmask[i]=util.set_adjacent_to_true(retmask[i])

        return retmask

    def get_smoke_mask(self, d0, dN=None, thresh=0.001):
        '''
            Read OMAERUVd AAOD(500nm), regrid into local resoluion, mask days above thresh
        '''
        assert False, "To be implemented"

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
