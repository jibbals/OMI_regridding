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
from utilities import fio, GMAO
import utilities.utilities as util
from utilities import plotting as pp

###############
### GLOBALS ###
###############

__AMF_REL_ERR__=0.3 # 30% relative AMF error assumed

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
    def __init__(self, day0, dayn=None, latres=0.25, lonres=0.3125, keylist=None):
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
            
        self.dates = util.list_days(day0,dayn)
        self.n_times = len(self.dates)
        self.n_lats, self.n_lons = len(self.lats), len(self.lons)
        self.oceanmask=util.oceanmask(self.lats,self.lons)
        
    def uncertainty(self, masks, region=pp.__AUSREGION__, ramf=__AMF_REL_ERR__):
        '''
            set and return delta omega / omega at low resolution, daily and monthly?
               dO/O = sqrt (  (dsc^2 + drsc^2) / (sc-rsc)^2 + (damf/amf)^2   )
            RSC = mean at SC latitudes over all tracks
            dRSC = std monthly for Aus lats (-45 to -10) and all tracks
            
            Assume we have one month of data for monthly outputs
            ONLY looking at uncertainty in PP columns
        '''
        # low and high resolution
        lats_lr, lons_lr, lates, lones = util.lat_lon_grid(GMAO.__LATRES_GC__, GMAO.__LONRES_GC__)
        lats,lons = self.lats, self.lons
        ndays=len(self.dates)
        
        dSC_u       = np.copy(self.col_uncertainty_OMI) # high res mean 
        dSC         = np.copy(self.col_uncertainty_OMI) # high res mean 
        dSC[masks]  = np.NaN
        SC_u        = np.copy(self.SC)  # high res mean
        SC          = np.copy(self.SC)
        SC[masks]   = np.NaN
        pix_u       = np.copy(self.ppentries)
        pix         = np.copy(self.ppentries) # high res total
        pix[masks]  = 0
        
        RSC         = self.RSC[:,:,:,2] # [days, 500 lats, 60 tracks, 3 types]# type=omi,gc,pp
        lats_RSC    = self.RSC_latitude
        
        # error based on std over Aus latitudes
        auslats     = (lats_RSC < -10) * (lats_RSC > -45)
        dRSC        = np.nanstd(RSC[:,auslats,:])
        
        # we also want to bin the RSC onto the same grid as SC
        RSC         = np.nanmean(RSC,axis=2) # average over the tracks
        RSC         = np.repeat(RSC[:,:,np.newaxis],len(lons),axis=2)
        RSCnew      = np.copy(SC)
        for i in range(ndays):
            RSCnew[i]  = util.regrid_to_higher(RSC[i],lats_RSC,lons,lats,lons )
        RSC         = RSCnew
        
        
        # Need to convert to low res:
        lrshape=[ndays, len(lats_lr),len(lons_lr)]
        dSC_lr, dSC_u_lr    = np.zeros(lrshape),np.zeros(lrshape) 
        SC_lr, SC_u_lr      = np.zeros(lrshape),np.zeros(lrshape)
        pix_lr, pix_u_lr    = np.zeros(lrshape),np.zeros(lrshape)
        RSC_lr              = np.zeros(lrshape)
        for i in range(ndays):
            dSC_lr[i]       = util.regrid_to_lower(dSC[i], lats,lons, lats_lr, lons_lr, pixels=pix[i])
            dSC_u_lr[i]     = util.regrid_to_lower(dSC_u[i], lats,lons, lats_lr, lons_lr, pixels=pix_u[i])
            SC_lr[i]        = util.regrid_to_lower(SC[i], lats,lons, lats_lr, lons_lr,pixels=pix[i])
            SC_u_lr[i]      = util.regrid_to_lower(SC_u[i], lats,lons, lats_lr, lons_lr,pixels=pix_u[i])
            RSC_lr[i]       = util.regrid_to_lower(RSC[i], lats, lons, lats_lr, lons_lr)
            
            # store pixel count in lower resolution also, using sum of pixels in each bin
            pix_lr[i]    = util.regrid_to_lower(pix[i],lats,lons,lats_lr,lons_lr,func=np.nansum)
            pix_u_lr[i] = util.regrid_to_lower(pix_u[i],lats,lons,lats_lr,lons_lr,func=np.nansum)
        
        
        # Finally calculate the relative uncertainty of the OMI vertical columns
        # VCC = (SC - RSC)/AMF: ERR(SC-RSC) = sqrt(err2(SC) + err2(RSC))
        
        # monthly pixel counts
        pixm=np.nansum(pix,axis=0)
        pixm_lr=np.nansum(pix_lr,axis=0)
        pixm_u=np.nansum(pix_u,axis=0)
        pixm_u_lr=np.nansum(pix_u_lr,axis=0)
        
        # Relative error per day
        # ignore divide by zero and nan values
        with np.errstate(divide='ignore'):
            ## relative Omega error: dO/O = sqrt (  (dsc^2 + drsc^2) / (sc-rsc)^2 + (damf/amf)^2   )
            
            # per pixel
            rO = np.sqrt( (dSC**2 + dRSC**2) / (SC-RSC)**2 + (ramf)**2 )                    
            rO_u = np.sqrt( (dSC_u**2 + dRSC**2) / (SC_u-RSC)**2 + (ramf)**2 )              
            rO_lr = np.sqrt( (dSC_lr**2 + dRSC**2) / (SC_lr-RSC_lr)**2 + (ramf)**2 )        
            rO_u_lr = np.sqrt( (dSC_u_lr**2 + dRSC**2) / (SC_u_lr-RSC_lr)**2 + (ramf)**2 )  
            
            # daily
            rOd         = rO        / np.sqrt(pix)
            rOd_u       = rO_u      / np.sqrt(pix_u)
            rOd_lr      = rO_lr     / np.sqrt(pix_lr)
            rOd_u_lr    = rO_u_lr   / np.sqrt(pix_u_lr)
            
            # monthly:
            rOm = np.nanmean(rO,axis=0)             /np.sqrt(pixm)
            rOm_u = np.nanmean(rO_u,axis=0)         /np.sqrt(pixm_u)
            rOm_lr = np.nanmean(rO_lr,axis=0)       /np.sqrt(pixm_lr)
            rOm_u_lr = np.nanmean(rO_u_lr,axis=0)   /np.sqrt(pixm_u_lr)
            
            
        # Also lets get daily remote pacific background error
        # This matches process in Inversion.py
        # get mean error over remote pacific, and total pixels used
        bg, _,_      = util.remote_pacific_background(rO, lats, lons,
                              average_lons=True, has_time_dim=True)
        bgpix, _,_      = util.remote_pacific_background(pix, lats, lons,
                              average_lons=True, has_time_dim=True, func = np.nansum)
        bg_lr, _,_   = util.remote_pacific_background(rO_lr, lats_lr, lons_lr,
                              average_lons=True, has_time_dim=True)
        bgpix_lr, _,_   = util.remote_pacific_background(pix_lr, lats_lr, lons_lr,
                              average_lons=True, has_time_dim=True, func = np.nansum)
        bg_u, _,_    = util.remote_pacific_background(rO_u, lats, lons,
                              average_lons=True, has_time_dim=True)
        bgpix_u, _,_    = util.remote_pacific_background(pix_u, lats, lons,
                              average_lons=True, has_time_dim=True, func = np.nansum)
        bg_u_lr, _,_ = util.remote_pacific_background(rO_u_lr, lats_lr, lons_lr,
                              average_lons=True, has_time_dim=True)
        bgpix_u_lr, _,_ = util.remote_pacific_background(pix_u_lr, lats_lr, lons_lr,
                              average_lons=True, has_time_dim=True, func = np.nansum)
        
        # for background error using daily reduced error
        
        # daily error reduced by sqrt(n), monthly too
        
        bgm_lr   = np.nanmean(bg_lr,axis=0)/np.nansum(bgpix_lr,axis=0)
        bg       = bg/np.sqrt(bgpix)
        bg_u     = bg_u/np.sqrt(bgpix_u)
        bg_lr    = bg_lr/np.sqrt(bgpix_lr)
        bg_u_lr  = bg_u_lr/np.sqrt(bgpix_u_lr)
        # finally repeat them over the longitudinal dimension
        bg       = np.repeat(bg[:,:,np.newaxis],len(lons), axis=2)
        bg_u     = np.repeat(bg_u[:,:,np.newaxis],len(lons), axis=2)
        bg_lr    = np.repeat(bg_lr[:,:,np.newaxis],len(lons_lr), axis=2)
        bg_u_lr  = np.repeat(bg_u_lr[:,:,np.newaxis],len(lons_lr), axis=2)
        # monthly also
        bgm_lr   = np.repeat(bgm_lr[:,np.newaxis],len(lons_lr), axis=1)
        
        # remove infinites?
        #rO[~np.isfinite(rO)] = np.NaN
        
        # Subset to region
        errs        = [rOd, rOd_u, pix, pix_u, bg, bg_u, RSC]
        errsm       = [rOm, rOm_u, pixm, pixm_u ]
        errs_lr     = [rOd_lr, rOd_u_lr, pix_lr, pix_u_lr, bg_lr, bg_u_lr, RSC_lr]
        errsm_lr    = [rOm_lr, rOm_u_lr, pixm_lr, pixm_u_lr, bgm_lr]
        subs=util.lat_lon_subset(lats,lons,region,data=errs, has_time_dim=True)
        subs_lr=util.lat_lon_subset(lats_lr, lons_lr, region, data=errs_lr, has_time_dim=True)
        subsm=util.lat_lon_subset(lats,lons,region,data=errsm, has_time_dim=False)
        subsm_lr=util.lat_lon_subset(lats_lr, lons_lr, region, data=errsm_lr, has_time_dim=False)
        lats=subs['lats']
        lons=subs['lons']
        lats_lr=subs_lr['lats']
        lons_lr=subs_lr['lons']
        
        [rOd, rOd_u, pix, pix_u, bg, bg_u, RSC]     = subs['data']
        [rOm, rOm_u, pixm, pixm_u,]                 = subsm['data']
        [rOd_lr, rOd_u_lr, pix_lr, pix_u_lr, 
         bg_lr, bg_u_lr, RSC_lr]                    = subs_lr['data']
        [rOm_lr, rOm_u_lr, pixm_lr, pixm_u_lr, 
         bgm_lr, ]                                  = subsm_lr['data']
                  # these are the mean per pixel relative errors
        return  { 'rO':rOd, 'rO_u':rOd_u, 'rO_lr':rOd_lr, 'rO_u_lr':rOd_u_lr, 
                 # number of pixels per grid square per day
                 'pix':pix,'pix_u':pix_u, 'pix_lr':pix_lr, 'pix_u_lr':pix_u_lr,
                 # same but per month
                 'pixm':pixm,'pixm_u':pixm_u, 'pixm_lr':pixm_lr, 'pixm_u_lr':pixm_u_lr,
                 # monthly uncertainty
                 'rOm':rOm, 'rOm_u':rOm_u, 'rOm_lr':rOm_lr, 'rOm_u_lr':rOm_u_lr, 
                 # Reference sector correction errors (RSC is repeated along longitudes)
                 'RSC':RSC, 'RSC_lr':RSC_lr, 'dRSC':dRSC,
                 # background errors per day from remote pacific
                 'bg':bg, 'bg_u':bg_u, 'bg_lr':bg_lr, 'bg_u_lr':bg_u_lr, 
                 # background errors per monnth
                 'bgm_lr':bgm_lr,
                 # Fitting Error:
                 'dSC':dSC, 'dSC_lr':dSC_lr,
                 # extras
                 'lats':lats,'lons':lons, 'lats_lr':lats_lr,'lons_lr':lons_lr,
                 }

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

    def lower_resolution(self, key='VCC_PP', factor=8, dates=None):
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

    def time_averaged(self, day0, dayn=None, keys=['VCC_PP'],
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

    def plot_map(self,key='VCC_PP',day0=None,dayn=None,region=pp.__AUSREGION__,**cmargs):
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
    print("One day data shape: %s"%str(om.VCC_PP.shape))
    om.plot_map(**{'pname':'map_test_1.png'})
    om=omhchorp(datetime(2005,1,1), dayn=datetime(2005,1,4))
    print("4 day data shape: %s"%str(om.VCC_PP.shape))
    om.plot_map(**{'pname':'map_test_2.png'})

    landinds=om.inds_subset(maskocean=True)
    ausinds=om.inds_aus(maskocean=True)

    print("%d land elements"%np.sum(landinds))
    print("%d australia land elements"%np.sum(ausinds))

    #om=omhchorp(datetime(2005,1,1),datetime(2005,1,31))
    #om.plot_map(**{'pname':'map_test_3.png'})
