#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:41:53 2017

    Class for E_new files

@author: jesse
"""

### LIBRARIES/MODULES ###

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from mpl_toolkits.basemap import maskoceans #Basemap, maskoceans
#from matplotlib.colors import LogNorm # lognormal colour bar

import numpy as np
from datetime import datetime#, timedelta
from scipy.constants import N_A as N_avegadro

# Local libraries
# my file reading library
from utilities import fio, JesseRegression
import utilities.utilities as util
from utilities import plotting as pp


###############
### GLOBALS ###
###############

__VERBOSE__=False
__DEBUG__=False
__E_new_keys__=[            # #  # {time, lats, lons}
                'BG_OMI',       #  {31, 152, 152}
                'BG_PP',        #  {31, 152, 152}
                'BG_VCC',       #  {31, 152, 152}
                'E_MEGAN',      #  {31, 18, 19}
                'E_GC',     #  {31, 152, 152}
                'E_GC_u',   #  With unfiltered by fire/smoke/anthro
                'E_GC_lr',  #  at low resolution: {31,18,19}
                'E_OMI',    #  {31, 152, 152}
                'E_OMI_u',  #  E_VCC_OMI without any filters applied
                'E_OMI_lr', #  E_VCC_OMI at low resolution
                'E_PP',     #  {31, 152, 152}
                'E_PP_u',   #  without using filters
                'E_PP_lr',  #  at low resolution
#                'E_VCC_GC',     #  {31, 152, 152}
#                'E_VCC_GC_u',   #  With unfiltered by fire/smoke/anthro
#                'E_VCC_GC_lr',  #  at low resolution: {31,18,19}
#                'E_VCC_OMI',    #  {31, 152, 152}
#                'E_VCC_OMI_u',  #  E_VCC_OMI without any filters applied
#                'E_VCC_OMI_lr', #  E_VCC_OMI at low resolution
#                'E_VCC_PP',     #  {31, 152, 152}
#                'E_VCC_PP_u',   #  without using filters
#                'E_VCC_PP_lr',  #  at low resolution
                'firemask',   # Fire filter used for e estimates [time,lat,lon]
                'anthromask', # anthro filter
                'smearmask',  # smearing filter
                'VCC_GC',       # {31, 152, 152}
                'VCC_OMI',      # {31, 152, 152}
                'VCC_PP',       # {31, 152, 152}
                'VCC_GC_u',       # Unmasked versions {31, 152, 152}
                'VCC_OMI_u',      # {31, 152, 152}
                'VCC_PP_u',       # {31, 152, 152}
                #'smearing',     # model resolution monthly smearing
                'ModelSlope',   # model resolution monthly slope (after filtering smearing)
                # Uncertainty stuff:
                'E_PP_err_lr', # E_new error  
                'E_PPm_err_lr', # monthly E_new error
                'SC_err', # fitting error
                'SC_err_lr', 
                'VCC_err', # Omega_pp error, relative error, and low res too
                'VCC_rerr',
                'VCC_err_lr',
                'VCC_rerr_lr',
                'slope_rerr_lr', # relative slope error per month
                #'ModelSlopeUncertainty', # monthly low res uncertainty in slope
                #'ModelSlope_sf',
                'pixels',       # OMI pixel count
                'pixels_u',     # before filtering pixel count
                'pixels_lr',    # OMI pixel count at low resolution
                'pixels_PP',    # OMI pixel count using PP code
                'pixels_PP_u',  # before filtering
                'pixels_PP_lr', # OMI pixel count using PP code at low resolution
                'time',         # time is a dimension but also data: datetime string stored in file
                'dates',        # datetime objects created from time field
                ]
__E_new_keys_lr__ = [
                     'E_MEGAN',      #  {31, 18, 19}
                     'E_GC_lr',  #  at low resolution
                     'E_OMI_lr', #  E_VCC_OMI at low resolution
                     'E_PP_lr',  #  at low resolution
                     'E_PP_err_lr', # E_new error  
                     'E_PPm_err_lr',
#                     'VC_relative_uncertainty_lr', # relative uncertainty per grid square for OMI VC
#                     'E_VCC_GC_lr',  #  at low resolution
#                     'E_VCC_OMI_lr', #  E_VCC_OMI at low resolution
#                     'E_VCC_PP_lr',  #  at low resolution
                     'smearfilter',  # smearing filter
                     'smearing',     # model resolution monthly smearing
                     'ModelSlope',   # model resolution monthly slope
                     'pixels_lr',    # OMI pixel count at low resolution
                     'pixels_PP_lr', # OMI pixel count using PP code at low resolution
                     'SC_err_lr', 
                     'VCC_err_lr',
                     'VCC_rerr_lr',
                     'slope_rerr_lr', # relative slope error per month
                    ]
__E_new_dims__=['lats',         # 0.25x0.3125 resolution
                'lats_e',       # edges
                'lons',         #
                'lons_e',       #
                'lats_lr',      # 2x2.5 resolution
                'lons_lr',      # 2x2.5 resolution
                'SA',           # Surface area [lats,lons] km2
                'SA_lr',        # low res
                ]
__E_new_topd__=['E_GC',     #  {31, 152, 152}
                'E_GC_u',   #  With unfiltered by fire/smoke/anthro
                'E_GC_lr',  #  at low resolution: {31,18,19}
                'E_OMI',    #  {31, 152, 152}
                'E_OMI_u',  #  E_VCC_OMI without any filters applied
                'E_OMI_lr', #  E_VCC_OMI at low resolution
                'E_PP',     #  {31, 152, 152}
                'E_PP_u',   #  without using filters
                'E_PP_lr',  #  at low resolution
                ]


# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Change to -175 to avoid crossing the 179 -> -179 boundary?
__REMOTEPACIFIC__=util.__REMOTEPACIFIC__

########################################################################
########################  E_new CLASS ###############################
########################################################################

class E_new:
    # Keys: time,lats,lons,E_isop,...
    def __init__(self, day0, dayn=None, dkeys=__E_new_keys__):
        '''
            Read E_new from day0 to dayn
            if dkeys is set then only those datakeys will be saved
            data should be ----- ([date,]lat,lon[,lev])
        '''

        # day0=datetime(2005,1,1); dayn=datetime(2005,3,1)
        # Get list of months including requested data
        months=util.list_months(day0,dayn)
        n_months=len(months)
        E_new_list=[]

        # For each month read the data
        for month in months:
            if __VERBOSE__:
                print('reading month ',month)
            data,attrs=fio.read_E_new_month(month=month)
            if __VERBOSE__:
                print(data.keys())
            # remove unwanted data if possible
            if dkeys is not None:
                for key in __E_new_keys__:
                    if (key not in dkeys) and (key not in __E_new_dims__+['dates','time']):
                        rem=data.pop(key)

            E_new_list.append(data)
            if __VERBOSE__:
                print('keys after pruning')
                print(data.keys())
        self.attributes=attrs

        # Combine the data
        for key in E_new_list[0].keys():
            if __VERBOSE__:
                print("Found ",key, np.shape(E_new_list[0][key]))

            # Read the dimensions
            if key in __E_new_dims__:
                setattr(self,key,E_new_list[0][key])
                if __VERBOSE__:
                    print("Reading %s"%key )
            # Read the data and append to time dimensions if there's more than
            # one month file being read
            elif (key in ['ModelSlope']) and key in dkeys:

                # np array of the data [lats, lons]
                data0=np.array(E_new_list[0][key])

                data=np.zeros([n_months, data0.shape[0], data0.shape[1]])
                data[0] = data0

                # for each extra month, append onto time dim:
                for i in range(1,n_months):
                    data[i]=np.array(E_new_list[i][key])

                setattr(self, key, data)
                # convert filters to boolean
                if 'filter' in key:
                    setattr(self,key,data.astype(np.bool))
                if __VERBOSE__:
                    print("Reading %s"%key )

            elif (key in ['time','dates']) or (key in dkeys):

                # np array of the data [time, lats, lons]
                data=np.array(E_new_list[0][key])

                # for each extra month, append onto time dim:
                for i in range(1,n_months):
                    data=np.append(data,np.array(E_new_list[i][key]),axis=0)

                setattr(self, key, data)
                # convert filters to boolean
                if 'filter' in key:
                    setattr(self,key,data.astype(np.bool))
                if __VERBOSE__:
                    print("Reading %s"%key )
            #elif __VERBOSE__:
                #print("KEY %s not being read from E_new dataset"%key )


        self.dates=[datetime.strptime(str(t),"%Y%m%d") for t in self.time]

        # True over ocean squares
        mlons,mlats=np.meshgrid(self.lons,self.lats)
        mlons_lr,mlats_lr=np.meshgrid(self.lons_lr,self.lats_lr)
        self.oceanmask=maskoceans(mlons,mlats,mlons,inlands=False).mask
        self.oceanmask_lr=maskoceans(mlons_lr,mlats_lr,mlons_lr,inlands=False).mask
        self.oceanmask3d=np.repeat(self.oceanmask[np.newaxis,:,:],len(self.dates),axis=0)

        if not hasattr(self,'SA'):
            self.SA = util.area_grid(self.lats,self.lons)
            self.SA_lr = util.area_grid(self.lats_lr,self.lons_lr)
        ## conversions to kg/s
        # [atom C / cm2 / s ] * 1/5 * cm2/km2 * km2 * kg/atom_isop
        # = isoprene kg/s
        # kg/atom_isop = grams/mole * mole/molec * kg/gram
        kg_per_atom = util.__grams_per_mole__['isop'] * 1.0/N_avegadro * 1e-3
        conversion= 1./5.0 * 1e10 * self.SA * kg_per_atom
        conversion_lr = 1./5.0 * 1e10 * self.SA_lr * kg_per_atom
        self.conversion_to_kg    = np.repeat(conversion[np.newaxis,:,:],len(self.dates),axis=0)
        self.conversion_to_kg_lr = np.repeat(conversion_lr[np.newaxis,:,:],len(self.dates),axis=0)

        # HANDLE NEGATIVE EMISSIONS ESTIMATIONS
        for key in __E_new_topd__:
            if hasattr(self,key):
                topd = getattr(self,key)
                avg=np.nanmean(topd)
                topd[topd<0] = 0
                if __VERBOSE__:
                    print("Removing negatives from ",key)
                    print('%.2e -> %.2e'%(avg, np.nanmean(topd)))
                setattr(self,key,topd)

    def get_monthly_multiyear(self, key, region, maskocean=True):
        '''
            multiyear monthly average of key over region
        '''
        assert False, 'Not implemented yet'
        # Get months in class
        months=util.list_months(self.dates[0],self.dates[-1])

        # initialise array
        mavg=np.zeros(len(months))+np.NaN

        # grab time series over region
        dates,arr=self.get_series(key=key,region=region,maskocean=maskocean,testplot=False)

        # average of each month
        for i,m in enumerate(months):
            di=self.date_indices(util.list_days(m,month=True))
            # Average of the month:
            mavg[i]=np.nanmean(arr[di])
            # deseasonalised data:
            arr[di]=arr[di]-mavg[i]
        return mavg, arr

    def get_monthly(self,key,region,maskocean=True):
        '''
            Monthly average of key over region
        '''
        # Get months in class
        months=util.list_months(self.dates[0],self.dates[-1])

        # initialise array
        mavg=np.zeros(len(months))+np.NaN

        # grab time series over region
        dates,arr=self.get_series(key=key,region=region,maskocean=maskocean,testplot=False)

        # average of each month
        for i,m in enumerate(months):
            di=self.date_indices(util.list_days(m,month=True))
            # Average of the month:
            mavg[i]=np.nanmean(arr[di])
            # deseasonalised data:
            arr[di]=arr[di]-mavg[i]
        return mavg, arr

    def get_series(self, key, region,lowres=False, maskocean=True,
                   lat=None,lon=None,
                   testplot=False,):
        '''
            Grab time series of given key (time,lats,lons)
            optionally can pull out series at single lat,lon
            Return dates, data[key]
        '''
        data=np.copy(getattr(self,key)) # copy so we don't overwrite anything
        lats=[self.lats, self.lats_lr][lowres]
        lons=[self.lons, self.lons_lr][lowres]
        n_times=np.shape(data)[0] # time dim

        # Mask oceans with NANs
        if maskocean:
            oceanmask=[self.oceanmask,self.oceanmask_lr][lowres]
            oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],n_times,axis=0)
            data[oceanmask3d]= np.NaN
        #TEST
        if testplot:
            pp.createmap(data[0], lats, lons, region=region,
                         linear=True,pname="test_mask.png")

        # Subset to region:
        lati,loni=util.lat_lon_range(lats,lons,region)
        data=data[:,lati,:]
        data=data[:,:,loni]
        dates=self.dates


        return dates,np.nanmean(data,axis=(1,2))

    def plot_series(self, key='VCC_OMI',
                    lat=pp.__cities__['Syd'][0], lon=pp.__cities__['Syd'][1],
                    *ptsargs, **pltargs):
        '''
            plot a particular key over time in a single lat,lon
            *ptsargs are taken to pp.plot_time_series,
            **pltargs are taken to plt.plot
        '''

        # index of lat,lon
        lats,lons=self.lats,self.lons
        if key in __E_new_keys_lr__:
            lats,lons=self.lats_lr,self.lons_lr

        lati,loni=util.lat_lon_index(lat,lon,lats,lons)

        data=getattr(self,key)[:,lati,loni] # pull out slice over time

        print(data)
        pp.plot_time_series(self.dates,data,*ptsargs,**pltargs)


    def date_indices(self,datelist):
        ''' return indexes of listed dates '''
        ret=[]
        for d in datelist:
            ret.append(np.where(np.array(self.dates)==d)[0][0])
        return ret

    def plot_map(self, day, dayn=None, key='E_isop', region=pp.__AUSREGION__):
        '''
            plot map of key over region for day (or averaged from day to dayn)
        '''
        di=self.date_indices(util.list_days(day,dayn))
        data=getattr(self,key)
        if len(di)==1:
            data=data[di]
        else:
            data=np.nanmean(data[di,:,:],axis=0)

        units=self.attributes[key]['units']
        pp.createmap(data, self.lats_e, self.lons_e, edges=True,
                     latlon=True, region=region, linear=True,
                     clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
              pname=None,title=None,suptitle=None, smoothed=False,
              cmapname=None, fillcontinents=None)


    def plot_regression(self, day0,dayn, region=pp.__AUSREGION__, deseasonalise=False, **ppargs):
        '''
            plot regression of E_isop and GC_E_isop from day0 to dayn
            Limit to region
            optionally deseasonalise using monthly averages
        '''
        # Get time series:
        dates, E_isop    = self.get_series('E_isop',region=region)
        dates, GC_E_isop = self.get_series('GC_E_isop', region=region)

        if deseasonalise:
            season,E_isop    = self.get_season('E_isop',region=region,maskocean=True)
            season,GC_E_isop = self.get_season('GC_E_isop',region=region,maskocean=True)

        # Regression over desired time
        di = self.date_indices(util.list_days(day0,dayn))
        x = GC_E_isop[di]; y=E_isop[di]

        # Add limits if not there already
        if not 'lims' in ppargs:
            ppargs['lims']=[-0.4e12,3.5e12]
            if deseasonalise:
                ppargs['lims']=[-2e12,2e12]

        # Plot the regression

        pp.plot_regression(x,y,logscale=False,**ppargs)
        m,b,r,p,sterr = JesseRegression.RMA(x,y) # y = mx + b

        return [x,y]

