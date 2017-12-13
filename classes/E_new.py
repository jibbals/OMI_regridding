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

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

# my file reading library
from utilities import fio, JesseRegression
import utilities.utilities as util
from utilities import plotting as pp
sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=True
__DEBUG__=False
__E_new_keys__=['E_isop','E_isop_kg','GC_E_isop','GC_E_isop_kg',
                'GC_background', 'GC_slope', 'background', 'lats',
                'lats_e', 'lons', 'lons_e', 'time','smearing']

# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Change to -175 to avoid crossing the 179 -> -179 boundary?
__REMOTEPACIFIC__=[-15, -180, 15, -120]

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
            data,attrs=fio.read_E_new_month(month=month)
            E_new_list.append(data)
        self.attributes=attrs
        dimensions=['lons','lons_e','lats','lats_e']

        # Combine the data
        for key in E_new_list[0].keys():
            if __VERBOSE__:
                print("Reading ",key) 
            #print(key,np.shape(E_new_list[0][key]))
            if key in dimensions:
                setattr(self,key,E_new_list[0][key])
            elif (key == 'time') or (key in dkeys):
                data=np.array(E_new_list[0][key])
                for i in range(1,n_months):
                    data=np.append(data,np.array(E_new_list[i][key]),axis=0)
                setattr(self, key, data)
            elif __VERBOSE__:
                print("KEY %s not being read from E_new dataset"%key )
        self.dates=[datetime.strptime(str(t),"%Y%m%d") for t in self.time]

        # True over ocean squares
        mlons,mlats=np.meshgrid(self.lons,self.lats)
        self.oceanmask=maskoceans(mlons,mlats,mlons,inlands=False).mask

    def get_season(self,key,region,maskocean=True):
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

    def get_series(self, key, region, maskocean=True, testplot=False):
        '''
            Average over region[SWNE]
            Return dates, data[key]
        '''
        data=np.copy(getattr(self,key)) # copy so we don't overwrite anything
        lats=self.lats
        lons=self.lons

        # Mask oceans with NANs
        if maskocean:
            for i in range(np.shape(data)[0]):
                data[i,self.oceanmask]= np.NaN
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

