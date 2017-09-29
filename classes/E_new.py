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
#from mpl_toolkits.basemap import maskoceans #Basemap, maskoceans
#from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
from datetime import datetime#, timedelta

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

__VERBOSE__=False
__DEBUG__=False

# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# Change to -175 to avoid crossing the 179 -> -179 boundary?
__REMOTEPACIFIC__=[-15, -180, 15, -120]

########################################################################
########################  OMHCHORP CLASS ###############################
########################################################################

class E_new:
    def __init__(self, day0, dayn=None):
        '''
            Read E_new from day0 to dayn
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
            #print(key,np.shape(E_new_list[0][key]))
            if key in dimensions:
                setattr(self,key,E_new_list[0][key])
            else:
                data=np.array(E_new_list[0][key])
                for i in range(1,n_months):
                    data=np.append(data,np.array(E_new_list[i][key]),axis=0)
                setattr(self, key, data)

    def get_series(self, key, region):
        '''
            Average over region[SWNE]
            Return dates, data[key]
        '''
        data=getattr(self,key)
        lats=self.lats
        lons=self.lons
        lati,loni=util.lat_lon_range(lats,lons,region)
        data=data[:,lati,:]
        data=data[:,:,loni]
        dates=self.dates

        return dates,np.nanmean(data,axis=(1,2))