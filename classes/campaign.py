#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:40:03 2017

@author: jesse
"""

###############
### MODULES ###
###############
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

from utilities import fio
from utilities import plotting as pp
sys.path.pop(0)

###############
### GLOBALS ###
###############
__VERBOSE__=True

###############
### CLASS   ###
###############

class campaign:
    '''
    Class for holding the campaign datasets

    '''
    def __init__(self):
        self.dates=[]
        # site lat and lon
        self.lat=0.0
        self.lon=0.0
        self.hcho=np.NaN # numpy array of measured hcho []
        self.isop=np.NaN # '' isoprene []
        self.hcho_units='ppb'
        self.isop_units='ppb'
        self.hcho_detection_limit=np.NaN
        self.isop_detection_limit=np.NaN

    def read_SPS1(self):
        '''
            Read the sps 1 dataset
            # Measurements are reported in local time (UTC+10)
            # units are unlisted in the doi
        '''

        # TODO: remove ../ when not testing any more
        self.fpath='../Data/campaigns/SPS1/SPS1_PTRMS.csv'
        data=fio.read_csv(self.fpath)
        # PTRMS names the columns with m/z ratio, we use
        #   HCHO = 31, ISOP = 69
        h_key='m/z 31'
        i_key='m/z 69'

        # First row is detection limits

        self.hcho_detection_limit=float(data[h_key][0])
        self.isop_detection_limit=float(data[i_key][0])
        # second row is empty
        # data begins at third row, last row is empty
        hcho=np.array(data[h_key][2:-1])
        isop=np.array(data[i_key][2:-1])

        # convert from strings to floats
        hcho[hcho=='<mdl'] = 'NaN'
        self.hcho = hcho.astype(np.float)
        isop[isop=='<mdl'] = 'NaN'
        self.isop = isop.astype(np.float)

        # Convert strings to dates:
        for date in data['Timestamp'][2:-1]:
            #if date == '':
            #    self.dates.append(np.NaN)
            #else:
            # Timestamp like this: 18/02/2011 17:00
            self.dates.append(datetime.strptime(date,'%d/%m/%Y %H:%M'))

        if __VERBOSE__:
            print("read %d entries from %s to %s"%(len(self.dates),self.dates[0],self.dates[-1]))
        #self.dates=[datetime.strptime('%d/%m/%Y %H',d) for d in dates]

        #print(data)
    def plot_series(self,pname):

        dates=self.dates

        for key,c in zip(['hcho','isop'],['orange','green']):
            data=getattr(self,key)
            dlim=getattr(self,key+'_detection_limit')
            #plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, xtickrot=30, dfmt='%Y%m', **pltargs)
            pp.plot_time_series(dates, data, ylabel='[ppb]', dfmt='%d %b', label=key,color=c)
            plt.plot([dates[0],dates[-1]], [dlim,dlim], color=c, linestyle='--')

        plt.savefig(pname)
        print("Saved %s"%pname)


if __name__=='__main__':
    sps1=campaign()
    sps1.read_SPS1()
    sps1.plot_series('sps1.png')