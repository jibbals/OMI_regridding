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

from utilities import fio
from utilities import plotting as pp

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
        #self.attrs={'hcho':{'units':'ppb','DL':np.NaN},
        #            'isop':{'units':'ppb','DL':np.NaN},}
        self.hcho_units='ppb'
        self.isop_units='ppb'
        self.hcho_detection_limit=np.NaN
        self.isop_detection_limit=np.NaN

    def read_SPS(self, number=1):
        '''
            Read the sps1 or sps2 dataset
            # Measurements are reported in local time (UTC+10)
            # units are unlisted in the doi
        '''

        # data
        self.fpath='Data/campaigns/SPS%d/SPS%d_PTRMS.csv'%(number,number)
        data=fio.read_csv(self.fpath)
        # PTRMS names the columns with m/z ratio, we use
        #   HCHO = 31, ISOP = 69
        h_key='m/z 31'
        i_key='m/z 69'

        self.lat,self.lon=-33.8688, 151.2093 # sydney lat/lon

        # First row is detection limits
        self.hcho_detection_limit=float(data[h_key][0])
        self.isop_detection_limit=float(data[i_key][0])


        # second row is empty in sps1
        # last row is empty
        start=[1,2][number==1]
        hcho=np.array(data[h_key][start:-1])
        isop=np.array(data[i_key][start:-1])

        # convert from strings to floats
        hcho[hcho=='<mdl'] = 'NaN'
        self.hcho = hcho.astype(np.float)
        isop[isop=='<mdl'] = 'NaN'
        self.isop = isop.astype(np.float)

        self.dates=[]
        # Convert strings to dates:
        for date in data['Timestamp'][start:-1]:
            #if date == '':
            #    self.dates.append(np.NaN)
            #else:
            # Timestamp like this: 18/02/2011 17:00
            self.dates.append(datetime.strptime(date,'%d/%m/%Y %H:%M'))

        if __VERBOSE__:
            print("read %d entries from %s to %s"%(len(self.dates),self.dates[0],self.dates[-1]))
        #self.dates=[datetime.strptime('%d/%m/%Y %H',d) for d in dates]

        #print(data)
    def plot_series(self,title=None,pname=None):

        dates=self.dates

        for key,c in zip(['hcho','isop'],['orange','green']):
            data=getattr(self,key)
            dlim=getattr(self,key+'_detection_limit')
            #plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, xtickrot=30, dfmt='%Y%m', **pltargs)
            pp.plot_time_series(dates, data, dfmt='%d %b', label=key,color=c)
            plt.plot([dates[0],dates[-1]], [dlim,dlim], color=c, linestyle='--')

        plt.ylabel('[%s]'%self.hcho_units)
        plt.title(title)
        # 3 xticks:
        plt.gca().xaxis.set_ticks([dates[0],dates[len(dates)//2],dates[-1]])

        if pname is not None:
            plt.legend()
            plt.savefig(pname)
            print("Saved %s"%pname)
            plt.close()

if __name__=='__main__':
    sps=campaign()
    sps.read_SPS(1)
    sps.plot_series('SPS1: 2011','sps1.png')
    sps.read_SPS(2)
    sps.plot_series('SPS2: 2012','sps2.png')