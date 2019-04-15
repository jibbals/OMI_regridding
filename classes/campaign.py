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
import pandas as pd

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
        self.dates      = []
        # site lat and lon
        self.lat        = 0.0
        self.lon        = 0.0
        self.hcho       = np.NaN  # numpy array of measured hcho []
        self.isop       = np.NaN  # '' isoprene []
        self.ozone      = np.NaN #
        self.attrs      = { 'hcho':{'units':'ppb','DL':np.NaN},
                            'isop':{'units':'ppb','DL':np.NaN},
                            'ozone':{'units':'ppb','DL':np.NaN},
                           }
        self.height    = 0# measurement height from ground level (m)
        self.elevation = 0# ground level above sea level (m)


    def get_daily_hour(self, hour=13,key='hcho'):
        '''
            Return one value per day matching the hour (argument)
            returns dates, data
        '''

        inds = np.array([d.hour == hour for d in self.dates])
        dates=np.array(self.dates)[inds]
        data=getattr(self,key)[inds]
        return dates,data

    def plot_series(self,title=None,pname=None,dfmt='%Y %b %d'): #'%d %b'):

        dates=self.dates

        for key,c in zip(['hcho','isop'],['orange','green']):
            data=getattr(self,key)
            dlim=self.attrs[key]['DL']
            #plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, xtickrot=30, dfmt='%Y%m', **pltargs)
            # Plot time series
            pp.plot_time_series(dates, data, dfmt=dfmt, label=key, color=c)
            # add detection limit
            if not np.isnan(dlim):
                plt.plot([dates[0],dates[-1]], [dlim,dlim], color=c, linestyle='--')

        plt.ylabel('[%s]'%self.attrs['hcho']['units'])
        plt.title(title)
        # 3 xticks:
        plt.gca().xaxis.set_ticks([dates[0],dates[len(dates)//2],dates[-1]])

        if pname is not None:
            plt.legend()
            plt.savefig(pname)
            print("Saved %s"%pname)
            plt.close()

class mumba(campaign):
    '''
    '''
    def __init__(self):
        '''
            Reads MUMBA campaign data isoprene, ozone, and hcho
            Code taken from Jenny's READ_MUMBA method
        '''
        # set up attributes
        super(mumba,self).__init__()

        # Read files with desired data, resampled to 60 minute means
        isop_df=fio.read_mumba_var('ISOP')
        hcho_df=fio.read_mumba_var('CH2O')
        o3_df  =fio.read_mumba_var('O3')

        # LATITUDE: -34.397200 * LONGITUDE: 150.899600
        self.lat        = -34.3972
        self.lon        = 150.8996
        self.height     = 10 # Metres off ground
        self.elevation  = 30 # Metres

        # dates
        dates_isop=[ts.to_pydatetime() for ts in isop_df.index]

        self.dates=dates_isop # UTC+10

        # NAMES HAVE PPBV but documentations shows units are ppb
        self.isop                   = np.array(isop_df['C5H8 [ppbv]'])
        self.attrs['isop']['units'] = 'ppb'
        self.attrs['isop']['DL']    = 0.003
        self.attrs['isop']['DL_full']    = '20121221-20121229: .003, 20122912-20130118: .005, 20130119-20130215: .003'
        #range1=dates_isop <

        self.hcho                   = np.array(hcho_df['HCHO [ppbv]'])
        self.attrs['hcho']['units'] = 'ppb'
        self.attrs['hcho']['DL']    = 0.105
        self.attrs['hcho']['DL_full']    = '20121221-20121229: .205, 20122912-20130118: .105, 20130119-20130215: .186'

        # ignore last 16 o3 measurements as they occur without matching hcho and isop measurements.
        self.ozone                  = np.array(o3_df['O3 [ppbv] (mean of hourly O3 concentration)'])[0:-16]
        self.attrs['ozone']['units']= 'ppb'
        self.attrs['ozone']['DL']   = 0.5

        # Any measurements below detection limits are set to half the DL
        range1= np.array([ d < datetime(2012,12,30) for d in self.dates ])
        range3= np.array([ d > datetime(2013,1,18) for d in self.dates ])
        range2= np.array(~range1 * ~range3)

        # check sub detection limit occurences
        #isop_dl1 = isop[range1 * (isop<0.003)] # occurs just once
        self.isop[range1 * (self.isop<0.003)] = 0.003/2.0
        #isop_dl2 = isop[range2 * (isop<0.005)] # none
        #isop_dl3 = isop[range3 * (isop<0.003)] # none

        #hcho_dl1 = hcho[range1 * (hcho<0.205)] # 9 times
        self.hcho[range1 * (self.hcho<0.205)] = 0.205/2.0
        #hcho_dl2 = hcho[range2 * (hcho<0.105)] # none
        #hcho_dl3 = hcho[range3 * (hcho<0.186)] # 13 times
        self.hcho[range1 * (self.hcho<0.186)] = 0.186/2.0

class sps(campaign):
    '''
    '''
    def __init__(self, number=1):
        '''
            Read the sps1 or sps2 dataset
            # Measurements are reported in local time (UTC+10)
            # units are unlisted in the doi
        '''
        # set up attributes
        super(sps,self).__init__()

        # data
        self.fpath='Data/campaigns/SPS%d/SPS%d_PTRMS.csv'%(number,number)
        data=fio.read_csv(self.fpath)
        # PTRMS names the columns with m/z ratio, we use
        #   HCHO = 31, ISOP = 69
        h_key='m/z 31'
        i_key='m/z 69'

        self.lat,self.lon=-33.8688, 151.2093 # sydney lat/lon

        # First row is detection limits
        self.attrs['hcho']['DL']=float(data[h_key][0])
        self.attrs['isop']['DL']=float(data[i_key][0])


        # second row is empty in sps1
        # last row is empty
        start=[1,2][number==1]
        hcho=np.array(data[h_key][start:-1])
        isop=np.array(data[i_key][start:-1])

        # convert from strings to floats
        hcho_low = hcho=='<mdl'
        hcho[hcho_low] = 'NaN'
        self.hcho = hcho.astype(np.float)
        self.hcho[hcho_low] = self.attrs['hcho']['DL'] / 2.0 # set to half of detection limit

        isop_low = isop=='<mdl'
        isop[isop_low] = 'NaN'
        self.isop = isop.astype(np.float)
        self.isop[isop_low] = self.attrs['isop']['DL'] / 2.0 # set to half of detection limit

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

__ftir_keys__ = {'H2CO.COLUMN_ABSORPTION.SOLAR':'VC', # vertical column 1d}
                 'H2CO.COLUMN_ABSORPTION.SOLAR_APRIORI':'VC_apri',
                 'H2CO.COLUMN_ABSORPTION.SOLAR_AVK':'VC_AK', # avg kernal [dates,ALTS]
                 'H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR':'VMR', # vertical mixing ratio [dates, ALTS]
                 'H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_APRIORI':'VMR_apri',
                 'H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_AVK':'VMR_AK',
                 
                 # dims
                 'PRESSURE_INDEPENDENT':'P', # pressure mid level
                 'DATETIME':'dates', # dates in MJD2000
                 'SURFACE.PRESSURE_INDEPENDENT':'Psurf', 
                 }

class Wgong(campaign):
    '''
    '''
    def __init__(self):#, year=datetime(2007,1,1)):
        '''
            Read the h5 data
        '''
        # read first year
        datadir='Data/campaigns/Wgong/'
        data, attrs= fio.read_hdf5(datadir+'ftir_2007.h5')
        self.attrs={}
        for key in __ftir_keys__.keys():
            nicekey=__ftir_keys__[key]
            setattr(self, nicekey, data[key])
            self.attrs[nicekey] = attrs[key]
        
        for year in np.arange(2008,2014):
            data, attrs= fio.read_hdf5(datadir+'ftir_%d.h5'%year)
            # extend along time dim for things we want to keep
            for key in __ftir_keys__.keys():
                nicekey=__ftir_keys__[key]
                array=getattr(self,nicekey)
                array = np.append(array, data[key], axis=0)
                setattr(self, nicekey, array)
        
        # convert modified julian days to local datetimes
        UTC = [datetime(2000,1,1)+timedelta(days=d) for d in self.dates]
        dates = [ d + timedelta(hours=10) for d in UTC ] # UTC to local time for wollongong
        self.dates=dates
        self.lat = data['LATITUDE.INSTRUMENT']
        self.lon = data['LONGITUDE.INSTRUMENT']
        self.alt = 310 # 310 metres
        # dimensions
        self.alts = data['ALTITUDE']
        self.alts_e = data['ALTITUDE.BOUNDARIES'] # altitude limits in km
        
    def resample_middays(self,key):
        '''
            TODO: just get midday averages using resampling in dataframe
        '''
        # First just pull out measurements within the 13-14 window
        inds        = [ d.hour == 13 for d in self.dates ]
        middays     = np.array(self.dates)[inds]
        middatas    = {}
        for key in ['VC','VC_apri']:
            middatas[key] = np.array(getattr(self,key))[inds]
        
        
    def Deconvolve(self,ModelledProfile):
        '''
            Return what instrument would see if modelled profile was the Truth
            VMR = APRI + AK * (True - APRI) ?? Check this
        '''
        print("Not Implemented, TODO")
        return None