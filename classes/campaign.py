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
from utilities import utilities as util
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
                 'PRESSURE_INDEPENDENT':'p', # pressure mid level
                 #'ALTITUDE':'alts', # altitude mid level not on time dim
                 #'ALTITUDE.BOUNDARIES':'alts_e', # altitude edges not on time dim
                 'DATETIME':'dates', # dates in MJD2000
                 'SURFACE.PRESSURE_INDEPENDENT':'psurf', 
                 }

class Wgong(campaign):
    '''
    '''
    def __init__(self):#, year=datetime(2007,1,1)):
        '''
            Read the h5 data
        '''
        # read first year (200711 - 20071231)
        datadir='Data/campaigns/Wgong/'
        data, attrs= fio.read_hdf5(datadir+'ftir_2007.h5')
        self.attrs={}
        for key in __ftir_keys__.keys():
            nicekey=__ftir_keys__[key]
            setattr(self, nicekey, data[key])
            self.attrs[nicekey] = attrs[key]
            if __VERBOSE__:
                print("FTIR Reading : ",key, nicekey, data[key].shape)
        
        # Read 20080101-20121231
        for year in np.arange(2008,2013):
            data, attrs= fio.read_hdf5(datadir+'ftir_%d.h5'%year)
            # extend along time dim for things we want to keep
            for key in __ftir_keys__.keys():
                nicekey=__ftir_keys__[key]
                array=getattr(self,nicekey)
                array = np.append(array, data[key], axis=0)
                setattr(self, nicekey, array)
        
        self.DOF = np.trace(self.VMR_AK, axis1=1,axis2=2)
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
        
    def resample_middays(self):
        '''
            midday averages using resampling in dataframe
            Returns dictionary and alsoo stores output as middatas
        '''
        # if recalled, just use last calculation
        if hasattr(self,'middatas'):
            return self.middatas
        
        # First just pull out measurements within the 13-14 window
        if __VERBOSE__:
            print("Middays to be resampled between ",self.dates[0], self.dates[-1])
        inds        = np.array([ d.hour == 13 for d in self.dates ])
        middays     = np.array(self.dates)[inds]
        middatas    = {}
        for key in ['VC','VC_apri', 'VMR', 'VMR_apri', 'VC_AK','p', 'psurf', 'DOF']:
            # pull out midday entries
            middata = np.array(getattr(self,key))[inds]
            # save into a DataFrame
            mids = pd.DataFrame(middata,index=middays)
            # resample to get daily mean values
            daily = mids.resample('D',axis=0).mean()
            # save to dict
            middatas[key] = daily
        
        # VMR Avg Kernal is 3-D, need to reesample manually.....!!
        days=middatas['VC'].index.to_pydatetime()
        middatas['VMR_AK'] = np.zeros([len(days),48,48]) + np.NaN
        for i,day in enumerate(days):
            # for each day where midday data exists
            dinds = np.array([ (d.year == day.year) and ( d.month==day.month) and (d.day==day.day) for d in middays ])
            #if i == 0:
            #    print(self.VMR_AK.shape, self.VMR_AK[inds].shape, self.VMR_AK[inds][dinds].shape)
            #elif i < 50:
            #    print(self.VMR_AK[inds][dinds].shape)
            if np.sum(dinds) < 1:
                continue
            middatas['VMR_AK'][i] = np.nanmean(self.VMR_AK[inds][dinds], axis=0)
        # remove hours and store datetimes
        just_dates0 = datetime(middays[0].year, middays[0].month, middays[0].day)
        just_dates1 = datetime(middays[-1].year, middays[-1].month, middays[-1].day)
        if __VERBOSE__:
            print("Middays resampled: between ",just_dates0, just_dates1)
        middatas['dates']=util.list_days(just_dates0,just_dates1)
        self.middatas=middatas
        return middatas
        
        
        
    def Deconvolve(self,x_m, dates, p, checkname='Figs/FTIR_check_interpolation.png'):
        '''
            Return what instrument would see if modelled profile was the Truth
            VMR = APRI + AK * (True - APRI)
            x_m' = APRI + AK * (x_m - APRI)
            # everything is calculated after flipping the input vertical dim to go from toa to surf
        
        '''
        # Copy inputs, flip so that vertical dim is from TOA to surf
        # first dim is time, second dim is pressures
        x_m = np.copy(np.fliplr(x_m)) 
        dates=np.copy(dates)
        p=np.copy(np.fliplr(p))
        
        # get midday columns
        middatas=self.resample_middays()
        x_a = np.copy(middatas['VMR_apri'])
        ftdates = np.copy(middatas['dates'])
        A = np.copy(middatas['VMR_AK'])
        ftp = np.copy(middatas['p'])
        x_ret = np.copy(middatas['VMR'])
        
        if __VERBOSE__:
            print("model dates",dates[0],'..',dates[-1])
            print("ftir dates", ftdates[0],'..',ftdates[-1])
            print("subsetting...")
        # First just subset x_m to the same dates that we have
        # if input starts before ftir, cut input
        if dates[0] < ftdates[0]:
            dpre = util.date_index(ftdates[0],dates)[0]
            #print('dpre0:',dpre)
            #print( dates[0], ftdates[0])
            x_m = x_m[dpre:]
            dates=dates[dpre:]
            p = p[dpre:]
        # else cut ftir
        elif dates[0] > ftdates[0]:
            dpre = util.date_index(dates[0],ftdates)[0]
            #print('dpre1:',dpre, dates[0], ftdates[dpre])
            x_a = x_a[dpre:]
            ftdates=ftdates[dpre:]
            ftp = ftp[dpre:]
            A = A[dpre:]
            x_ret = x_ret[dpre:]
        # if input ends before ftir, then cut ftir down
        if dates[-1] < ftdates[-1]:
            dpost = util.date_index(dates[-1],ftdates)[0] + 1 # need to add 1 to get right subset
            #print('dpost0:',dpost)
            #print( dates[-1], ftdates[-1])
            x_a = x_a[:dpost]
            ftdates=ftdates[:dpost]
            ftp = ftp[:dpost]
            A = A[:dpost]
            x_ret = x_ret[:dpost]
        # else if input ends after ftir cut input down
        elif dates[-1] > ftdates[-1]:
            dpost = util.date_index(ftdates[-1], dates)[0]  + 1 # need to add 1 to get right subset
            #print('dpost1:',dpost)
            #print( dates[-1], ftdates[-1])
            x_m = x_m[:dpost]
            dates=dates[:dpost]
            p=p[:dpost]
        
        # check subsetting dates worked OK
        if __VERBOSE__:
            print("model dates",dates[0],'..',dates[-1])
            print("ftir dates", ftdates[0],'..',ftdates[-1])
        assert np.all(dates == ftdates), "dates don't match after subsetting"
        
        # Now make sure x_m is interpolated to the same vertical resolution...
        matched_x_m = np.copy(x_a)
        new_x_m     = np.copy(x_a)
        check       = checkname is not None
        for i in range(len(dates)):
            if not np.all(np.isnan(x_a[i])):
                # interpolate to FTIR pressures
                matched_x_m[i] = np.interp(ftp[i],p[i],x_m[i],left=None,right=None)
                
                new_x_m[i] = x_a[i] + np.matmul(A[i],(matched_x_m[i] - x_a[i]))
        
                # can check last available column
                checki = i
                
                if check:
                    
                    plt.plot(x_m[i],p[i],':x',label='x$_{GC}$')
                    plt.plot(matched_x_m[i], ftp[i], '--+', label='interpolated x$_{GC}$')
                    plt.plot(1000.0*x_a[i], ftp[i], '--1', label='x$_{apri}$')
                    plt.legend(loc='best')
                    plt.yscale('log')
                    plt.ylim([1.2e3, 5e1])
                    plt.ylabel('pressure [hPa]')
                    plt.xlabel('HCHO [ppbv]')
                    plt.savefig(checkname)
                    plt.close()
                    print("Saved ", checkname)
                    check = False
        
        TOA = np.zeros(np.shape(dates))*ftp[:,0] # zeros or nans as TOA
        pedges = (ftp[:,1:]+ftp[:,0:-1])/(2.0)
        psurf  = ftp[:,-1] + (ftp[:,-1] - ftp[:,-2])/2.0 # approximated by extension
        #print('pmids : ',np.shape(ftp),ftp[checki,-6:])
        #print('pedges: ',np.shape(pedges),pedges[checki,-5:])
        #print('adding TOA and psurf : ',np.shape(TOA),TOA[checki],np.shape(psurf),psurf[checki])
        
        # insert TOA at start of pedges
        pedges = np.insert(pedges, 0, TOA, axis=1)
        # append psurf to end, needs to be same number of dims as pedges
        psurf = np.expand_dims(psurf,axis=1)
        pedges = np.append(pedges, psurf, axis=1)
        
        #print('pedges: ',np.shape(pedges),pedges[checki,:6],'...',pedges[checki,-6:])
        
        # ppbv -> molec/cm2 assuming dry air profile
        dp = pedges[:,1:] - pedges[:,:-1]
        
        new_TC = np.sum(new_x_m*2.12e13 * dp, axis=1)
        orig_TC = np.sum(matched_x_m * 2.12e13 * dp, axis=1)
        TC_ret = np.sum(x_ret * 2.12e16 * dp, axis=1) # ppmv -> molec/cm2
        
        
        return {'new_x_m':new_x_m, 'dates':dates, 'p':ftp,
                # Delta p and pedges
                'dp':dp, 'pedges':pedges, 
                # Original and interpolated profiles
                'x_m':x_m, 'matched_x_m':matched_x_m, 
                # FTIR outputs
                'x_a':x_a, 'x_ret':x_ret, 'A':A, 
                # TOtal columns
                # new_TC is total column after deconvolution, orig_TC is after interpolation only, TC_ret is the ftir total column for matching day
                'new_TC':new_TC, 'orig_TC':orig_TC, 'TC_ret':TC_ret}
