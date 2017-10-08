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

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

from utilities import fio
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
    

    def read_SPS1(self):
        
        
        
        # TODO: remove ../ when not testing any more
        self.fpath='../Data/campaigns/SPS1/SPS1_PTRMS.csv'
        data=fio.read_csv(self.fpath)
        # PTRMS names the columns with m/z ratio, we use 
        #   HCHO = 31, ISOP = 69
        self.hcho=data['m/z 31']
        self.isop=data['m/z 69']
        # Timestamp like this: 18/02/2011 17:00
        dates=data['Timestamp']
        
        # First row is detection limits
        
        # second row is empty
        
        # data begins at third row
        
        print(dates[2:])
        #self.dates=[datetime.strptime('%d/%m/%Y %H',d) for d in dates]
        
        #print(data)
            
if __name__=='__main__':
    sps1=campaign()
    sps1.read_SPS1()