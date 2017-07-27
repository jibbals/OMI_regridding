# -*- coding: utf-8 -*-
"""
Created 20170216

Preprocess satellite and GC datasets for randal martin's code to calculate the AMF

@author: jesse
"""

# module to read/write data to file
import reprocess
import numpy as np
import csv

from datetime import timedelta, datetime
#from glob import glob

# GLOBALS
__VERBOSE__=True # set to true for more print statements
__DEBUG__=True # set to true for even more print statements
#_AusNESW=[-10,160, -50, 110] # Australian region
#_RefSect=[90,-140, -90, -160] # reference sector
#
# function to create .csv from swaths:
#
def pixel_list_to_csv(date=datetime(2005,1,1),nesw=None):
    '''
    Read good pixel list, 
        These pixels are read from the omi swath dataset
        [optional: pull out the section we want,]
        save to csv for amf program to use.
    CSV: 
    linenumber, scan, pixel,lat, lon, sza, sva, cloud frac, cloud top pressure
    '''
    fname='Data/omhcho_csv/%s_for_AMF.csv'%date.strftime('%Y-%m-%d')
    csv_params=['scan','track', 'lat', 'lon', 'sza','vza','cloudfrac','ctp' ]
    
    # list of good pixels
    # We are going to create the Palmer AMFs, not read them
    gp=reprocess.get_good_pixel_list(date, PalmerAMF=False) 
    
    # cut the pixel list down to a desired region:
    lats=np.array(gp['lat'])
    lons=np.array(gp['lon'])
    zone=np.ones(len(lats),dtype=bool)
    if nesw is not None:
        zone=(lats < nesw[0]) * (lons < nesw[1]) * (lats > nesw[2]) * (lons > nesw[3])
        assert (np.max(lats[zone]) < nesw[0]), "subsetting region didn't work"
        if __DEBUG__: 
            print("range cut down to (s,w,n,e): %s"%str((min(lats[zone]),min(lons[zone]),max(lats[zone]),max(lons[zone]))))
        # One day for AUS cuts pixels from ~ 700k to 38k
    
    # save the 9 parameters we want for randal martin AMF package to csv:
    pixelnumber=np.arange(0,len(lats))
    # Create a list of 9 lists, each with the same length, then transpose
    # to get an array of x rows and 9 columns
    lists=[list(pixelnumber[zone])] # an id number to map the pixels to the AMFs
    if __DEBUG__: print("%d entries for column %s"%(len(lists[-1]),'pixelID'))
    for name in csv_params:
        lists.append(list(np.array(gp[name])[zone]))
        if __DEBUG__: print("%d entries for column %s"%(len(lists[-1]),name))
    
    # Transpose the list
    rows=list(map(list,zip(*lists)))
    
    with open(fname,'w') as f:
        writer=csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print("Saved: %s"%fname)
