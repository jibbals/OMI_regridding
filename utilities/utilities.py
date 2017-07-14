#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:36:52 2017

    Helpful functions that are used around the place...

@author: jesse
"""

##############################
### LIBRARIES/MODULES ########
##############################
# imports:
from datetime import datetime, timedelta
import calendar
import numpy as np
from scipy.interpolate import griddata # for regrid function

###############
### GLOBALS ###
###############
__VERBOSE__=False

###############
### METHODS ###
###############

def date_from_gregorian(greg):
    '''
        gregorian = "hours since 1985-1-1 00:00:0.0"
        Returns list of datetimes
    '''
    d0=datetime(1985,1,1,0,0,0)
    greg=np.array(greg)
    #if isinstance(greg, (list, tuple, np.ndarray)):
    return([d0+timedelta(seconds=int(hr*3600)) for hr in greg])

def findvminmax(data,lats,lons,region):
    '''
    return vmin, vmax of data[lats,lons] within region: list=[SWNE]
    '''
    latinds,loninds=lat_lon_range(lats,lons,region)
    data2=data[latinds,:]
    data2=data2[:,loninds]
    vmin=np.nanmin(data2)
    vmax=np.nanmax(data2)
    return vmin,vmax

def gregorian_from_dates(dates):
    ''' gregorian array from datetime list'''
    d0=datetime(1985,1,1,0,0,0)
    return np.array([(date-d0).seconds/3600 for date in dates ])

def index_from_gregorian(gregs, date):
    '''
        Return index of date within gregs array
    '''
    greg =(date-datetime(1985,1,1,0,0,0)).days * 24.0
    if __VERBOSE__:
        print("gregorian %s = %.2e"%(date.strftime("%Y%m%d"),greg))
        print(gregs)
    ind=np.where(gregs==greg)[0]
    return (ind)

def last_day(date):
    lastday=calendar.monthrange(date.year,date.month)[1]
    dayn=datetime(date.year,date.month,lastday)
    return dayn

def lat_lon_range(lats,lons,region):
    '''
    returns indices of lats, lons which are within region: list=[S,W,N,E]
    '''
    S,W,N,E = region
    lats,lons=np.array(lats),np.array(lons)
    latinds1=np.where(lats>=S)[0]
    latinds2=np.where(lats<=N)[0]
    latinds=np.intersect1d(latinds1,latinds2, assume_unique=True)
    loninds1=np.where(lons>=W)[0]
    loninds2=np.where(lons<=E)[0]
    loninds=np.intersect1d(loninds1,loninds2, assume_unique=True)
    return latinds, loninds

def list_days(day0,dayn=None,month=False):
    '''
        return list of days from day0 to dayn, or just day0
        last day is excluded, so 20050101 - 20050201 is just January
        if month is True, return [day0,...,end_of_month]
    '''
    if month:
        dayn=last_day(day0)
    if dayn is None: return [day0,]
    numdays = (dayn-day0).days + 1 # timedelta
    return [day0 + timedelta(days=x) for x in range(0, numdays)]

def ppbv_to_molecs_per_cm2(ppbv, pedges):
    '''
    Inputs:
        ppbv[levs,lats,lons]
        pedges[levs]
    USING http://www.acd.ucar.edu.au/mopitt/avg_krnls_app.pdf
    NOTE: in pdf the midpoints are used rather than the edges
        due to how the AK is defined
    '''
    dims=np.shape(ppbv)
    N = dims[0]
    #P[i] - P[i+1] = pressure differences
    inds=np.arange(N)
    #THERE should be N+1 pressure edges since we have the ppbv for each slab
    diffs= pedges[inds] - pedges[inds+1]
    t = (2.12e13)*diffs # multiplication factor
    out=np.zeros(dims)
    # there's probably a good way to vectorise this, but not worth the time
    for x in range(dims[1]):
        for y in range(dims[2]):
            out[:,x,y] = t*ppbv[:,x,y]
    return out

def regrid(data,lats,lons,newlats,newlons):
    '''
    Regrid a data array [lat,lon] onto [newlat,newlon]
    Assumes a regular grid!
    '''
    if __VERBOSE__:
        print("utilities.regrid transforming %s to %s"%(str((len(lons),len(lats))),str((len(newlons),len(newlats)))))
        print("data input looks like %s"%str(np.shape(data)))
    # if no resolution change then just throw back input
    if len(newlats) == len(lats):
        return data

    # make into higher resolution
    if len(newlats) > len(lats):
        mlons,mlats = np.meshgrid(lons,lats)
        mnewlons,mnewlats = np.meshgrid(newlons,newlats)

        interp = griddata( (mlats.ravel(), mlons.ravel()), data.ravel(),
                          (mnewlats, mnewlons), method='nearest')
        return interp

    # transform to lower resolution
    print("UNTESTED REGRIDDING")
    avgbefore=np.nanmean(data)
    if len(newlats) < len(lats):
        ni, nj = len(newlats), len(newlons)
        interp = np.zeros([ni,nj]) + np.NaN
        for i in range(ni):
            latlower=newlats[i]
            if i == ni-1: # final lat
                latupper=89.99
            else:
                latupper=newlats[i+1]
            lat=lats[i]
            irange = np.where((lat >= latlower) * (lat < latupper))[0]
            for j in range(nj):
                lonlower=newlons[j]
                if j == nj-1: # final lat
                    lonupper=179.99
                else:
                    lonupper=newlons[j+1]
                lon=lons[i]
                jrange = np.where((lon >= lonlower) * (lon < lonupper))
                interp[i,j] = np.nanmean(data[irange,jrange])
        assert np.isclose(avgbefore, np.nanmean(interp)), "Average changes too much!"
        return interp
    return None
