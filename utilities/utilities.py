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
from mpl_toolkits.basemap import maskoceans #

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

def edges_from_mids(x,fix=False):
    '''
        Take a lat or lon vector input and return the edges
        Works for REGULAR grids only
    '''
    assert x[1]-x[0] == x[2]-x[1], "Resolution at edge not representative"
    # replace assert with this if it works, HANDLES GEOS CHEM LATS PROBLEM ONLY
    if x[1]-x[0] != x[2]-x[1]:
        xres=x[2]-x[1]   # Get resolution away from edge
        x[0]=x[1]-xres   # push out the edges
        x[-1]=x[-2]+xres #

    # new vector for array
    newx=np.zeros(len(x)+1)
    # resolution from old vector
    xres=x[1]-x[0]
    # edges will be mids - resolution / 2.0
    newx[0:-1]=np.array(x) - xres/2.0
    # final edge
    newx[-1]=newx[-2]+xres

    # Finally if the ends are outside 90N/S or 180E/W then bring them back
    if fix:
        if newx[-1] >= 90: newx[-1]=89.99
        if newx[0] <= -90: newx[0]=-89.99
        if newx[-1] >= 180: newx[-1]=179.99
        if newx[0] <= -180: newx[0]=-179.99

    return newx

def edges_to_mids(x):
    ''' lat or lon array and return midpoints '''
    out=np.zeros(len(x)-1)
    out= (x[0:-1]+x[1:]) / 2.0
    return out

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

def get_mask(arr, lats=None, lons=None, masknan=True, maskocean=False, maskland=False):
    '''
        Return array which is a mask for the input array
        to mask the ocean or land you need to put in the lats, lons of the data
    '''
    mask=np.isnan(arr)
    if maskocean:
        mask = mask + maskoceans(lons,lats,arr, inlands=False).mask
    if maskland:
        mask = mask + ~(maskoceans(lons,lats,arr, inlands=False).mask)
    return mask

def get_masked(arr, lats=None, lons=None, masknan=True, maskocean=False, maskland=False):
    '''
        return array masked by nans and optionally ocean/land
    '''
    mask=get_mask(arr, lats=None, lons=None, masknan=True, maskocean=False, maskland=False)
    return np.ma.masked_array(arr,mask=mask)

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
    Assumes a regular grid, and that boundaries are compatible!!
    '''
    if __VERBOSE__:
        print("utilities.regrid transforming %s to %s"%(str((len(lons),len(lats))),str((len(newlons),len(newlats)))))
        print("data input looks like %s"%str(np.shape(data)))
    interp=None
    # if no resolution change then just throw back input
    if len(newlats) == len(lats) and len(newlons) == len(lons):
        if all(newlats == lats) and all(newlons == lons):
            return data

    # If new lats have higher resolution
    #elif len(newlats) > len(lats):
    # make into higher resolution
    mlons,mlats = np.meshgrid(lons,lats)
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)

    #https://docs.scipy.org/doc/scipy/reference/interpolate.html
    interp = griddata( (mlats.ravel(), mlons.ravel()), data.ravel(),
                      (mnewlats, mnewlons), method='nearest')

    # Check shape is as requested
    assert np.shape(interp)== (len(newlats),len(newlons)), "Regridded shape new lats/lons!"

    return interp
