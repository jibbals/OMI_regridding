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
from datetime import datetime, timedelta, timezone
import calendar
import numpy as np
from scipy.interpolate import griddata # for regrid function
from mpl_toolkits.basemap import maskoceans #
from utilities import GMAO

###############
### GLOBALS ###
###############
__VERBOSE__=True
__grams_per_mole__={'isop':60.06+8.08, # C5H8
                    'hcho':30.02598,
                    'carbon':12.01}
NA     = GMAO.edges_containing_region([-21,115,-11,150])
SWA    = GMAO.edges_containing_region([-36,114,-29,128])
SEA    = GMAO.edges_containing_region([-39,144,-29,153])

###############
### METHODS ###
###############

def area_quadrangle(SWNE):
    '''
        Return area of sphere with earths radius bounded by S,W,N,E quadrangle
        units = km^2
    '''
    #Earths Radius
    R=6371.0
    # radians from degrees
    S,W,N,E=SWNE
    Sr,Wr,Nr,Er = np.array(SWNE)*np.pi/180.0
    # perpendicular distance from plane containing line of latitude to the pole
    # (checked with trig)

    h0=R*(1-np.sin(Sr))
    h1=R*(1-np.sin(Nr))

    # Area north of a latitude: (Spherical cap - wikipedia)
    A0= 2*np.pi*R*h0
    A1= 2*np.pi*R*h1
    A_zone= A0-A1 # Area of zone from south to north

    # portion between longitudes
    p=(E-W)/360.0

    # area of quadrangle
    A= A_zone*p
    return A

def area_grid(lats,lons, latres, lonres):
    '''
        Area give lats and lons in a grid in km^2
    '''
    areas=np.zeros([len(lats),len(lons)]) + np.NaN
    yr,xr=latres/2.0,lonres/2.0

    for yi,y in enumerate(lats):
        for xi, x in enumerate(lons):
            if not np.isfinite(x+y):
                continue
            SWNE=[y-yr, x-xr, y+yr, x+xr]
            areas[yi,xi] = area_quadrangle(SWNE)
    return areas

def combine_dicts(d1,d2):
    '''
    Add two dictionaries together
    '''
    return dict(d1.items() + d2.items() + [ (k, d1[k] + d2[k]) for k in set(d2) & set(d1) ])

def date_from_gregorian(greg):
    '''
        gregorian = "hours since 1985-1-1 00:00:0.0"
        Returns list of datetimes
    '''
    d0=datetime(1985,1,1,0,0,0)
    greg=np.array(greg)
    if greg.ndim==0:
        return([d0+timedelta(seconds=int(greg*3600)),])
    return([d0+timedelta(seconds=int(hr*3600)) for hr in greg])

def date_index(date,dates):

    whr=np.where(np.array(dates) == date) # returns (matches_array,something)
    if len(whr[0])==0:
        print (date, 'not in', dates)

    return whr[0][0] # We just want the match

def datetimes_from_np_datetime64(times, reverse=False):
    # '2005-01-01T00:00:00.000000000'
    if reverse:
        return [np.datetime64(d.strftime('%Y-%m-%dT%H:%M:%S.000000000')) for d in times]
    return [datetime.strptime(str(d),'%Y-%m-%dT%H:%M:%S.000000000') for d in times]


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
    if maskocean or maskland:
        if len(np.shape(lats)) == 1:
            lons,lats = np.meshgrid(lons,lats)
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

def lat_lon_index(lat,lon,lats,lons):
    ''' lat,lon index from lats,lons    '''
    with np.errstate(invalid='ignore'):
            latind=(np.abs(lats-lat)).argmin()
            lonind=(np.abs(lons-lon)).argmin()
    return(latind,lonind)

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

def lat_lon_subset(lats,lons,region, data=[]):
    '''
        Returns dict with lats, lons, lats_e, lons_e, and each data aray subsetted
        data should be list of arrays
        Returned list will be list of arrays [lats,lons] along with lats, lats_e...
    '''
    lati,loni=lat_lon_range(lats,lons,region)
    lats_m,lons_m = lats[lati],lons[loni]
    lats_e=edges_from_mids(lats_m)
    lons_e=edges_from_mids(lons_m)
    out={'lats':lats_m,'lons':lons_m,'lats_e':lats_e,'lons_e':lons_e,
         'data':[], 'lati':lati, 'loni':loni, }
    for arr in data:
        # if lats is second dimension:
        if (len(lats) != len(lons)) and (len(lats)==np.shape(arr)[1]):
            arr=arr.T # make it arr[lats,lons]
        arr=arr[lati,:]
        arr=arr[:,loni]
        out['data'].append(arr)
    return out

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

def list_days_strings(day0,dayn=None,month=False,fmt='%Y%m%d'):
    '''
    '''
    days=list_days(day0,dayn=dayn,month=month)
    return [day.strftime(fmt) for day in days]

def list_months(day0,dayn):
    '''
        Return list of months (day=1) included between day0 and dayN
    '''
    # first get list of days
    days=list_days(day0,dayn)
    # Just pull out entries with day==1
    months=[d for d in days if d.day==1]
    return months

def local_time_offsets(lons,n_lats=0, astimedeltas=False):
    '''
        GMT is 12PM, AEST is + 10 hours, etc...
        offset by one hour every 15 degrees
    '''
    offset = np.array(lons) // 15

    if n_lats > 0:
        # lats,lons
        offset=np.transpose(np.repeat(offset[:,np.newaxis],n_lats,axis=1)).astype(int)

    if astimedeltas:
        # hours to ms
        offset=offset * 3600 * 1000
        timedeltas=np.array(offset, dtype='timedelta64[ms]')
        return timedeltas

    return offset


def monthly_averaged(dates,data):
    '''
        return monthly averaged version of inputs
    '''
    months=list_months(dates[0],dates[-1])
    allyears=np.array([d.year for d in dates])
    allmonths=np.array([d.month for d in dates])
    #ind = [100*d.year+d.month for d in dates]
    ret={}
    mdates=[]
    mdata=[]
    mstd=[]
    mcount=[]
    for m in months:
        inds=(allyears==m.year) * (allmonths==m.month)
        mdates.append(m)
        mdata.append(np.nanmean(data[inds]))
        mstd.append(np.nanstd(data[inds]))
        mcount.append(np.nansum(inds))
    mdata=np.array(mdata); mstd=np.array(mstd); mcount=np.array(mcount);
    mid_dates=[d+timedelta(days=15) for d in mdates]
    return {'dates':mdates, 'data':mdata, 'std':mstd,'count':mcount, 'middates':mid_dates}

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

def regrid_to_lower(data, lats, lons, newlats_e, newlons_e):
    '''
        Regrid data to lower resolution
        using EDGES of new grid and mids of old grid
    '''
    ret=np.zeros([len(newlats_e)-1,len(newlons_e)-1])+np.NaN
    for i in range(len(newlats_e)-1):
        for j in range(len(newlons_e)-1):
            lati= (lats >= newlats_e[i]) * (lats < newlats_e[i+1])
            loni= (lons >= newlons_e[j]) * (lons < newlons_e[j+1])

            tmp=data[lati,:]
            tmp=tmp[:,loni]
            ret[i,j]=np.nanmean(tmp)
    return ret

def regrid(data,lats,lons,newlats,newlons):
    '''
    Regrid a data array [lat,lon] onto [newlat,newlon]
    Assumes a regular grid, and that boundaries are compatible!!
    '''
    if __VERBOSE__:
        print("utilities.regrid transforming %s to %s"%(str((len(lats),len(lons))),str((len(newlats),len(newlons)))))
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

def reshape_time_lat_lon_lev(data,ntimes,nlats,nlons,nlevs):
    ''' return reference to data array with time,lat,lon,lev dims '''

    # grab array and shape
    shp=np.array(data.shape)
    n_dims=len(shp)

    #if __VERBOSE__:
    #    print("changing shape:",shp,'->',ntimes,nlats,nlons,nlevs)

    if ntimes is None:
        ntimes = -1
    if nlevs is None:
        nlevs = -2

    # make sure data is not square
    if len(set([ntimes,nlats,nlons,nlevs])) < 4:
        print("ERROR: could not reshape data array")
        print(shp,'->',ntimes,nlats,nlons,nlevs)
        return data

    if n_dims>1:
        lati=np.argwhere(shp==nlats)[0,0]
        loni=np.argwhere(shp==nlons)[0,0]
        newshape=(lati,loni)

        # do we have time and level dimensions?
        ti=np.argwhere(shp==ntimes)
        levi=np.argwhere(shp==nlevs)

        if len(ti)==1 and len(levi)==1:
            newshape=(ti[0,0],lati,loni,levi[0,0])
        elif len(ti)==0 and len(levi)==1:
            newshape=(lati,loni,levi[0,0])
        elif len(ti)==1 and len(levi)==0:
            newshape=(ti[0,0],lati,loni)

        arr=np.transpose(data,axes=newshape)
        if __VERBOSE__:
            print('changed data array shape:',shp," -> ",np.shape(arr))

    return arr


def utc_to_local(utc_dt):
    '''
        Convert utc time tom local time (of computer I guess?)
    '''
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
