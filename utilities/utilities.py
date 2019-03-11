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
import timeit
import concurrent.futures # parallelism
import pandas as pd # for daily cycles grouping

from utilities import GMAO, JesseRegression

###############
### GLOBALS ###
###############
__VERBOSE__=False
__grams_per_mole__={'isop':60.06+8.08, # C5H8
                    'hcho':30.02598,
                    'carbon':12.01}
NA     = GMAO.edges_containing_region([-21,115,-11,150])
SWA    = GMAO.edges_containing_region([-36,114,-29,128])
SEA    = GMAO.edges_containing_region([-39,144,-29,153])
# Want to look at timeseires and densities in these subregions:
__subregions__ = GMAO.__subregions__
__subregions_colors__ = GMAO.__subregions_colors__
__subregions_labels__ = GMAO.__subregions_labels__


# Remote pacific as defined in De Smedt 2015 [-15, 180, 15, 240]
# [lat,lon,lat,lon]
__REMOTEPACIFIC__=[-15, -180, 15, -120]

#__MISSING_OMHCHORP_DAY__=fio.read_omhchorp_day(datetime(2005,1,1))
#print("__MISSING_OMHCHORP_DAY__={")
#for key,entry in __MISSING_OMHCHORP_DAY__.items():
#    #print(key, type(entry))
#    if hasattr(entry, 'shape'):
#        print("    '%s':np.zeros(%s)+np.NaN,"%(key,entry.shape))
#    else:
#        print("    '%s':[],"%(key))
#print("}")
__MISSING_OMHCHORP_DAY__={
    'AMF_PP':np.zeros((721, 1152))+np.NaN,
    'RSC_latitude':np.zeros((500,))+np.NaN,
    'RSC_GC':np.zeros((500,))+np.NaN,
    'AMF_GCz':np.zeros((721, 1152))+np.NaN,
    'VCC_OMI':np.zeros((721, 1152))+np.NaN,
    'gridentries':np.zeros((721, 1152))+np.NaN,
    'col_uncertainty_OMI':np.zeros((721, 1152))+np.NaN,
    'longitude':np.zeros((1152,))+np.NaN,
    'VCC_PP':np.zeros((721, 1152))+np.NaN,
    'VC_OMI':np.zeros((721, 1152))+np.NaN,
    'VC_PP':np.zeros((721, 1152))+np.NaN,
    'mod_times':[],
    'AMF_GC':np.zeros((721, 1152))+np.NaN,
    'SC':np.zeros((721, 1152))+np.NaN,
    'RSC_region':np.zeros((4,))+np.NaN,
    'latitude':np.zeros((721,))+np.NaN,
    'ppentries':np.zeros((721, 1152))+np.NaN,
    'RSC':np.zeros((500, 60, 3))+np.NaN,
    'VCC_OMI_newrsc':np.zeros((721, 1152))+np.NaN,
    'VCC_GC':np.zeros((721, 1152))+np.NaN,
    'AMF_OMI':np.zeros((721, 1152))+np.NaN,
    'VC_GC':np.zeros((721, 1152))+np.NaN,
}

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

def area_grid(lats, lons):
    '''
        Area give lats and lons in a grid in km^2
        can do non grid with provided lat, lon arrays

        Lats and Lons are centres of gridpoints
    '''
    areas=np.zeros([len(lats),len(lons)]) + np.NaN
    latres=np.abs(lats[2]-lats[1])
    lonres=np.abs(lons[2]-lons[1])
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
        Returns nparray of datetimes
    '''
    d0=datetime(1985,1,1,0,0,0)
    greg=np.array(greg)
    if greg.ndim==0:
        return np.array( [d0+timedelta(seconds=int(greg*3600)),])
    return np.array([d0+timedelta(seconds=int(hr*3600)) for hr in greg])

def date_from_mjd2k(mjd2k):
    '''
        gregorian = "days since 1, jan, 2000 00:00:00"
        Returns list of datetimes
    '''
    d0=datetime(2000,1,1,0,0,0)
    mjd2k=np.array(mjd2k)
    if mjd2k.ndim==0:
        return([d0+timedelta(seconds=int(mjd2k*3600*24)),])
    return([d0+timedelta(seconds=int(hr*3600*24)) for hr in mjd2k])

def date_index(date,dates, dn=None):

    whr=np.where(np.array(dates) == date) # returns (matches_array,something)
    if len(whr[0])==0:
        print (date, 'not in', dates[0], '...', dates[-1])
    elif dn is None:
        return np.array([whr[0][0]]) # We just want the match
    else:
        whrn=np.where(np.array(dates) == dn) # returns last date match
        if len(whrn[0])==0: # last date not in dataset
            print (dn, 'not in', dates[0], '...', dates[-1])
        return np.arange(whr[0][0],whrn[0][0]+1)


def datetimes_from_np_datetime64(times, reverse=False):
    # '2005-01-01T00:00:00.000000000'

    if reverse:
        return np.squeeze([np.datetime64(d.strftime('%Y-%m-%dT%H:%M:%S.000000000')) for d in times])
    return np.squeeze([datetime.strptime(str(d),'%Y-%m-%dT%H:%M:%S.000000000') for d in times])


def edges_from_mids(x,fix_max=179.99):
    '''
        Take a lat or lon vector input and return the edges
        Works for monotonic increasing grids only
        Doesn't matter if grid is irregular
    '''
    if __VERBOSE__:
        print("VERBOSE: CHECK Edges_from_mids")
        print("MIDS : ", x[0:4],'...',x[-4:])

    # new vector for array
    newx=np.zeros(len(x)+1)
    # x left side = x minus half the distance to the next x
    # x right side = x plus half the distance to the next x
    # all x but first and last just take midpoints:
    newx[1:-1]  = (x[0:-1]+x[1:]) / 2.0
    # for very edges take again half the distance to the next inward
    newx[0]     = x[0] - (x[1]-x[0]) / 2.0
    newx[-1]    = x[-1] + (x[-1]-x[-2]) / 2.0

    if __VERBOSE__:
        print("EDGES: ", newx[0:5],'...',newx[-5:])

    # Finally if the ends are outside 90N/S or 180E/W then bring them back
    if fix_max is not None:
        if newx[-1] > fix_max: newx[-1]=fix_max
        if newx[0] < -1*fix_max: newx[0]=-1*fix_max

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
    mask=np.zeros(np.shape(arr),dtype=np.bool)
    if masknan:
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
    ''' gregorian array from datetime list
        gregorian is hours since 1985,1,1,0,0

    '''
    d0=datetime(1985,1,1,0,0,0)
    return np.array([(date-d0).total_seconds()/3600.0 for date in dates ])

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

def first_day(date):
    ''' Return first day in month matching input argument '''
    return datetime(date.year,date.month,1)

def last_day(date):
    ''' Return last day in month matching input argument '''
    lastday=calendar.monthrange(date.year,date.month)[1]
    dayn=datetime(date.year,date.month,lastday)
    return dayn

def lat_lon_grid(latres=GMAO.__LATRES__,lonres=GMAO.__LONRES__, regular=False):
    '''
    Returns lats, lons, latbounds, lonbounds for grid with input resolution
    By default this uses GMAO structure of half length lats at +- 90 degrees
        to turn this off use regular=True
    '''

    if regular:
        # lat and lon bin boundaries
        lat_bounds=np.arange(-90, 90+latres/2.0, latres)
        lon_bounds=np.arange(-180, 180+lonres/2.0, lonres)

        # lat and lon bin midpoints
        lats=np.arange(-90,90,latres)+latres/2.0
        lons=np.arange(-180,180,lonres)+lonres/2.0
    else:
        lats,lat_bounds=GMAO.GMAO_lats(latres)
        lons,lon_bounds=GMAO.GMAO_lons(lonres)

    return (lats,lons,lat_bounds,lon_bounds)

def lat_lon_index(lat,lon,lats,lons):
    ''' lat,lon index from lats,lons    '''
    with np.errstate(invalid='ignore'):
        latind=(np.abs(lats-lat)).argmin()
        lonind=(np.abs(lons-lon)).argmin()
    return latind,lonind

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

def lat_lon_subset(lats,lons,region, data=[], has_time_dim=False):
    '''
        Returns dict with lats, lons, lats_e, lons_e, and each data aray subsetted
        data should be list of arrays with dimensions [[time,]lats,lons]
        Returned list will be list of arrays [lats,lons] along with lats, lats_e...
    '''
    lati,loni=lat_lon_range(lats,lons,region)
    lats_m,lons_m = lats[lati],lons[loni]
    lats_e=edges_from_mids(lats_m)
    lons_e=edges_from_mids(lons_m)
    out={'lats':lats_m,'lons':lons_m,'lats_e':lats_e,'lons_e':lons_e,
         'data':[], 'lati':lati, 'loni':loni, }
    for arr in data:

        if has_time_dim:
            assert len(lats)==len(arr[0,:,0]), "Lats need to be second dimension"
            arr=arr[:,lati,:]
            arr=arr[:,:,loni]
        else:
            assert len(lats)==len(arr[:,0]), "Lats need to be first dimension"
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

def list_years(day0,dayn,dates=None):
    '''
        list years from day0 -> dayn
        if dates is supplied, return list of lists: one list for each year
    '''
    days=list_days(day0,dayn)
    yearset=set( d.year for d in days )
    years = [datetime(year,1,1) for year in yearset]

    if dates is not None:
        years=[]
        for year in yearset:
            yeardates = [d for d in dates if d.year==year]
            years.append(yeardates)
    return years

def list_to_lists(mylist,chunksize=30):
    '''
        split long list into lists of length 30
    '''
    return [ mylist[start:start+chunksize] for start in range(0, len(mylist), chunksize) ]

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

def match_bottom_levels(p1i, p2i, arr1i, arr2i):
    '''
    Takes two arrays of pressure (from surface to some altitude)
    Returns the same two arrays with the lowest levels set to the same pressure
    This method raises the lowest levels of the pressure arrays to match each other.
    Also interpolates arr1, arr2 to the new pressures if their pedges have changed
    '''
    # update:20160905, 20161109
    # one array will have a lower bottom level than the other
    #
    p1,p2=np.array(p1i.copy()),np.array(p2i.copy())
    arr1,arr2=np.array(arr1i.copy()),np.array(arr2i.copy())
    if p1[0] > p2[0]:
        plow=p1
        phigh=p2
        alow=arr1
    elif p1[0] == p2[0]:
        return p1,p2,arr1,arr2
    else:
        plow=p2
        phigh=p1
        alow=arr2
    plow_orig=plow.copy()

    # now lower has a lower surface altitude(higher surface pressure)
    movers=np.where(plow+0.05*plow[0] > phigh[0])[0] # index of pressure edges below higher surface pressure
    above_ind = movers[-1]+1
    if above_ind >= len(plow):
        print("Whole plow array below surface of phigh array")
        print("plow:",plow.shape)
        print(plow)
        print("phigh:",phigh.shape)
        print(phigh)
        assert False, "Fix this please"


    above = plow[above_ind] # pressure edge above the ones we need to move upwards
    rmovers=movers[::-1] # highest to lowest list of pmids to relevel
    # for all but the lowest pmid, increase above lowest pmid in other pressure array
    for ii in rmovers[0:-1]:
        plow[ii]=(phigh[0]*above)**0.5
        above=plow[ii]
    # for the lowest pmid, raise to match other array's lowest pmid
    plow[0]=phigh[0]

    # now interpolate the changed array ( reversed as interp wants increasing array xp)
    alow[:]=np.interp(plow,plow_orig[::-1],alow[::-1])

    return p1,p2,arr1,arr2

def multi_year_average(data,dates, grain='monthly'):
    '''
        Use pandas dataframes to get average for each month of the year
        grain = { 'hourly' | 'daily' | 'monthly' }
        ONLY WORKS ON 1D DATA ARRAY (time dim)
    '''

    # data is a list or 1d array of data, index is the datetimes of that same data
    df=pd.DataFrame(data=data, index=dates)

    # grouping monthly
    ind = [df.index.month,]

    # also grouping either daily or hourly
    if grain=='daily':
        ind.append(df.index.day)
    elif grain=='hourly':
        ind.append(df.index.hour)

    # grouping by hour and month, returns 288 (24x12) rows if doing hourly grain
    #  columns: count, mean, std, min, 25%, 50%, 75%, max
    return df.groupby(ind)
    # reshape to [month, hour]
    #rets[key] = data.values.reshape([12,24])
    #assert np.all(rets[key][0,:]==data[0:24]), 'reshape lost consistency'


def multi_year_average_regional(data,dates,lats,lons, grain='monthly',regions=__subregions__):
    '''
        Use pandas dataframes to get average for each month of the year
        grain = { 'hourly' | 'daily' | 'monthly' }
    '''

    rets={}
    rets['subsets'] = [] # keeping all the subsets
    rets['df']  = []
    for region in regions:
        tmp=lat_lon_subset(lats,lons,region,data=[data],has_time_dim=True)
        rets['subsets'].append(tmp)

        rseries=np.nanmean(tmp['data'][0], axis=(1,2)) # average over space
        # grouping by hour and month, returns 288 (24x12) rows if doing hourly grain
        #  columns: count, mean, std, min, 25%, 50%, 75%, max
        df = multi_year_average(rseries, dates , grain=grain)
        rets['df'].append(df)

    return rets
    # reshape to [month, hour]
    #rets[key] = data.values.reshape([12,24])
    #assert np.all(rets[key][0,:]==data[0:24]), 'reshape lost consistency'

def multi_year_average_spatial(data,dates):
    ''' multiyear monthly average over spatial dims '''

    allmonths = np.array([d.month for d in dates])

    # Things that get returned
    mdates=np.arange(1,13)
    mmean=[]
    mmedian=[]
    mstd=[]
    mcount=[]
    msum=[]
    for month in range(12):
        inds= allmonths == month+1

        mmedian.append(np.nanmedian(data[inds],axis=0))
        mmean.append(np.nanmean(data[inds],axis=0))
        mstd.append(np.nanstd(data[inds],axis=0))
        mcount.append(np.nansum(inds,axis=0))
        msum.append(np.nansum(data[inds],axis=0))

    mmean=np.array(mmean); mstd=np.array(mstd); mcount=np.array(mcount);
    mmedian=np.array(mmedian); msum=np.array(msum)
    return {'dates':mdates, 'mean':mmean, 'median':mmedian, 'sum':msum,
            'std':mstd,'count':mcount}

def monthly_averaged(dates,data,keep_spatial=False):
    '''
        return monthly averaged version of inputs
    '''
    months=list_months(dates[0],dates[-1])
    allyears=np.array([d.year for d in dates])
    allmonths=np.array([d.month for d in dates])
    #ind = [100*d.year+d.month for d in dates]

    # Things that get returned
    mdates=[]
    mmean=[]
    mmedian=[]
    mstd=[]
    mcount=[]
    msum=[]
    for m in months:
        inds=(allyears==m.year) * (allmonths==m.month)
        mdates.append(m)
        axis=[None,0][keep_spatial]

        mmedian.append(np.nanmedian(data[inds],axis=axis))
        mmean.append(np.nanmean(data[inds],axis=axis))
        mstd.append(np.nanstd(data[inds],axis=axis))
        mcount.append(np.nansum(inds,axis=axis))
        msum.append(np.nansum(data[inds],axis=axis))

    mmean=np.array(mmean); mstd=np.array(mstd); mcount=np.array(mcount);
    mmedian=np.array(mmedian); msum=np.array(msum)
    mid_dates=[d+timedelta(days=15) for d in mdates]
    return {'dates':mdates, 'mean':mmean, 'median':mmedian, 'sum':msum,
            'std':mstd,'count':mcount, 'middates':mid_dates}

def oceanmask(lats,lons,inlands=False):
    '''
        Return oceanmask, true over ocean squares
        inlands=False means don't mask inland water squares
    '''
    mlats,mlons=lats,lons
    if len(np.shape(lats)) == 1:
        mlons,mlats=np.meshgrid(lons,lats)
    # lonsin, latsin, datain arguments for maskoceans
    # we just want mask, so datain doesn't matter
    ocean=maskoceans(mlons,mlats,mlats,inlands=False).mask
    return ocean

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


def pull_out_subregions(data, lats, lons, subregions=__subregions__):
    ''' pull out subregions from data: returns list of data subsets, list of lats, list of lons '''

    # assume time dim if three dimensions
    has_time_dim = len(np.shape(data)) == 3
    outdata=[]
    outlats=[]
    outlons=[]

    for region in subregions:

        # pull out region:
        lati,loni = lat_lon_range(lats,lons,region)

        # my time dim is always the first one
        if has_time_dim:
            sub=data[:,lati,:]
            sub=sub[:,:,loni]
        else:
            sub=data[lati,:]
            sub=sub[:,loni]

        outdata.append(np.copy(sub))
        outlats.append(np.copy(lats[lati]))
        outlons.append(np.copy(lons[loni]))
    return outdata, outlats, outlons

def regrid_to_higher(data,lats,lons,newlats,newlons,interp='nearest',fill_value=np.NaN):
    '''
        regrid data[lats,lons] to data[newlats,newlnos] using interp method
        interp = {‘linear’, ‘nearest’, ‘cubic’}
            gridsquares outside the convex hull of input points are filled with fill_value
            (does not affect 'nearest' method)
    '''
    # make into higher resolution
    mlons,mlats = np.meshgrid(lons,lats)
    mnewlons,mnewlats = np.meshgrid(newlons,newlats)

    #https://docs.scipy.org/doc/scipy/reference/interpolate.html
    # take nearest datapoint from old to give value to new gridpoint value
    newdata = griddata( (mlats.ravel(), mlons.ravel()), data.ravel(),
                      (mnewlats, mnewlons), method=interp, fill_value=fill_value)
    return newdata

def regrid_to_lower(data, lats, lons, newlats, newlons, func=np.nanmean, pixels=None):
    '''
        Regrid data to lower resolution
        using midpoints of new and old grid
        optionally weight by pixel counts (at same resolution as data)
    '''
    start=timeit.default_timer()
    if __VERBOSE__:
        print('regridding from %s to %s'%(str(data.shape),str([len(newlats),len(newlons)])))

    # don't run if we don't have to, this function takes ages
    if np.all(lats == newlats) and np.all(lons==newlons):
        print('WARNING: regrid_to_lower called using identical grid..')
        return data

    newlats_e,newlons_e=edges_from_mids(newlats),edges_from_mids(newlons)
    ret=np.zeros([len(newlats_e)-1,len(newlons_e)-1])+np.NaN
    # Ignore numpy warnings... (they suck)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        for i in range(len(newlats_e)-1):
            for j in range(len(newlons_e)-1):
                lati= (lats >= newlats_e[i]) * (lats < newlats_e[i+1])
                loni= (lons >= newlons_e[j]) * (lons < newlons_e[j+1])

                # data set subsetted to new lat/lon bin
                tmp=data[lati,:]
                tmp=tmp[:,loni]
                # potentially pixels weighting the same subset
                if pixels is not None:
                    prebin=np.nanmean(tmp)
                    sub_pixels = pixels[lati,:]
                    sub_pixels = sub_pixels[:,loni]
                    n_pixels   = float(np.nansum(sub_pixels))

                    # check how many pixels are added in NaN squares...
                    #nanpix= np.nansum(sub_pixels[np.isnan(tmp)]) # how many pixels when value is nan (should be 0)
                    #assert nanpix==0, 'nan pixels: %d, (Should be zero)'%nanpix
                    #if __VERBOSE__:
                    #    if nanpix > 0:
                    #        print("WARNING:",nanpix,' pixels used to create np.NaN column...')


                    # Pixel weighted average is sum of entries / how many entries within this subregion
                    tmp= np.nansum( (tmp * sub_pixels) / n_pixels )

                    # Testing
                    #if __VERBOSE__:
                    #    if not np.isnan(prebin):
                    #        if prebin/tmp < 0.5 or prebin/tmp > 2:
                    #            print("Warning: weighted binning has more than halved or doubled the bin from %.2e (avg) to %.2e (weighted avg)"%(prebin,tmp))

                    #print('post_binning, positive subset mean: %.2e'%(np.nansum(pos*sub_pixels)/n_pixels_pos))
                    #print('post_binning, positive2 subset mean: %.2e'%(np.nansum(pos2*sub_pixels)/n_pixels))
                    #if not np.isnan(np.nanmean(tmp)):
                    #    print(lats[lati],lons[loni])
                    #    print(tmp)
                    #    print(pos)
                    #    print(sub_pixels)


                ret[i,j]=func(tmp)

    if __VERBOSE__:
        print("TIMEIT: it took %6.2f seconds to REGRID"%(timeit.default_timer()-start))
    return ret

def regrid_3d(data,lats,lons,newlats,newlons,interp='nearest', groupfunc=np.nanmean,max_procs=4):
    '''
    call regrid in parallel over time (first) dimension of data argument
    '''
    ntimes=data.shape[0]
    # turn input args into repeated list of same argument
    nlats=[lats]*ntimes
    nlons=[lons]*ntimes
    nnewlats=[newlats]*ntimes
    nnewlons=[newlons]*ntimes
    ninterp=[interp]*ntimes
    ngroupfunc=[groupfunc]*ntimes
    ndata = [data[i] for i in range(ntimes)]

    ret=np.zeros([ntimes,len(newlats),len(newlons)])

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_procs) as executor:
        procreturns=executor.map(regrid,ndata,nlats,nlons,nnewlats,nnewlons,ninterp,ngroupfunc)
        # loop over returned dictionaries from read_omno2d()
        for ii, pret in enumerate(procreturns):
            ret[ii] = pret

    return ret

def regrid(data,lats,lons,newlats,newlons, interp='nearest', groupfunc=np.nanmean):
    '''
    Regrid a data array [lat,lon] onto [newlat,newlon]
    This function estimates if you want higher or lower resolution based on number of lats/lons
    When making resolution finer, need to interpolate:
        interp = 'nearest' | 'linear' | 'cubic'
    When making resolution lower, need to group using some function
        groupfunc = np.nanmean by default

    '''
    if __VERBOSE__:
        print("utilities.regrid transforming %s to %s"%(str((len(lats),len(lons))),str((len(newlats),len(newlons)))))
        print("data input looks like %s"%str(np.shape(data)))
    newdata=None
    # if no resolution change then just throw back input
    oldy=lats[2]-lats[1]
    newy=newlats[2]-newlats[1]
    oldx=lons[2]-lons[1]
    newx=newlons[2]-newlons[1]
    if oldx==newx and oldy==newy:
        return data
    # If new resolution is strictly finer, use griddata
    elif newx < oldx and newy < oldy:
        newdata = regrid_to_higher(data,lats,lons,newlats,newlons,interp=interp)
    # if new res is not strictly finer
    else:
        newdata = regrid_to_lower(data,lats,lons,newlats,newlons,func=groupfunc)

    # Check shape is as requested
    assert np.shape(newdata)== (len(newlats),len(newlons)), "Regridded shape new lats/lons!"

    return newdata

def remote_pacific_background(data, lats, lons,
                              average_lons=False, average_lats=False,
                              has_time_dim=False, pixels=None):
    '''
        Get remote pacific ocean background from data array
        Can average the lats and lons if desired
        can weight by pixel counts if provided.

        Returns: rp, bglats, bglons
            rp: remote pacific subset from data input
            bglats: latitudes in rp
            bglons: longitudes in rp

    '''
    # First pull out region in the remote pacific
    # Use the lats from input data
    remote_bg_region=[lats[0],__REMOTEPACIFIC__[1],lats[-1],__REMOTEPACIFIC__[3]]
    if pixels is not None:
        subset=lat_lon_subset(lats, lons, remote_bg_region, [data,pixels],has_time_dim=has_time_dim)
    else:
        subset=lat_lon_subset(lats, lons, remote_bg_region, [data],has_time_dim=has_time_dim)
    rp=subset['data'][0]
    bglats=subset['lats']
    bglons=subset['lons']

    htd=[0,1][has_time_dim] # convert bool to int
    # If we're weighting the average by pixel counts
    if pixels is not None:
        subpix=subset['data'][1]
        rp = rp*subpix
        with np.errstate(invalid='ignore'): # ignore divide by zero
            if average_lons:
                rp=np.nansum(rp,axis=1+htd) / np.nansum(subpix,axis=1+htd)
            if average_lats:
                rp=np.nansum(rp,axis=0+htd) / np.nansum(subpix,axis=0+htd)
    else:
        if average_lons:
            rp=np.nanmean(rp,axis=1+htd)
        if average_lats:
            rp=np.nanmean(rp,axis=0+htd)


    bglons = [bglons, np.nanmean(bglons)][average_lons]
    bglats = [bglats, np.nanmean(bglats)][average_lats]

    return rp, bglats, bglons


def reshape_time_lat_lon_lev(data,ntimes,nlats,nlons,nlevs):
    ''' return reference to data array with time,lat,lon,lev dims '''

    # grab array and shape
    shp=np.array(data.shape)
    n_dims=len(shp)

    if __VERBOSE__:
        print("changing shape:",shp,'->',ntimes,nlats,nlons,nlevs)

    if ntimes is None:
        ntimes = -1
    if nlevs is None:
        nlevs = -2

    # make sure data is not square
    ti = None
    if len(set([ntimes,nlats,nlons,nlevs])) < 4:
        # can assume first dimension is time if it's the one with = len
        if (len(set([nlats,nlons,nlevs])) < 3) or ( ntimes != shp[0] ):
            print("ERROR: could not reshape data array")
            print(shp,'->',ntimes,nlats,nlons,nlevs)
            assert False, 'reshaping failed'
            return data
        else:
            shp[0]=-5
            ti=np.array([[0,],])

    # reshape automatically
    if n_dims>1:
        lati=np.argwhere(shp==nlats)[0,0]
        loni=np.argwhere(shp==nlons)[0,0]
        newshape=(lati,loni)

        # do we have time and level dimensions?
        if ti is None:
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

def resample(data,dates, bins='M', **resampleargs):
    '''
        Resample time series using pandas
        bin examples: 'M' = monthly, 'D'= daily, '3T'= 3 minutes, '30S' = 30 seconds,
        Seasonal bins: 'Q-NOV' # indicates quaterly with year ending in Nov
        other args:
            closed='right'  : closed intervals
            label='left'    : bin labels (default left)
            loffset='5s'    : bin label offset

    '''
    series = pd.Series(data,index=dates)
    resampled = series.resample(bins, **resampleargs)
    #newdates = resampled.mean().index.to_pydatetime()
    #mean=resampled.mean()
    #median=resampled.median()
    return resampled
     

def satellite_mean(data,pixels, spatial=True, temporal=False):
    '''
        Take satellite mean and pixel count and get average of all pixels
            can be spatial and or temporal average
    '''
    out = np.copy(data)
    total = out * pixels

    # only want to average over spatial dimensions
    has_time_dim = len(out.shape) > 2
    spatial_dims = [(0,1),(1,2)][has_time_dim]
    temporal_dims= [None, 0][has_time_dim]

    out[pixels<1] = np.NaN

    # which dims do we want to average over?
    axes=None
    # maybe just want spatial average
    if spatial and (not temporal):
        axes=spatial_dims
    # maybe want just temporal
    elif temporal and (not spatial):
        assert has_time_dim, 'Averaging temporally array with no time dim'
        axes=temporal_dims
    elif (not spatial) and (not temporal):
        assert False, 'Don\'t call averaging function with all false flags'


    # ignore division by nan, or zero
    with np.errstate(invalid='ignore'):
        count = np.nansum(pixels,axis=axes)
        out = np.nansum(total, axis=axes) / count
        out[np.isinf(out)] = np.NaN

    return out, count

def set_adjacent_to_true(mask):
    nx,ny=mask.shape
    mask=mask.astype(bool) # zero is false, nonzero is true
    mask_copy = np.zeros([nx,ny]).astype(bool)


    for y in range(ny):
        # copy row shifted both left and right
        mask_copy[:,y]   += mask[:,y]
        mask_copy[:-1,y] += mask[1:,y] # shifted right
        mask_copy[1:,y]  += mask[:-1,y] # shifted left
        # left and right edges
        mask_copy[0,y]   += mask[1,y] + mask[0,y]
        mask_copy[-1,y]  += mask[-2,y]+ mask[-1,y]

        # also copy row above and below in the same way
        if y>0:
            mask_copy[:,y] += mask[:,y-1]
            mask_copy[:-1,y] += mask[1:,y-1] # shifted right
            mask_copy[1:,y] += mask[:-1,y-1] # shifted left
            mask_copy[0,y]   += mask[1,y-1]  + mask[0,y-1]
            mask_copy[-1,y]  += mask[-2,y-1] + mask[-1,y-1]
        if y<ny-1:
            mask_copy[:,y] += mask[:,y+1]
            mask_copy[:-1,y] += mask[1:,y+1] # shifted right
            mask_copy[1:,y] += mask[:-1,y+1] # shifted left
            mask_copy[0,y]   += mask[1,y+1] + mask[0,y+1]
            mask_copy[-1,y]  += mask[-2,y+1]+ mask[-1,y+1]

    return mask_copy

def trend(data,dates,resample_monthly=True, remove_mya=True, remove_outliers=False):
    '''
        Take a time series, remove the multi year monthly average, determine trend
            Also test significance of trend using students t test
        
    '''
    monthly=np.copy(data)
    months=list_months(dates[0],dates[-1])
    
    # resample if necessary    
    if resample_monthly:
        monthly = np.array(resample(monthly,dates,bins='M').mean()).squeeze()
    
    # first detrend if necessary
    mya=None
    anomaly=np.copy(monthly)
    if remove_mya:
        # use original data to get multi-year average
        mya_df = multi_year_average(data, dates, grain='monthly')
        mya = np.squeeze(np.array(mya_df.mean()))
        anomaly = np.array([ monthly[i] - mya[i%12] for i in range(len(months)) ])
    
    outliers=None
    X=np.arange(len(months))
    Y=anomaly
    if remove_outliers:
        
        std=np.nanstd(anomaly)
        mean=np.nanmean(anomaly)
        outliers = ( anomaly > mean + 3 * std ) + ( anomaly < mean - 3*std )
        X=X[~outliers]
        Y=Y[~outliers]
    # USE OLS FOR NOW....
    slope, intercept, r, p, sterr = JesseRegression.OLS(X,Y)
    
    # TEST REGRESSION WAS SIGNIFICANT
    
    return {'anomaly':anomaly, 'monthly':monthly, 
            'mya':mya, 'outliers':outliers,
            'slope':slope, 'intercept':intercept,
            'r':r, 'p':p, 'sterr':sterr}


def utc_to_local(utc_dt):
    '''
        Convert utc time tom local time (of computer I guess?)
    '''
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
