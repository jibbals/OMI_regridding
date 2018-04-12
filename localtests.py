#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO,GC_fio,fio
import Inversion


from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.omhchorp import omhchorp
from classes.gchcho import gchcho
from classes.campaign import campaign
import xbpch
import xarray
import pandas as pd

import timeit

###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)


month=datetime(2005,1,1)
GC=None
OMI=None
region=pp.__AUSREGION__
'''
    Store a month of new emissions estimates into an he5 file
    TODO: Add monthly option to just store month averages and month emissions
'''
# Dates required: day0, dayN, and list of days between
day0=util.first_day(month)
dayn=util.last_day(day0)
days=util.list_days(day0,dayn)

# Handy date strings
mstr=dayn.strftime('%Y%m')
d0str=day0.strftime("%Y%m%d")
dnstr=dayn.strftime("%Y%m%d")

# File location to write to
ddir="Data/Isop/E_new/"
fname=ddir+"emissions_%s.h5"%(mstr)

if __VERBOSE__:
    print("Calculating %s-%s estimated emissions over %s"%(d0str,dnstr,str(region)))
    print("will save to file %s"%(fname))

# Dicts which will be saved:
outdata={}
outattrs={}


# Read omhchorp VCs, AMFs, Fires, Smoke, etc...
if OMI is None:
    OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=False)
if GC is None:
    GC=GC_class.GC_biogenic(day0,) # data like [time,lat,lon,lev]

# subset our lats/lons
omilats=OMI.lats
omilons=OMI.lons
omilati, omiloni = util.lat_lon_range(omilats,omilons,region=region)
newlats=omilats[omilati]
newlons=omilons[omiloni]

# We need to make the fire and smoke masks:
firemask=OMI.make_fire_mask(d0=day0, dN=dayn, days_masked=8,
                            fire_thresh=1, adjacent=True)
smokemask=OMI.make_smoke_mask(d0=day0,dN=dayn, aaod_thresh=0.2)
firefilter=(firemask+smokemask).astype(np.bool)


# Need Vertical colums, slope, and backgrounds all at same resolution to get emissions
VCC                   = np.copy(OMI.VCC)
VCC_PP                = np.copy(OMI.VCC_PP)
VCC_OMI               = np.copy(OMI.VC_OMI_RSC)
pixels                = np.copy(OMI.gridentries)
pixels_PP             = np.copy(OMI.ppentries)
SArea                 = np.copy(OMI.surface_areas)
#VCC_orig              = np.copy(OMI.VCC) # to see the effect of the fire mask do some without

# Remove gridsquares affected by Fire or Smoke
VCC[firefilter]       = np.NaN
VCC_PP[firefilter]    = np.NaN
VCC_OMI[firefilter]   = np.NaN
pixels[firefilter]    = 0
pixels_PP[firefilter] = 0

# GC.model_slope gets slope and subsets the region
# Then Map slope onto higher omhchorp resolution:
slope_dict=GC.model_slope(region=region)
GC_slope=slope_dict['slope']
gclats,gclons = slope_dict['lats'],slope_dict['lons']
GC_slope = util.regrid_to_higher(GC_slope,gclats,gclons,omilats,omilons,interp='nearest')

# Subset our arrays to desired region!
GC_slope    = GC_slope[omilati,:]
GC_slope    = GC_slope[:,omiloni]
VCC         = VCC[:,omilati,:]
VCC         = VCC[:,:,omiloni]
VCC_PP      = VCC_PP[:,omilati,:]
VCC_PP      = VCC_PP[:,:,omiloni]
VCC_OMI     = VCC_OMI[:,omilati,:]
VCC_OMI     = VCC_OMI[:,:,omiloni]
SArea       = SArea[omilati,:]
SArea       = SArea[:,omiloni]
pixels      = pixels[:,omilati,:]
pixels      = pixels[:,:,omiloni]
pixels_PP   = pixels_PP[:,omilati,:]
pixels_PP   = pixels_PP[:,:,omiloni]

# emissions using different columns as basis
E_vcc       = np.zeros(VCC.shape) + np.NaN
E_pp        = np.zeros(VCC.shape) + np.NaN
E_omi       = np.zeros(VCC.shape) + np.NaN
BG_VCC      = np.zeros(VCC.shape) + np.NaN
BG_PP       = np.zeros(VCC.shape) + np.NaN
BG_OMI      = np.zeros(VCC.shape) + np.NaN

time_emiss_calc=timeit.default_timer()
#for i,day in enumerate(days):
for i in [0,]:

    # Need background values from remote pacific
    BG_VCCi,bglats,bglons = util.remote_pacific_background(OMI.VCC[i], omilats, omilons, average_lons=True)
    BG_PPi ,bglats,bglons = util.remote_pacific_background(OMI.VCC_PP[i], omilats, omilons, average_lons=True)
    BG_OMIi,bglats,bglons = util.remote_pacific_background(OMI.VC_OMI_RSC[i], omilats, omilons, average_lons=True)

    # can check that reshaping makes sense with:
    #bgcolumn=np.copy(BG_VCCi)
    #BG_VCCi = BG_VCCi.repeat(len(omilons)).reshape([len(omilats),len(omilons)])
    # check all values in column are either equal or both nan
    #assert all( (bgcolumn == BG_VCCi[:,0]) + (np.isnan(bgcolumn) * np.isnan(BG_VCCi[:,0])))

    # we only want the subset of background values matching our region
    BG_VCCi = BG_VCCi[omilati]
    BG_PPi  = BG_PPi[omilati]
    BG_OMIi = BG_OMIi[omilati]

    # The backgrounds need to be the same shape so we can subtract from whole array at once.
    # done by repeating the BG values ([lats]) N times, then reshaping to [lats,N]
    BG_VCCi = BG_VCCi.repeat(len(newlons)).reshape([len(newlats),len(newlons)])
    BG_PPi  = BG_PPi.repeat(len(newlons)).reshape([len(newlats),len(newlons)])
    BG_OMIi = BG_OMIi.repeat(len(newlons)).reshape([len(newlats),len(newlons)])

    # Store the backgrounds for later analysis
    BG_VCC[i,:,:] = BG_VCCi
    BG_PP[i,:,:]  = BG_PPi
    BG_OMI[i,:,:] = BG_OMIi


    E_vcc[i,:,:]   = (VCC[i] - BG_VCCi) / GC_slope
    E_pp[i,:,:]    = (VCC_PP[i] - BG_PPi) / GC_slope
    E_omi[i,:,:]   = (VCC_OMI[i] - BG_OMIi) / GC_slope

elapsed = timeit.default_timer() - time_emiss_calc
print ("TIMEIT: Took %6.2f seconds to calculate backgrounds and estimate emissions()"%elapsed)

#Inversion.store_emissions_month(d0)

# to be moved somewhere:
def firetest():
    fires_per_area,lats,lons=fio.read_MOD14A1(d0,True)
    fires,lats,lons=fio.read_MOD14A1(d0,False)
    earth_sa=510e6 # 510.1 million km2
    count_a=np.sum(fires)
    count_b=np.mean(fires_per_area)*earth_sa*1e3
    print(count_a,count_b)
    print((count_a-count_b)/count_b)

    region=[-20,-30,40,50]
    f,axes=plt.subplots(3,1)
    plt.sca(axes[0])
    pp.createmap(fires,lats,lons,title='MODIS Fires 20050102',
                 colorbar=None, region=region,
                 linear=False, vmin=1,vmax=3e6)

    # lats lons are .1x.1 degrees
    # Try lower and higher resolution function:
    hlats,hlons,hlat_e,hlon_e = util.lat_lon_grid(latres=0.25,lonres=0.3125)
    llats,llons,llat_e,llon_e = util.lat_lon_grid(latres=2.0,lonres=2.5)

    hfires=util.regrid(fires,lats,lons,hlats,hlons,groupfunc=np.nansum)
    lfires=util.regrid(fires,lats,lons,llats,llons,groupfunc=np.nansum)
    print(np.nansum(hfires))
    print(np.nansum(lfires))
    plt.sca(axes[1])
    pp.createmap(hfires,hlats,hlons,title='High res',
                 colorbar=False,region=region,
                 linear=False, vmin=1,vmax=3e6)

    plt.sca(axes[2])
    pp.createmap(lfires,llats,llons,title='Low res', region=region,
                 clabel='fire pixels', pname='test_fires.png',
                 linear=False,cmapname='Reds',vmin=1,vmax=3e6)


#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')
