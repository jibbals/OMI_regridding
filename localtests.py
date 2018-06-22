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
import tests
import reprocess

from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.omhchorp import omhchorp

from classes.campaign import campaign
import xbpch
import xarray
import pandas as pd

import timeit

###############
### Globals ###
###############
__VERBOSE__=True
region=pp.__AUSREGION__

#####
## SETUP STUFFS
#####

d0=datetime(2005,1,1)
dstr=d0.strftime('%Y%m%d')
mstr=d0.strftime('%Y%m')
latres=0.25
lonres=0.3125
dN=datetime(2005,1,5)
d3=datetime(2005,3,1)
dates=util.list_days(d0,dN,month=False)
# start timer
start1=timeit.default_timer()

##########
### DO STUFFS
##########

#def store_emissions_month(month=datetime(2005,1,1), GCB=None, OMHCHORP=None,
#                          region=pp.__AUSREGION__):
'''
    Store a month of new emissions estimates into an he5 file
    TODO: Add monthly option to just store month averages and month emissions
'''
month=datetime(2005,1,1)
GCB=None
OMHCHORP=None
#                          region=pp.__AUSREGION__
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
if OMHCHORP is None:
    OMHCHORP=omhchorp(day0=day0,dayn=dayn, ignorePP=False)
if GCB is None:
    GCB=GC_class.GC_biogenic(day0,) # data like [time,lat,lon,lev]

# subset our lats/lons
# Arrays to be subset
arrs_names=['VCC_OMI','VCC_GC','VCC_PP',
            'firemask','smokemask','anthromask',
            'gridentries','ppentries','col_uncertainty_OMI',
            ]
# list indices
arrs_i={s:i for i,s in enumerate(arrs_names)}
# data from OMHCHORP
arrs=[getattr(OMHCHORP,s) for s in arrs_names]

OMHsubsets=util.lat_lon_subset(OMHCHORP.lats,OMHCHORP.lons,region,data=arrs, has_time_dim=True)
omilats=OMHsubsets['lats']
omilons=OMHsubsets['lons']
omilati=OMHsubsets['lati']
# map subsetted arrays into another dictionary
OMHsub = {s:OMHsubsets['data'][arrs_i[s]] for s in arrs_names}

# Need Vertical colums, slope, and backgrounds all at same resolution to get emissions
VCC_GC                = OMHsub['VCC_GC']
VCC_PP                = OMHsub['VCC_PP']
VCC_OMI               = OMHsub['VCC_OMI']
pixels                = OMHsub['gridentries']
pixels_PP             = OMHsub['ppentries']
uncert                = OMHsub['col_uncertainty_OMI']
firefilter            = OMHsub['firemask']+OMHsub['smokemask']
anthrofilter          = OMHsub['anthromask']

# lets have a look at these things
if True:
    vmin=1e14
    vmax=4e16
    f=plt.figure(figsize=(14,14))
    plt.subplot(221)
    pp.createmap(VCC_GC[0],omilats,omilons, title='GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(222)
    pp.createmap(VCC_OMI[0],omilats,omilons, title='OMI', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(223)
    pp.createmap(np.nanmean(VCC_GC,axis=0),omilats,omilons, title='mean_GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(224)
    pp.createmap(np.nanmean(VCC_OMI,axis=0),omilats,omilons, title='mean_OMI', region=region,vmin=vmin,vmax=vmax)
    plt.savefig('temp_VCC_check.png')
    plt.close()

# GC.model_slope gets slope and subsets the region
# Then Map slope onto higher omhchorp resolution:
slope_dict=GCB.model_slope(region=region)
GC_slope=slope_dict['slope']
gclats,gclons = slope_dict['lats'],slope_dict['lons']
GC_slope = util.regrid_to_higher(GC_slope,gclats,gclons,omilats,omilons,interp='nearest')
if True:
    plt.figure()
    vmin=1e3;vmax=2e4
    pp.createmap(GC_slope,omilats,omilons,region=region, cbarfmt='%.0f', cbarxtickrot=20,
                 linear=True,vmin=vmin,vmax=vmax,title='GC_slope')
    plt.savefig('temp_slope_check.png')
    plt.close()

# Also save smearing
smear, slats,slons = Inversion.smearing(month,region=region)
pp.createmap(smear,slats,slons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
             linear=True, vmin=1000, vmax=10000,
             clabel='S', pname='Figs/GC/smearing_%s.png'%mstr, title='Smearing %s'%mstr)
smear = util.regrid_to_higher(smear,slats,slons,omilats,omilons,interp='nearest')
pp.createmap(smear,omilats,omilons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
             linear=True, vmin=1000, vmax=10000,
             clabel='S', pname='Figs/GC/smearing_%s_interp.png'%mstr, title='Smearing %s'%mstr)
print("Smearing plots saved in Figs/GC/smearing...")
outdata['smearing'] = smear
outattrs['smearing']= {'desc':'smearing = Delta(HCHO)/Delta(E_isop), where Delta is the difference between full and half isoprene emission runs from GEOS-Chem for %s, interpolated linearly from 2x2.5 to 0.25x0.3125 resolution'%mstr}

# TODO: Smearing Filter
smearfilter = smear > Inversion.__Thresh_Smearing__#5000 # something like this


# emissions using different columns as basis
# Fully filtered
out_shape=VCC_GC.shape
E_gc        = np.zeros(out_shape) + np.NaN
E_pp        = np.zeros(out_shape) + np.NaN
E_omi       = np.zeros(out_shape) + np.NaN

# unfiltered:
E_gc_u      = np.zeros(out_shape) + np.NaN
E_pp_u      = np.zeros(out_shape) + np.NaN
E_omi_u     = np.zeros(out_shape) + np.NaN

BG_VCC      = np.zeros(out_shape) + np.NaN
BG_PP       = np.zeros(out_shape) + np.NaN
BG_OMI      = np.zeros(out_shape) + np.NaN

# Need background values from remote pacific
BG_VCCa, bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_GC,
                                                        OMHCHORP.lats, OMHCHORP.lons,
                                                        average_lons=True,has_time_dim=True)
BG_PPa , bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_PP,
                                                        OMHCHORP.lats, OMHCHORP.lons,
                                                        average_lons=True,has_time_dim=True)
BG_OMIa, bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_OMI,
                                                        OMHCHORP.lats, OMHCHORP.lons,
                                                        average_lons=True,has_time_dim=True)

if True:
    vmin=1e13
    vmax=1e15
    lats=OMHCHORP.lats
    f=plt.figure(figsize=(14,14))
    plt.plot(lats,BG_VCCa[0],label='BG_GC[0]')
    plt.plot(lats,BG_OMIa[0],label='BG_OMI[0]')
    plt.plot(lats,np.nanmean(BG_VCCa,axis=0),label='mean BG_GC')
    plt.plot(lats,np.nanmean(BG_OMIa,axis=0),label='mean BG_OMI')
    plt.legend(loc='best')
    plt.xlabel('latitude')
    plt.ylabel('VCC [molec/cm2]')
    plt.savefig('temp_BG_check.png')
    plt.close()


for i,day in enumerate(days):

    BG_VCCi = BG_VCCa[i]
    BG_PPi  = BG_PPa[i]
    BG_OMIi = BG_OMIa[i]

    # can check that reshaping makes sense with:
    #bgcolumn=np.copy(BG_VCCi)
    #BG_VCCi = BG_VCCi.repeat(len(omilons)).reshape([len(omilats),len(omilons)])
    # check all values in column are either equal or both nan
    #assert all( (bgcolumn == BG_VCCi[:,0]) + (np.isnan(bgcolumn) * np.isnan(BG_VCCi[:,0])))

    # we only want the subset of background values matching our region
    BG_VCCi = BG_VCCi[omilati]
    BG_PPi  = BG_PPi[omilati]
    BG_OMIi = BG_OMIi[omilati]

    plt.plot(omilats,BG_VCCi,'r')
    plt.plot(omilats,BG_OMIi,'m')

    # The backgrounds need to be the same shape so we can subtract from whole array at once.
    # done by repeating the BG values ([lats]) N times, then reshaping to [lats,N]
    BG_VCCi = BG_VCCi.repeat(len(omilons)).reshape([len(omilats),len(omilons)])
    BG_PPi  = BG_PPi.repeat(len(omilons)).reshape([len(omilats),len(omilons)])
    BG_OMIi = BG_OMIi.repeat(len(omilons)).reshape([len(omilats),len(omilons)])


    # Store the backgrounds for later analysis
    BG_VCC[i,:,:] = BG_VCCi
    BG_PP[i,:,:]  = BG_PPi
    BG_OMI[i,:,:] = BG_OMIi

    # Run calculation with no filters applied:
    E_gc_u[i,:,:]       = (VCC_GC[i] - BG_VCCi) / GC_slope
    E_pp_u[i,:,:]       = (VCC_PP[i] - BG_PPi) / GC_slope
    E_omi_u[i,:,:]      = (VCC_OMI[i] - BG_OMIi) / GC_slope
    if i == 0:
        print("UNFILTERED:")
        print(np.nanmean(E_omi_u))

    # run with filters
    # apply filters
    allmasks            = firefilter[i] + anthrofilter[i] # + smearfilter
    if True:
        print("ALLMASKS")
        print(allmasks)
        print(np.sum(allmasks))
        print(np.nansum(VCC_GC[i][allmasks]))
        exit(0)
    vcc_gci             = np.copy(VCC_GC[i])
    vcc_gci[allmasks]   = np.NaN
    vcc_ppi             = np.copy(VCC_PP[i])
    vcc_ppi[allmasks]   = np.NaN
    vcc_omii            = np.copy(VCC_OMI[i])
    vcc_omii[allmasks]  = np.NaN
    plt.plot(allmasks)
    # estimate emissions
    E_gc[i,:,:]         = (vcc_gci - BG_VCCi) / GC_slope
    E_pp[i,:,:]         = (vcc_ppi - BG_PPi) / GC_slope
    E_omi[i,:,:]        = (vcc_omii - BG_OMIi) / GC_slope
    if i == 0:
        print("FILTERED")
        print(np.nanmean(E_omi))

if True:
    vmin=0
    vmax=1
    f=plt.figure(figsize=(14,14))
    plt.subplot(221)
    pp.createmap(firefilter[0],omilats,omilons,vmin=vmin,vmax=vmax,
                 region=region,linear=True,title='firefilter[0]')
    plt.subplot(222)
    pp.createmap(anthrofilter[0],omilats,omilons,vmin=vmin,vmax=vmax,
                 region=region,linear=True,title='anthrofilter[0]')
    plt.subplot(223)
    pp.createmap(np.nanmean(firefilter,axis=0),omilats,omilons,vmin=vmin,vmax=vmax,
                 region=region,linear=True,title='mean firefilter')
    plt.subplot(224)
    pp.createmap(np.nanmean(anthrofilter,axis=0),omilats,omilons,vmin=vmin,vmax=vmax,
                 region=region,linear=True,title='mean anthrofilter')
    plt.savefig('temp_Filters.png')
    plt.close()

if True:
    vmin=1e10
    vmax=1e13
    f=plt.figure(figsize=(14,14))
    plt.subplot(221)
    pp.createmap(E_gc_u[0], omilats,omilons, title='E_GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(222)
    pp.createmap(E_omi_u[0], omilats,omilons, title='E_omi', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(223)
    pp.createmap(np.nanmean(E_gc_u,axis=0),omilats,omilons, title='mean_E_GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(224)
    pp.createmap(np.nanmean(E_omi_u,axis=0),omilats,omilons, title='mean_E_OMI', region=region,vmin=vmin,vmax=vmax)
    plt.suptitle('unfiltered')
    plt.savefig('temp_E_unfiltered_check.png')
    plt.close()

    f=plt.figure(figsize=(14,14))
    plt.subplot(221)
    pp.createmap(E_gc[0], omilats,omilons, title='E_GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(222)
    pp.createmap(E_omi[0], omilats,omilons, title='E_omi', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(223)
    pp.createmap(np.nanmean(E_gc,axis=0),omilats,omilons, title='mean_E_GC', region=region,vmin=vmin,vmax=vmax)
    plt.subplot(224)
    pp.createmap(np.nanmean(E_omi,axis=0),omilats,omilons, title='mean_E_OMI', region=region,vmin=vmin,vmax=vmax)
    plt.suptitle('filtered')
    plt.savefig('temp_E_filtered_check.png')
    plt.close()

# Lets save both monthly averages and the daily amounts
#

# Save the backgrounds, as well as units/descriptions
outdata['BG_VCC']    = BG_VCC
outdata['BG_PP']     = BG_PP
outdata['BG_OMI']    = BG_OMI
outattrs['BG_VCC']   = {'units':'molec/cm2','desc':'Background: VCC zonally averaged from remote pacific'}
outattrs['BG_PP']    = {'units':'molec/cm2','desc':'Background: VCC_PP zonally averaged from remote pacific'}
outattrs['BG_OMI']   = {'units':'molec/cm2','desc':'Background: VCC_OMI zonally averaged from remote pacific'}

# Save the Vertical columns, as well as units/descriptions
outdata['VCC_GC']     = VCC_GC
outdata['VCC_PP']     = VCC_PP
outdata['VCC_OMI']    = VCC_OMI
outattrs['VCC_GC']    = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column using recalculated shape factor, fire and anthro masked'}
outattrs['VCC_PP']    = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column using PP code, fire and anthro masked'}
outattrs['VCC_OMI']   = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column, fire and anthro masked'}

# Save the Emissions estimates, as well as units/descriptions
outdata['E_VCC_GC']     = E_gc
outdata['E_VCC_PP']     = E_pp
outdata['E_VCC_OMI']    = E_omi
outdata['E_VCC_GC_u']   = E_gc_u
outdata['E_VCC_PP_u']   = E_pp_u
outdata['E_VCC_OMI_u']  = E_omi_u
if True:
    print("______EMISSIONS_____")
    print("Filtered  ,    Unfiltered")
    _blah = [print("%.2e  ,  %.2e  "%(np.nanmean(E),np.nanmean(E_u))) for E,E_u in zip([E_omi,E_gc,E_pp],[E_omi_u,E_gc_u,E_pp_u])]

outattrs['E_VCC_GC']    = {'units':'molec OR atom C???/cm2/s',
                           'desc':'Isoprene Emissions based on VCC and GC_slope'}
outattrs['E_VCC_PP']    = {'units':'molec OR atom C??/cm2/s',
                           'desc':'Isoprene Emissions based on VCC_PP and GC_slope'}
outattrs['E_VCC_OMI']   = {'units':'molec OR/cm2/s',
                           'desc':'Isoprene emissions based on VCC_OMI and GC_slope'}
outattrs['E_VCC_GC_u']  = {'units':'molec OR atom C???/cm2/s',
                           'desc':'Isoprene Emissions based on VCC and GC_slope, unmasked by fire or anthro'}
outattrs['E_VCC_PP_u']  = {'units':'molec OR atom C??/cm2/s',
                           'desc':'Isoprene Emissions based on VCC_PP and GC_slope, unmasked by fire or anthro'}
outattrs['E_VCC_OMI_u'] = {'units':'molec OR/cm2/s',
                           'desc':'Isoprene emissions based on VCC_OMI and GC_slope, unmasked by fire or anthro'}

# Extras like pixel counts etc..
outdata['firefilter']   = firefilter.astype(np.int)
outdata['anthrofilter'] = anthrofilter.astype(np.int)
outdata['smearfilter']  = smearfilter.astype(np.int)
outdata['pixels']       = pixels
outdata['pixels_PP']    = pixels_PP
outdata['uncert_OMI']   = uncert
outattrs['firefilter']  = {'units':'N/A',
                           'desc':'Squares with more than one fire (over today or last two days, in any adjacent square) or AAOD greater than %.1f'%(fio.__Thresh_AAOD__)}
outattrs['anthrofilter']= {'units':'N/A',
                           'desc':'Squares with tropNO2 from OMI greater than %.1e or yearly averaged tropNO2 greater than %.1e'%(fio.__Thresh_NO2_d__,fio.__Thresh_NO2_y__)}
outattrs['smearfilter'] = {'units':'N/A',
                           'desc':'Squares where smearing greater than %.1f'%(Inversion.__Thresh_Smearing__)}
outattrs['uncert_OMI']  = {'units':'?? molec/cm2 ??',
                           'desc':'OMI pixel uncertainty averaged for each gridsquare'}
outattrs['pixels']      = {'units':'n',
                           'desc':'OMI pixels used for gridsquare VC'}
outattrs['pixels_PP']   = {'units':'n',
                           'desc':'OMI pixels after PP code used for gridsquare VC'}

# Adding time dimension (needs to be utf8 for h5 files)
#dates = np.array([d.strftime("%Y%m%d").encode('utf8') for d in days])
dates = np.array([int(d.strftime("%Y%m%d")) for d in days])
outdata["time"]=dates
outattrs["time"]={"format":"%Y%m%d", "desc":"year month day as integer (YYYYMMDD)"}
fattrs={'region':"SWNE: %s"%str(region)}
fattrs['date range']="%s to %s"%(d0str,dnstr)

# Save lat,lon
outdata['lats']=omilats
outdata['lons']=omilons
outdata['lats_e']=util.edges_from_mids(outdata['lats'])
outdata['lons_e']=util.edges_from_mids(outdata['lons'])


# Save file, with attributes
print("NORMALLY WOULD SAVE NOW BUT THS IS TEST ENV")
#fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)
if __VERBOSE__:
    print("%s should now be saved"%fname)

#plt.pcolormesh(tracks,lats,RSC,cmap='plasma',vmin=-1e16,vmax=1e16)
#cb=plt.colorbar()

###########
### Record and time STUJFFS
###########

end=timeit.default_timer()
print("TIME: %6.2f minutes for stuff"%((end-start1)/60.0))

#plt.savefig('Figs/GC/interp.png')
#plt.close()


def emisssions_vs_firefilter(d0=datetime(2005,1,1)):
    '''
    '''
    Enew=E_new(d0,)
    # mean Emissions estimates vs eachother

    f,axes = plt.subplots(3,3,figsize=(15,15))
    region=[Enew.lats[0],Enew.lons[0],Enew.lats[-1],Enew.lons[-1]]

    lats=Enew.lats
    lons=Enew.lons
    arrs=[[Enew.E_VCC_OMI, Enew.E_VCC, Enew.E_VCC_PP],
          [Enew.E_VCC_OMI_f, Enew.E_VCC_f, Enew.E_VCC_PP_f],
          []]

    labels=[['OMI','GC','PP'],
            ['OMI_f','GC_f','PP_f'],
            ['fire-nofire','fire-nofire','fire-nofire']]

    linear=False
    vmin=1e10
    vmax=1e13
    for i in range(3):
        for j in range(3):
            plt.sca(axes[i,j])
            if i < 2:
                arr=np.nanmean(arrs[i][j], axis=0) # average over time
            elif i==2:
                arr=np.nanmean(arrs[1][j], axis=0) - np.nanmean(arrs[0][j], axis=0)
                vmin=-5e12; vmax=5e12; linear=True

            pp.createmap(arr,lats,lons,title=labels[i][j],
                         region=region,
                         vmin=vmin,vmax=vmax,linear=linear)

    pname='Figs/Emiss/FireFilter%s.png'%mstr
    plt.suptitle('Emissions with and without fire filter %s'%mstr,fontsize=25)
    plt.savefig(pname)
    plt.close()



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
