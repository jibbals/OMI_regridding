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
from tests import utilities_tests
import reprocess

from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.omhchorp import omhchorp

from utilities import masks

from classes.campaign import campaign
import xbpch
import xarray
import pandas as pd

import timeit


# parallelism
import concurrent.futures as confut


###############
### Globals ###
###############
__VERBOSE__=True
region=pp.__AUSREGION__

#####
## SETUP STUFFS
#####

d0=datetime(2005,1,1)
d1=datetime(2005,1,31)
dstr=d0.strftime('%Y%m%d')
mstr=d0.strftime('%Y%m')
latres=0.25
lonres=0.3125
dN=datetime(2005,12,31)
d3=datetime(2005,3,1)
dates=util.list_days(d0,dN,month=False)
# start timer
start1=timeit.default_timer()

##########
### DO STUFFS
##########

year=d0
prior_days_masked=2
fire_thresh=fio.__Thresh_fires__
adjacent=True
latres=fio.__LATRES__
lonres=fio.__LONRES__
max_procs=4
'''
    Create fire mask file for year, save into h5 file for re-use

    mask is true where more than fire_thresh fire pixels exist

'''

## First make year long filter using method above
#
d0=datetime(year.year,1,1)
dN=datetime(year.year,12,31)
#firemask,dates,lats,lons, fires = make_fire_mask(d0,dN,prior_days_masked=prior_days_masked,
#                                        fire_thresh=fire_thresh, adjacent=adjacent,
#                                        latres=latres,lonres=lonres,
#                                        region=None,max_procs=max_procs)

## to save an HDF we need to change boolean to int8 and dates to strings
#
firemask=np.zeros([365,5,5])
fires=np.zeros([365,5,5])
lats=np.arange(0,5,1)
lons=np.arange(0,5,1)
dates=util.list_days(d0,dN)
dates=util.gregorian_from_dates(dates)
firemask=firemask.astype(np.int8)

## add attributes to be saved in file
#
dattrs  = {'firemask':{'units':'int','desc':'0 or 1: grid square potentially affected by fire'},
          'fires':{'units':'int','desc':'sum of fire detected pixels'},
          'dates':{'units':'gregorian','desc':'hours since 1985,1,1,0,0: day axis of firemask array'},
          'lats':{'units':'degrees','desc':'latitude centres north (equator=0)'},
          'lons':{'units':'degrees','desc':'longitude centres east (gmt=0)'}, }

fattrs = {'fire_thresh':fire_thresh, 'adjacent':np.int8(adjacent), 'latres':latres, 'lonres':lonres,
          'prior_days_masked':prior_days_masked}
## data dictionary to save to hdf
#
datadict={'firemask':firemask,'fires':fires,'dates':dates,'lats':lats,'lons':lons}

# filename and save to h5 file
path=year.strftime(fio.__dir_fire__+'test_firemask_%Y.h5')
fio.save_to_hdf5(path, datadict, attrdicts=dattrs, fattrs=fattrs, verbose=True)

f,dates,lats,lons = fio.get_fire_mask(d0)

#data,attrs=fio.read_hdf5('Data/smearmask_2005.h5')
#print(data['dates'])
#print(util.date_from_gregorian(data['dates']))


#masks.make_smear_mask_file(2005)

###########
### Record and time STUJFFS
###########

end=timeit.default_timer()
print("TIME: %6.2f minutes for stuff"%((end-start1)/60.0))


def check_entries(d0=datetime(2005,1,1),d1=datetime(2005,1,31)):
    day=omhchorp(d0)
    month=omhchorp(d0,d1)

    inds_aus = day.inds_aus(maskocean=False)

    pp = day.data['AMF_PP'][inds_aus]
    gc = day.data['AMF_GC'][inds_aus]
    ppm = month.data['AMF_PP'][:,inds_aus]
    gcm = month.data['AMF_GC'][:,inds_aus]


    print(np.sum(~np.isnan(pp)),' good pp entries (1 day)')
    print(np.sum(~np.isnan(gc)),' good gc entries (1 day)')
    print(np.sum(~np.isnan(ppm)),' good pp entries (1 month)')
    print(np.sum(~np.isnan(gcm)),' good gc entries (1 month)')

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

def test_store_emissions_month(month=datetime(2005,1,1), GCB=None, OMHCHORP=None,
                              region=pp.__AUSREGION__):
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
    omiloni=OMHsubsets['loni']
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


    # cut the omhchorp backgrounds down to our latitudes
    BG_VCC = BG_VCCa[:,omilati]
    BG_PP = BG_PPa[:,omilati]
    BG_OMI = BG_OMIa[:,omilati]
    # Repeat them along our longitudes so we don't need to loop
    BG_VCC = np.repeat(BG_VCC[:,:,np.newaxis],len(omiloni),axis=2)
    BG_PP = np.repeat(BG_PP[:,:,np.newaxis],len(omiloni),axis=2)
    BG_OMI = np.repeat(BG_OMI[:,:,np.newaxis],len(omiloni),axis=2)
    # Also repeat Slope array along time axis to avoid looping
    GC_slope= np.repeat(GC_slope[np.newaxis,:,:], len(days),axis=0)



    if True:
        print("Enew Calc Shapes")
        print(VCC_GC.shape,BG_VCC.shape,GC_slope.shape)
        plt.plot(omilats,BG_VCC[0,:,0],'r',label='BG VCC')
        plt.plot(omilats,BG_OMI[0,:,0],'m', label='BG OMI')
        plt.legend()
        plt.title('Background over latitude')

    # Run calculation with no filters applied:
    E_gc_u       = (VCC_GC - BG_VCC) / GC_slope
    E_pp_u       = (VCC_PP - BG_PP) / GC_slope
    E_omi_u      = (VCC_OMI - BG_OMI) / GC_slope

    # run with filters
    # apply filters
    allmasks            = (firefilter + anthrofilter)>0 # + smearfilter
    print('FILTER DEETS')
    print(allmasks.shape)
    print(E_gc.shape)
    print(np.nansum(allmasks))
    print(np.nanmean(E_gc_u),np.nanmean(E_gc_u[allmasks]))


    assert not np.isnan(np.nansum(E_gc_u[allmasks])), 'Filtering nothing!?'

    # Mask gridsquares using fire and anthro filters
    E_gc                = np.copy(E_gc_u)
    E_pp                = np.copy(E_pp_u)
    E_omi               = np.copy(E_pp_u)
    E_gc[allmasks]      = np.NaN
    E_pp[allmasks]      = np.NaN
    E_omi[allmasks]     = np.NaN
    if True:
        # lets have a close look at these things
        vmin=1e10
        vmax=1e13
        f=plt.figure(figsize=(14,14))
        plt.subplot(211)
        # Plot E_new before and after filtering
        m,cs,cb=pp.createmap(E_gc_u[0],omilats,omilons, title='E_GC_u', region=[-40,130,-20,155],
                             vmin=vmin,vmax=vmax)

        # plot dots where filter should be
        for yi,y in enumerate(omilats):
            for xi,x in enumerate(omilons):
                if firefilter[0,yi,xi]:
                   mx,my = m(x,y)
                   m.plot(mx,my,'x',markersize=6,color='k')
                if anthrofilter[0,yi,xi]:
                   mx,my = m(x,y)
                   m.plot(mx,my,'d',markersize=3,color='k')
        plt.subplot(212)
        m,cs,cb=pp.createmap(E_gc[0],omilats,omilons, title='E_GC', region=[-40,130,-20,155],
                             vmin=vmin,vmax=vmax)

        pname='temp_E_filtering_check.png'
        plt.savefig(pname)
        print('saved ',pname)
        plt.close()

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
        _blah = [print("%.5e  ,  %.5e "%(np.nanmean(E),np.nanmean(E_u))) for E,E_u in zip([E_omi,E_gc,E_pp],[E_omi_u,E_gc_u,E_pp_u])]

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
