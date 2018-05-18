#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 09:48:11 2016

    Functions to creating the isoprene emissions estimates from my HCHO product

    Invert formaldehyde to Isoprene emissions using $\Omega_{HCHO} = S \times E_{ISOP} + B$
    Y_isop is the RMA regression between $k_{HCHO}*\Omega_{HCHO}$, and $k_{isop}*\Omega_{isop}$
    and $S = Y_{isop} / k_{HCHO}$
    B will be calculated from the remote pacific near the same latitude.

@author: jesse
"""


###############
### MODULES ###
###############
import numpy as np
from datetime import datetime
from scipy.constants import N_A as N_avegadro

# local imports
from utilities.JesseRegression import RMA
from classes import GC_class  # Class reading GC output
from classes.omhchorp import omhchorp # class reading OMHCHORP
from utilities import fio as fio
import utilities.plotting as pp
import utilities.utilities as util

import timeit
# EG using timer:
#start_time=timeit.default_timer()
#runprocess()
#elapsed = timeit.default_timer() - start_time
#print ("TIMEIT: Took %6.2f seconds to runprocess()"%elapsed)

###############
### GLOBALS ###
###############
__VERBOSE__=True
__DEBUG__=True

###############
### METHODS ###
###############

def Yield(H, k_H, I, k_I):
    '''
        H=HCHO, k_H=loss, I=Isoprene, k_I=loss
        The returned yield will match the first two dimensions.

        Y_hcho is the RMA regression between $k_{HCHO}*\Omega_{HCHO}$, and $k_{isop}*\Omega_{isop}$
        As is shown in fig. 7 Millet 2006

        As a test Yield can be calculated from Emissions as in Palmer2003
        Y_hcho = S E_isop

        I think this should be run on each 8 days of gridded model output to get the Yield for that 8 days

    '''
    # first handle the lats and lons:
    n0=np.shape(H)[0]
    n1=np.shape(H)[1]

    #Isoprene Yield, spatially varying, not temporally varying:
    #
    Y=np.zeros([n0,n1])
    A=k_H*H
    B=k_I*I
    for i in range(n0):
        for j in range(n1):
            m,b,r,ci1,ci2=RMA(A[i,j,:], B[i,j,:])
            Y[i,j] = m

    #Return the yields
    return Y

def Emissions(day, GC_biog, OMI, region=pp.__AUSREGION__):
    '''
        Return one day of emissions estimates
            Uses one month of GEOS-Chem (GC) estimated 'slope' for Y_hcho
            uses one day of OMI HCHO
            Use biogenic run for GC_biog
    '''

    # Check that GC_biog dates are the whole month - for slope calculation
    assert GC_biog.hemco.dates[0] == util.first_day(day), 'First day doesn\'t match in Emissions GC_biog parameter'
    assert GC_biog.hemco.dates[-1] == util.last_day(day), 'Last day doesn\'t match in Emissions GC_biog parameter'

    dstr=day.strftime("%Y%m%d")
    attrs={} # attributes dict for meta data
    GC=GC_biog.sat_out # Satellite output from biogenic run

    if __VERBOSE__:
        # Check the dims of our stuff
        print()
        print("Calculating emissions for %s"%dstr)
        print("GC data %s "%str(GC.hcho.shape)) # [t,lat,lon,lev]
        print("nanmean:",np.nanmean(GC.hcho),GC.attrs['hcho']['units']) # should be molecs/cm2
        print("OMI data %s"%str(OMI.VCC.shape)) # [t, lat, lon]
        print("nanmean:",np.nanmean(OMI.VCC),'molecs/cm2')# should be molecs/cm2

    omilats0, omilons0 = OMI.lats, OMI.lons
    omi_lats, omi_lons= omilats0.copy(), omilons0.copy()
    omi_SA=OMI.surface_areas # in km^2

    if __DEBUG__:
        GC_E_isop=GC_biog.hemco.E_isop_bio
        print("GC_E_isop%s [%s] before LT averaging:"%(str(np.shape(GC_E_isop)),GC_biog.hemco.attrs['E_isop_bio']['units']),np.nanmean(GC_E_isop))
        print("    non zero only:",np.nanmean(GC_E_isop[GC_E_isop > 0]))
    # Get GC_isoprene for this day also
    GC_days, GC_E_isop = GC_biog.hemco.daily_LT_averaged(hour=13)

    #GC_E_isop=GC.get_field(keys=['E_isop_bio',],region=region)['E_isop_bio']
    if __DEBUG__:
        print("GC_E_isop%s after LT averaging:"%str(np.shape(GC_E_isop)),np.nanmean(GC_E_isop))
        print("    non zero only:",np.nanmean(GC_E_isop[GC_E_isop > 0]))

        print("GC_E_isop.shape before and after dateindex")
        print(GC_E_isop.shape)
    GC_E_isop=GC_E_isop[util.date_index(day,GC_days)] # only want one day of E_isop_GC
    if __DEBUG__:
        print(GC_E_isop.shape)
    attrs['GC_E_isop'] = GC_biog.hemco.attrs['E_isop_bio']
    attrs['GC_E_isop']['desc']='biogenic isoprene emissions from MEGAN/GEOS-Chem'

    # subset to region
    if __DEBUG__:
        print('shape and mean before and after subsetting GC_E_isop:')
        print(np.shape(GC_E_isop), np.nanmean(GC_E_isop))
    GC_E_isop = util.lat_lon_subset(GC.lats,GC.lons,region=region,data=[GC_E_isop])['data'][0]

    if __DEBUG__:
        print(np.shape(GC_E_isop), np.nanmean(GC_E_isop))

    # model slope between HCHO and E_isop:
    # This also returns the lats and lons for just this region.
    S_model=GC_biog.model_slope(region=region) # in seconds I think
    GC_slope=S_model['slope']
    GC_lats, GC_lons=S_model['lats'], S_model['lons']

    if __VERBOSE__:
        print("%d lats, %d lons for GC(region)"%(len(GC_lats),len(GC_lons)))
        print("%d lats, %d lons for OMI(global)"%(len(omi_lats),len(omi_lons)))

    # OMI corrected vertical columns for matching day
    omi_day=OMI.time_averaged(day0=day,month=False,keys=['VCC','VCC_PP'])
    vcc=omi_day['VCC']
    vcc_pp=omi_day['VCC_PP']
    gridentries=omi_day['gridentries']
    ppentries=omi_day['ppentries']
    attrs['gridentries']={'desc':'OMI satellite pixels used in each gridbox'}
    attrs['ppentries']={'desc':'OMI satellite pixels used in each gridbox, recalculated using PP code'}
    # subset omi to region
    #
    omi_lati,omi_loni = util.lat_lon_range(omi_lats,omi_lons,region)
    omi_lats,omi_lons = omi_lats[omi_lati], omi_lons[omi_loni]
    attrs["lats"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    attrs["lons"]={"units":"degrees",
        "desc":"gridbox midpoint"}

    omi_SA=omi_SA[omi_lati,:]
    omi_SA=omi_SA[:,omi_loni]
    omi_hcho=omi_hcho[omi_lati,:]
    omi_hcho=omi_hcho[:,omi_loni]
    if __DEBUG__:
        print('%d lats, %d lons in region'%(len(omi_lats),len(omi_lons)))
        print('HCHO shape: %s'%str(omi_hcho.shape))

    ## map GC stuff onto same lats/lons as OMI
    slope_before=np.nanmean(GC_slope)
    GC_slope0=np.copy(GC_slope)
    GC_slope=util.regrid(GC_slope, GC_lats, GC_lons,omi_lats,omi_lons)
    GC_E_isop=util.regrid(GC_E_isop, GC_lats,GC_lons,omi_lats,omi_lons)
    slope_after=np.nanmean(GC_slope)
    attrs["GC_slope"]={"units":"s",
        "desc":"\"VC_H=S*E_i+B\" slope (S) between HCHO_GC (molec/cm2) and E_Isop_GC (atom c/cm2/s)"}

    with np.errstate(divide='ignore', invalid='ignore'):
        check=100.0*np.abs((slope_after-slope_before)/slope_before)

    # If the grids are compatible then slope shouldn't change from regridding
    if check > 1:
        print("Regridded slope changes by %.2f%%, from %.2f to %.2f"%(check,slope_before,slope_after))
        vmin,vmax=1000,300000
        regionB=np.array(region) + np.array([-10,-10,10,10])
        pp.createmap(GC_slope0, GC_lats, GC_lons, title="GC_slope0",
                     vmin=vmin, vmax=vmax, region=regionB,
                     pname="Figs/Checks/SlopeBefore_%s_%s.png"%(str(region),dstr))

        pp.createmap(GC_slope, omi_lats, omi_lons, title="GC_Slope",
                     vmin=vmin, vmax=vmax, region=regionB,
                     pname="Figs/Checks/SlopeAfter_%s_%s.png"%(str(region),dstr))
        print("CHECK THE SLOPE BEFORE AND AFTER REGRID IMAGES")
        #assert False, "Slope change too high"

    if __VERBOSE__:
        minmeanmax=(np.nanmin(GC_slope),np.nanmean(GC_slope),np.nanmax(GC_slope))
        print("min/mean/max GC_slope: %1.1e/%1.1e/%1.1e"%minmeanmax)

    # Determine background using region latitude bounds
    omi_background = OMI.get_background_array(lats=omi_lats,lons=omi_lons)
    attrs["background"]={"units":"molec HCHO/cm2",
        "desc":"background from recalculated OMI HCHO swathes"}

    ## Calculate Emissions from these
    ##

    # \Omega_{HCHO} = S \times E_{isop} + B
    # E_isop = (Column_hcho - B) / S
    E_new = (omi_hcho - omi_background) / GC_slope
    attrs["E_isop"]={"units":"atom C/cm2/s",
                     "desc" :"Emissions estimated using OMI_HCHO=S_GC * E_isop + OMI_BG"}

    # store GC_background for comparison
    GC_background=S_model['b']

    ## Calculate in kg/s for each grid box:
    # [atom C / cm2 / s ] * 1/5 * cm2/km2 * km2 * kg/atom_isop
    # = isoprene kg/s
    # kg/atom_isop = grams/mole * mole/molec * kg/gram
    kg_per_atom = util.__grams_per_mole__['isop'] * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * omi_SA * kg_per_atom
    E_isop_kgs=E_new*conversion
    GC_E_isop_kgs=GC_E_isop*conversion
    attrs["E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}
    attrs["GC_E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}

    return {'E_isop':E_new, 'E_isop_kg':E_isop_kgs,
            'GC_E_isop':GC_E_isop, 'GC_E_isop_kg':GC_E_isop_kgs,
            'lats':omi_lats, 'lons':omi_lons, 'background':omi_background,
            'GC_background':GC_background, 'GC_slope':GC_slope,
            'lati':omi_lati,'loni':omi_loni,'gridentries':gridentries,
            'attributes':attrs}

def Emissions_old(day0, dayn, GC = None, OMI = None,
              region=pp.__AUSREGION__, ignorePP=True):
    '''
        Determine emissions of isoprene averaged over some length of time.
        1) Calculates model slope E_isop -> Tropcol_HCHO
        2) Use Biogenic OMI HCHO product to calculate E_isoprene.
        Notes:
            Up to 1 month at a time, OMI is averaged over day0 -> dayn

        HCHO = S * E_isop + b
    '''

    # function will return a dictionary with all the important stuff in it
    outdict={}
    # also returns a dictionary of dicts for attributes:
    attrs={}

    dstr=day0.strftime("%Y%m%d")
    if __VERBOSE__:
        print("Entering Inversion.Emissions(%s)"%dstr)
    ## Read data for this date unless it's been passed in
    ##
    if GC is None:
        # GC satellite needs whole biogenic month to make slope
        GC=GC_class.GC_sat(day0=datetime(day0.year,day0.month,1), dayN=util.last_day(day0), run='biogenic')
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=ignorePP)

    if __VERBOSE__:
        # Check the dims of our stuff
        print("GC data %s"%str(GC.hcho.shape))
        #print("Lats from %.2f to %.2f"%(GC.lats[0],GC.lats[-1]))
        #print("Lons from %.2f to %.2f"%(GC.lons[0],GC.lons[-1]))
        print("OMI data %s"%str(OMI.VCC.shape))
        #print("Lats from %.2f to %.2f"%(OMI.lats[0],OMI.lats[-1]))
        #print("Lons from %.2f to %.2f"%(OMI.lons[0],OMI.lons[-1]))

    # Pull out the stuff we need from the classes
    latsomi0,lonsomi0=OMI.lats,OMI.lons
    latsomi, lonsomi= latsomi0.copy(), lonsomi0.copy()
    SA=OMI.surface_areas
    vcc=OMI.VCC
    pixels=OMI.gridentries
    if not ignorePP:
        vcc_pp=OMI.VCC_PP
        pixels_pp=OMI.ppentries

    # model slope between HCHO and E_isop:
    # This also returns the lats and lons for just this region.
    S_model=GC.model_slope(region=region) # in seconds I think

    slopegc=S_model['slope']
    latsgc, lonsgc=S_model['lats'], S_model['lons']

    if __VERBOSE__:
        print("%d lats, %d lons for GC(region)"%(len(latsgc),len(lonsgc)))
        print("%d lats, %d lons for OMI(global)"%(len(latsomi0),len(lonsomi0)))

    # Get OMI corrected vertical columns, averaged over time

    hchoomi=OMI.time_averaged(day0=day0,dayn=dayn,keys=['VCC'])['VCC']
    #if ReduceOmiRes > 0 :
    #    if __VERBOSE__:
    #        print("Lowering resolution by factor of %d"%ReduceOmiRes)
    #    omilow= OMI.lower_resolution('VCC', factor=ReduceOmiRes, dates=[day0,dayn])
    #    hchoomi=omilow['VCC']
    #    latsomi, lonsomi=omilow['lats'], omilow['lons']
    #    SA=omilow['surface_areas']


    # subset omi to region
    #
    latiomi,loniomi = util.lat_lon_range(latsomi,lonsomi,region)
    latsomi,lonsomi=latsomi[latiomi], lonsomi[loniomi]
    attrs["lats"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    outdict['lats']=latsomi
    attrs["lons"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    outdict['lons']=lonsomi

    SA=SA[latiomi,:]
    SA=SA[:,loniomi]
    hchoomi=hchoomi[latiomi,:]
    hchoomi=hchoomi[:,loniomi]
    if __VERBOSE__:
        print('%d lats, %d lons in region'%(len(latsomi),len(lonsomi)))
        print('HCHO shape: %s'%str(hchoomi.shape))

    ## map GC slope onto same lats/lons as OMI
    slope_before=np.nanmean(slopegc)
    GC_slope=util.regrid(slopegc, latsgc, lonsgc,latsomi,lonsomi)
    slope_after=np.nanmean(GC_slope)
    attrs["GC_slope"]={"units":"s",
        "desc":"slope between HCHO_GC (molec/cm2) and E_Isop_GC (atom c/cm2/s)"}
    outdict['GC_slope']=GC_slope

    check=100.0*np.abs((slope_after-slope_before)/slope_before)
    if check > 1:
        print("Regridded slope changes by %.2f%%, from %.2f to %.2f"%(check,slope_before,slope_after))

        #[ print(x) for x in [latsgc, lonsgc,latsomi,lonsomi] ]
        vmin,vmax=1000,300000
        regionB=np.array(region) + np.array([-10,-10,10,10])
        pp.createmap(slopegc, latsgc, lonsgc, title="slopegc",
                     vmin=vmin, vmax=vmax, region=regionB,
                     pname="Figs/Checks/SlopeBefore_%s_%s.png"%(str(region),dstr))
        #print("GC_Slope shapes")
        #[ print (np.shape(x)) for x in [GC_slope, latsomi, lonsomi] ]
        pp.createmap(GC_slope, latsomi, lonsomi, title="GC_Slope",
                     vmin=vmin, vmax=vmax, region=regionB,
                     pname="Figs/Checks/SlopeAfter_%s_%s.png"%(str(region),dstr))
        print("CHECK THE SLOPE BEFORE AND AFTER REGRID IMAGES FOR ERROR")
        #assert False, "Slope change too high"
    if __VERBOSE__:
        print("Regridding slope passes change tests")
        print("Mean slope = %1.3e"%np.nanmean(GC_slope))

    # Subset our slopes
    # This is done by util.regrid above
    #GC_slope=GC_slope[latiomi,:]
    #GC_slope=GC_slope[:,loniomi]

    # Determine background using region latitude bounds
    BGomi    = OMI.background_HCHO(lats=[region[0],region[2]])
    attrs["BC_OMI"]={"units":"molec HCHO/cm2",
                     "desc":"background from OMI HCHO swathes"}
    outdict['BG_OMI']=BGomi

    ## Calculate Emissions from these
    ##

    # \Omega_{HCHO} = S \times E_{isop} + B
    # E_isop = (Column_hcho - B) / S
    #   Determine S from the slope bewteen E_isop and HCHO
    #[print(np.nanmean(x)) for x in [hchoomi, BGomi, GC_slope]]
    E_new = (hchoomi - BGomi) / GC_slope
    attrs["Eisop_OMIGC"]={"units":"atom C/cm2/s",
                          "desc":"Emissions using GEOS-Chem modelled slope, applied against VCC (OMI Vertical columns with GC based apriori)"}
    outdict['Eisop_OMIGC']=E_new

    # E_new_PP
    if not ignorePP:
        E_new_pp = ()

    # loss rate of HCHO
    # loss rate of Isop

    #
    #TODO: store GC_background for comparison
    GC_BG=np.array([np.NaN])
    attrs['BG_GC']={"units":"molec HCHO/cm2",
                    "desc" :"background from GEOS-Chem"}
    outdict['BG_GC']=GC_BG

    ## TODO: Also store GC estimate (For easy MEGAN comparison)
    # GEOS-Chem over our region:
    #E_GC_sub=GC.get_field(keys=['E_isop_bio','E_isop_bio_kgs'], region=region)
    #Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values

    # map the lower resolution data onto the higher resolution data:
    #megan=E_GC_sub['E_isop_bio']
    #megan_kgs=E_GC_sub['E_isop_bio_kgs']
    #for i in range() #DAILY REGRID...
    #megan = util.regrid(megan,E_GC_sub['lats'],E_GC_sub['lons'],omilats,omilons)

    ## Calculate in kg/s for each grid box:
    # newE in atom C / cm2 / s  |||  * 1/5 * cm2/km2 * km2 * kg/atom_isop
    # = isoprene kg/s
    # kg/atom_isop = grams/mole * mole/molec * kg/gram
    kg_per_atom = util.__grams_per_mole__['isop'] * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * SA * kg_per_atom
    E_isop_kgs=E_new*conversion
    attrs["Eisop_OMIGC_kg"]={"units":"kg/s",
                        "desc":"emissions/cm2 multiplied by area, calculated using OMI with GC apriori"}
    outdict['Eisop_OMIGC_kg']=E_isop_kgs

    #outdict['attributes']=attrs

    # Return dict with:
    # 'BG_GC','GC_slope'
    # 'lats','lons'
    return outdict, attrs

def store_emissions_month(month=datetime(2005,1,1), GCB=None, OMHCHORP=None,
                          region=pp.__AUSREGION__):
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
    if OMHCHORP is None:
        OMHCHORP=omhchorp(day0=day0,dayn=dayn, ignorePP=False)
    if GCB is None:
        GCB=GC_class.GC_biogenic(day0,) # data like [time,lat,lon,lev]

    # subset our lats/lons
    #omilats=OMI.lats
    #omilons=OMI.lons
    #omilati, omiloni = util.lat_lon_range(omilats,omilons,region=region)
    #newlats=omilats[omilati]
    #newlons=omilons[omiloni]
    # Arrays to be subset
    arrs=[getattr(OMHCHORP,s) for s in ['firemask','smokemask','anthromask']]
    subsets=util.lat_lon_subset(OMHCHORP.lats,OMHCHORP.lons,region,)

    # We need to make the anthropogenic, fire and smoke masks:
    #firemask=fio.make_fire_mask(d0=day0, dN=dayn, prior_days_masked=2, fire_thresh=1, adjacent=True)
    #smokemask=fio.make_smoke_mask(d0=day0, dN=dayn, aaod_thresh=0.2)
    #anthrofilter=fio.make_anthro_mask(d0=day0, dN=dayn)

    # fire filter made up of two masks:
    firefilter=(firemask+smokemask).astype(np.bool)

    # Need Vertical colums, slope, and backgrounds all at same resolution to get emissions
    VCC                   = np.copy(OMI.VCC)
    VCC_PP                = np.copy(OMI.VCC_PP)
    VCC_OMI               = np.copy(OMI.VC_OMI_RSC)
    pixels                = np.copy(OMI.gridentries)
    pixels_PP             = np.copy(OMI.ppentries)
    SArea                 = np.copy(OMI.surface_areas)
    uncert                = np.copy(OMI.col_uncertainty_OMI)



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
    uncert      = uncert[:,omilati,:]
    uncert      = uncert[:,:,omiloni]
    firefilter  = firefilter[:,omilati,:]
    firefilter  = firefilter[:,:,omiloni]
    anthrofilter= anthrofilter[:,omilati,:]
    anthrofilter= anthrofilter[:,:,omiloni]

    # emissions using different columns as basis
    # Fully filtered:
    E_vcc       = np.zeros(VCC.shape) + np.NaN
    E_pp        = np.zeros(VCC.shape) + np.NaN
    E_omi       = np.zeros(VCC.shape) + np.NaN
    # Only fire filtered
    E_vcc_f     = np.zeros(VCC.shape) + np.NaN
    E_pp_f      = np.zeros(VCC.shape) + np.NaN
    E_omi_f     = np.zeros(VCC.shape) + np.NaN
    # Only anthro filtered
    E_vcc_a     = np.zeros(VCC.shape) + np.NaN
    E_pp_a      = np.zeros(VCC.shape) + np.NaN
    E_omi_a     = np.zeros(VCC.shape) + np.NaN
    # unfiltered:
    E_vcc_u     = np.zeros(VCC.shape) + np.NaN
    E_pp_u      = np.zeros(VCC.shape) + np.NaN
    E_omi_u     = np.zeros(VCC.shape) + np.NaN

    BG_VCC      = np.zeros(VCC.shape) + np.NaN
    BG_PP       = np.zeros(VCC.shape) + np.NaN
    BG_OMI      = np.zeros(VCC.shape) + np.NaN

    time_emiss_calc=timeit.default_timer()
    for i,day in enumerate(days):

        # Need background values from remote pacific
        BG_VCCi, bglats, bglons = util.remote_pacific_background(OMI.VCC[i], omilats, omilons, average_lons=True)
        BG_PPi , bglats, bglons = util.remote_pacific_background(OMI.VCC_PP[i], omilats, omilons, average_lons=True)
        BG_OMIi, bglats, bglons = util.remote_pacific_background(OMI.VC_OMI_RSC[i], omilats, omilons, average_lons=True)

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

        # Run calculation with no filters applied:
        E_vcc_u[i,:,:]      = (VCC[i] - BG_VCCi) / GC_slope
        E_pp_u[i,:,:]       = (VCC_PP[i] - BG_PPi) / GC_slope
        E_omi_u[i,:,:]      = (VCC_OMI[i] - BG_OMIi) / GC_slope

        # Again with fire filter applied
        ff                  = firefilter[i]
        vcci                = np.copy(VCC[i])
        vcci[ff]            = np.NaN
        vcc_ppi             = np.copy(VCC_PP[i])
        vcc_ppi[ff]         = np.NaN
        vcc_omii            = np.copy(VCC_OMI[i])
        vcc_omii[ff]        = np.NaN
        E_vcc_f[i,:,:]        = (vcci - BG_VCCi) / GC_slope
        E_pp_f[i,:,:]         = (vcc_ppi - BG_PPi) / GC_slope
        E_omi_f[i,:,:]        = (vcc_omii - BG_OMIi) / GC_slope

        # again with just anthro filter applied
        af                  = anthrofilter[i]
        vcci                = np.copy(VCC[i])
        vcci[af]            = np.NaN
        vcc_ppi             = np.copy(VCC_PP[i])
        vcc_ppi[af]         = np.NaN
        vcc_omii            = np.copy(VCC_OMI[i])
        vcc_omii[af]        = np.NaN
        E_vcc_a[i,:,:]        = (vcci - BG_VCCi) / GC_slope
        E_pp_a[i,:,:]         = (vcc_ppi - BG_PPi) / GC_slope
        E_omi_a[i,:,:]        = (vcc_omii - BG_OMIi) / GC_slope

        # finally with both filters
        faf                 = firefilter[i] + anthrofilter[i]
        vcci                = np.copy(VCC[i])
        vcci[faf]           = np.NaN
        vcc_ppi             = np.copy(VCC_PP[i])
        vcc_ppi[faf]        = np.NaN
        vcc_omii            = np.copy(VCC_OMI[i])
        vcc_omii[faf]       = np.NaN
        E_vcc[i,:,:]        = (vcci - BG_VCCi) / GC_slope
        E_pp[i,:,:]         = (vcc_ppi - BG_PPi) / GC_slope
        E_omi[i,:,:]        = (vcc_omii - BG_OMIi) / GC_slope

    elapsed = timeit.default_timer() - time_emiss_calc
    print ("TIMEIT: Took %6.2f seconds to calculate backgrounds and estimate emissions()"%elapsed)
    # should take very little time

    # Lets save both monthly averages and the daily amounts
    #

    # Save the backgrounds, as well as units/descriptions
    outdata['BG_VCC']    = BG_VCC
    outdata['BG_PP']     = BG_PP
    outdata['BG_OMI']    = BG_OMI
    outattrs['BG_VCC']   = {'unit':'molec/cm2','desc':'Background: VCC zonally averaged from remote pacific'}
    outattrs['BG_PP']    = {'unit':'molec/cm2','desc':'Background: VCC_PP zonally averaged from remote pacific'}
    outattrs['BG_OMI']   = {'unit':'molec/cm2','desc':'Background: VCC_OMI zonally averaged from remote pacific'}

    # Save the Vertical columns, as well as units/descriptions
    outdata['VCC']        = VCC
    outdata['VCC_PP']     = VCC_PP
    outdata['VCC_OMI']    = VCC_OMI
    outattrs['VCC']       = {'unit':'molec/cm2','desc':'OMI (corrected) Vertical column using recalculated shape factor, fire and anthro masked'}
    outattrs['VCC_PP']    = {'unit':'molec/cm2','desc':'OMI (corrected) Vertical column using PP code, fire and anthro masked'}
    outattrs['VCC_OMI']   = {'unit':'molec/cm2','desc':'OMI (corrected) Vertical column, fire and anthro masked'}

    # Save the Emissions estimates, as well as units/descriptions
    outdata['E_VCC']        = E_vcc
    outdata['E_VCC_PP']     = E_pp
    outdata['E_VCC_OMI']    = E_omi
    outdata['E_VCC_f']      = E_vcc_f
    outdata['E_VCC_PP_f']   = E_pp_f
    outdata['E_VCC_OMI_f']  = E_omi_f
    outdata['E_VCC_a']      = E_vcc_a
    outdata['E_VCC_PP_a']   = E_pp_a
    outdata['E_VCC_OMI_a']  = E_omi_a
    outdata['E_VCC_u']      = E_vcc_u
    outdata['E_VCC_PP_u']   = E_pp_u
    outdata['E_VCC_OMI_u']  = E_omi_u
    outattrs['E_VCC']       = {'unit':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope'}
    outattrs['E_VCC_PP']    = {'unit':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope'}
    outattrs['E_VCC_OMI']   = {'unit':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope'}
    outattrs['E_VCC_f']     = {'unit':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope, just fires masked'}
    outattrs['E_VCC_PP_f']  = {'unit':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope, just fires masked'}
    outattrs['E_VCC_OMI_f'] = {'unit':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope, just fires masked'}
    outattrs['E_VCC_a']     = {'unit':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope, just anthro masked'}
    outattrs['E_VCC_PP_a']  = {'unit':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope, just anthro masked'}
    outattrs['E_VCC_OMI_a'] = {'unit':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope, just anthro masked'}
    outattrs['E_VCC_u']     = {'unit':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_PP_u']  = {'unit':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_OMI_u'] = {'unit':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope, unmasked by fire or anthro'}

    # Extras like pixel counts etc..
    outdata['firefilter']   = firefilter.astype(np.int)
    outdata['anthrofilter'] = anthrofilter.astype(np.int)
    outdata['pixels']       = pixels
    outdata['pixels_PP']    = pixels_PP
    outdata['uncert_OMI']   = uncert
    outattrs['firefilter']  = {'unit':'N/A',
                               'desc':'Squares with more than one fire or more than 0.2 aaod: 1 for True (filtered)'}
    outattrs['anthrofilter']= {'unit':'N/A',
                               'desc':'Squares with tropNO2 from OMI greater than %.1e'%no2thresh}
    outattrs['uncert_OMI']  = {'unit':'?? molec/cm2 ??',
                               'desc':'OMI pixel uncertainty averaged for each gridsquare'}
    outattrs['pixels']      = {'unit':'n',
                               'desc':'OMI pixels used for gridsquare VC'}
    outattrs['pixels_PP']   = {'unit':'n',
                               'desc':'OMI pixels after PP code used for gridsquare VC'}

    # Adding time dimension (needs to be utf8 for h5 files)
    #dates = np.array([d.strftime("%Y%m%d").encode('utf8') for d in days])
    dates = np.array([int(d.strftime("%Y%m%d")) for d in days])
    outdata["time"]=dates
    outattrs["time"]={"format":"%Y%m%d", "desc":"year month day as integer (YYYYMMDD)"}
    fattrs={'region':"SWNE: %s"%str(region)}
    fattrs['date range']="%s to %s"%(d0str,dnstr)

    # Save lat,lon
    outdata['lats']=newlats
    outdata['lons']=newlons
    outdata['lats_e']=util.edges_from_mids(outdata['lats'])
    outdata['lons_e']=util.edges_from_mids(outdata['lons'])

    outdata['smearing'] = smearing(month,plot=True,region=region)
    outattrs['smearing']= {'desc':'smearing = Delta(HCHO)/Delta(E_isop), where Delta is the difference between full and half isoprene emission runs from GEOS-Chem for %s'%mstr}

    # Save file, with attributes
    fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)
    if __VERBOSE__:
        print("%s should now be saved"%fname)

def store_emissions(day0=datetime(2005,1,1), dayn=None,
                    region=pp.__GLOBALREGION__, ignorePP=True):
    '''
        Store many months of new emissions estimates into he5 files
    '''

    # If just a day is input, then save a month
    if dayn is None:
        dayn=util.last_day(day0)
    if __VERBOSE__:
        print("Running Inversion.store_emissions()")


    months=util.list_months(day0,dayn)
    # save each month seperately
    #
    for month in months:

        # Grab reprocessed data for the month:
        OMI=omhchorp(day0=month,dayn=util.last_day(month), ignorePP=ignorePP)
        print('mean HCHO column [molec/cm2] from omhchorp',np.nanmean(OMI.VCC))

        # Read GC month:
        GC=GC_class.GC_biogenic(month)
        print('mean surface hcho [ppb] from GC_biogenic run:',np.nanmean(GC.sat_out.hcho[:,:,:,0]))

        # save the month of emissions
        store_emissions_month(month=month, GC=GC, OMI=OMI,
                              region=region, ignorePP=ignorePP)

    if __VERBOSE__:
        print("Inversion.store_emissions() now finished")

def smearing(month, plot=False,region=pp.__AUSREGION__,thresh=0.0):
    '''
        Read full and half isop bpch output, calculate smearing
        S = d column_HCHO / d E_isop
        For now uses tavg instead of overpass times
    '''
    if __VERBOSE__:
        print('calculating smearing over ',region,' in month ',month)

    full=GC_class.GC_tavg(month, run='tropchem')
    half=GC_class.GC_tavg(month, run='halfisop') # month avg right now

    lats=full.lats
    lons=full.lons

    assert all(lats == half.lats), "Lats between runs don't match"

    full_month=full.month_average(keys=['O_hcho','E_isop_bio'])
    half_month=half.month_average(keys=['O_hcho','E_isop_bio'])
    f_hcho=full_month['O_hcho'] # molec/cm2
    h_hcho=half_month['O_hcho'] # molec/cm2
    f_E_isop=full_month['E_isop_bio'] # molec/cm2/s
    h_E_isop=half_month['E_isop_bio'] # molec/cm2/s

    dlist=[f_hcho,h_hcho,f_E_isop,h_E_isop]
    #for ddata in dlist:
    #    print('nanmean ',np.nanmean(ddata))
    sub=util.lat_lon_subset(lats,lons,region,dlist)
    lats=sub['lats']; lons=sub['lons']
    lats_e=sub['lats_e']; lons_e=sub['lons_e']
    f_hcho=sub['data'][0]
    h_hcho=sub['data'][1]
    f_E_isop=sub['data'][2]
    h_E_isop=sub['data'][3]
    #for ddata in [f_hcho,h_hcho,f_E_isop,h_E_isop]:
    #    print('nanmean after subsetting',np.nanmean(ddata))

    # where emissions are zero, smearing is infinite, ignore warning:
    with np.errstate(divide='ignore'):
        S = (f_hcho - h_hcho) / (f_E_isop - h_E_isop) # s

    #print("emissions from Full,Half:",np.sum(f_E_isop),np.sum(h_E_isop))
    #print("O_hcho from Full, Half:",np.sum(f_hcho),np.sum(h_hcho))
    S[f_E_isop <= thresh] = np.NaN

    print("S shape:",S.shape)
    print("Average S:",np.nanmean(S))
    if plot:
        pp.InitMatplotlib()
        dstr=month.strftime("%Y%m%d")
        pname='Figs/GC/smearing_%s.png'%dstr

        #pp.createmap(f_hcho,lats,lons,latlon=True,edges=False,linear=True,pname=pname,title='f_hcho')
        # lie about edges...
        pp.createmap(S,lats,lons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
                     linear=True,vmin=1000,vmax=10000,
                     clabel='S', pname=pname, title='Smearing %s'%dstr)

    return S

if __name__=='__main__':
    print('Inversion has been run')

    day0=datetime(2005,1,1)
    dayn=datetime(2005,2,28)
    store_emissions(day0=day0,dayn=dayn)
    #for day in [datetime(2005,9,1),datetime(2005,10,1),datetime(2005,11,1),datetime(2005,12,1),]:
    #    #smearing(day0)
    #    store_emissions(day0=day)
