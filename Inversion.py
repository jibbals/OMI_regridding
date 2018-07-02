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

__Thresh_Smearing__=5000

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


    # GC.model_slope gets slope and subsets the region
    # Then Map slope onto higher omhchorp resolution:
    slope_dict=GCB.model_slope(region=region)
    GC_slope=slope_dict['slope']
    gclats,gclons = slope_dict['lats'],slope_dict['lons']
    GC_slope = util.regrid_to_higher(GC_slope,gclats,gclons,omilats,omilons,interp='nearest')

    # Also save smearing
    smear, slats,slons = smearing(month,region=region)
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
    smearfilter = smear > __Thresh_Smearing__#5000 # something like this


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
    time_emiss_calc=timeit.default_timer()
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

        # run with filters
        allmasks            = firefilter[i] + anthrofilter[i] # + smearfilter
        vcc_gci             = np.copy(VCC_GC[i])
        vcc_gci[allmasks]   = np.NaN
        vcc_ppi             = np.copy(VCC_PP[i])
        vcc_ppi[allmasks]   = np.NaN
        vcc_omii            = np.copy(VCC_OMI[i])
        vcc_omii[allmasks]  = np.NaN
        E_gc[i,:,:]         = (vcc_gci - BG_VCCi) / GC_slope
        E_pp[i,:,:]         = (vcc_ppi - BG_PPi) / GC_slope
        E_omi[i,:,:]        = (vcc_omii - BG_OMIi) / GC_slope

    elapsed = timeit.default_timer() - time_emiss_calc
    print ("TIMEIT: Took %6.2f seconds to calculate backgrounds and estimate emissions()"%elapsed)
    # should take < 1 second

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
    outdata['E_VCC_GC_LR']  = E_gc_lr
    outdata['E_VCC_PP_LR']  = E_pp_lr
    outdata['E_VCC_OMI_LR'] = E_omi_lr
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
    outattrs['E_VCC_OMI_u'] = {'units':'molec/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_GC_LR'] = {'units':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_PP_LR'] = {'units':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_OMI_LR']= {'units':'molec/cm2/s',
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
                               'desc':'Squares where smearing greater than %.1f'%(__Thresh_Smearing__)}
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
    fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)
    if __VERBOSE__:
        print("%s should now be saved"%fname)

def store_emissions(day0=datetime(2005,1,1), dayn=None,
                    region=pp.__AUSREGION__, ignorePP=False):
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
        print('mean HCHO column [molec/cm2] from omhchorp',np.nanmean(OMI.VCC_GC))

        # Read GC month:
        GCB=GC_class.GC_biogenic(month)
        print('mean surface hcho [ppb] from GC_biogenic run:',np.nanmean(GCB.sat_out.hcho[:,:,:,0]))

        # save the month of emissions
        store_emissions_month(month=month, GCB=GCB, OMHCHORP=OMI,
                              region=region)

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
    lats=sub['lats']
    lons=sub['lons']
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
        mstr=month.strftime("%Y%m")
        pname='Figs/GC/smearing_%s.png'%mstr

        #pp.createmap(f_hcho,lats,lons,latlon=True,edges=False,linear=True,pname=pname,title='f_hcho')
        # lie about edges...
        pp.createmap(S,lats,lons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
                     linear=True, vmin=1000, vmax=10000,
                     clabel='S', pname=pname, title='Smearing %s'%mstr)

    return S, lats,lons

if __name__=='__main__':
    print('Inversion has been called...')

    t0=timeit.default_timer()
    day0=datetime(2005,1,1)
    dayn=datetime(2005,1,31)
    store_emissions(day0=day0,dayn=dayn)
    t1=timeit.default_timer()
    print("TIMEIT: took %6.2f minutes to run store_emissions(%s,%s)"%((t1-t0)/60.0,day0.strftime('%Y%m%d'),dayn.strftime('%Y%m%d')))
    #for day in [datetime(2005,9,1),datetime(2005,10,1),datetime(2005,11,1),datetime(2005,12,1),]:
    #    #smearing(day0)
    #    store_emissions(day0=day)
