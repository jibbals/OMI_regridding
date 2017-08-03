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
from classes.GC_class import GC_output # Class reading GC output
from classes.omhchorp import omhchorp # class reading OMHCHORP
from utilities import fio as fio
import utilities.plotting as pp
import utilities.utilities as util

###############
### GLOBALS ###
###############
__VERBOSE__=True

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

def Emissions(day0, dayn, GC = None, OMI = None,
              region=pp.__AUSREGION__, ReduceOmiRes=0, ignorePP=True):
    '''
        Determine emissions of isoprene averaged over some length of time.
        1) Calculates model slope E_isop -> Tropcol_HCHO
        2) Use Biogenic OMI HCHO product to calculate E_isoprene.
        Notes:
            Up to 1 month at a time, OMI is averaged over day0 -> dayn

        HCHO = S * E_isop + b
    '''
    dstr=day0.strftime("%Y%m%d")
    if __VERBOSE__:
        print("Entering Inversion.Emissions(%s)"%dstr)
    ## Read data for this date unless it's been passed in
    ##
    if GC is None:
        GC=GC_output(date=day0)
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=ignorePP)

    # will return also a dictionary of dicts for attributes:
    attrs={}

    if __VERBOSE__:
        # Check the dims of our stuff
        print("GC data %s"%str(GC.hcho.shape))
        #print("Lats from %.2f to %.2f"%(GC.lats[0],GC.lats[-1]))
        #print("Lons from %.2f to %.2f"%(GC.lons[0],GC.lons[-1]))
        print("OMI data %s"%str(OMI.VCC.shape))
        #print("Lats from %.2f to %.2f"%(OMI.lats[0],OMI.lats[-1]))
        #print("Lons from %.2f to %.2f"%(OMI.lons[0],OMI.lons[-1]))
    latsomi0,lonsomi0=OMI.lats,OMI.lons
    latsomi, lonsomi= latsomi0.copy(), lonsomi0.copy()
    SA=OMI.surface_areas

    # model slope between HCHO and E_isop:
    # This also returns the lats and lons for just this region.
    S_model=GC.model_slope(region=region) # in seconds I think

    slopegc=S_model['slope']
    latsgc, lonsgc=S_model['lats'], S_model['lons']

    if __VERBOSE__:
        print("%d lats, %d lons for GC(region)"%(len(latsgc),len(lonsgc)))
        print("%d lats, %d lons for OMI(global)"%(len(latsomi0),len(lonsomi0)))

    # Get OMI corrected vertical columns, averaged over time
    # And with reduced resolution if desired
    hchoomi=OMI.time_averaged(day0=day0,dayn=dayn,keys=['VCC'])['VCC']
    if ReduceOmiRes > 0 :
        if __VERBOSE__:
            print("Lowering resolution by factor of %d"%ReduceOmiRes)
        omilow=OMI.lower_resolution('VCC', factor=ReduceOmiRes, dates=[day0,dayn])
        hchoomi=omilow['VCC']
        latsomi, lonsomi=omilow['lats'], omilow['lons']
        SA=omilow['surface_areas']

    # subset omi to region
    #
    latiomi,loniomi = util.lat_lon_range(latsomi,lonsomi,region)
    latsomi,lonsomi=latsomi[latiomi], lonsomi[loniomi]

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
    # Molecules / cm2

    ## Calculate Emissions from these
    ##

    # \Omega_{HCHO} = S \times E_{isop} + B
    # E_isop = (Column_hcho - B) / S
    #   Determine S from the slope bewteen E_isop and HCHO
    #[print(np.nanmean(x)) for x in [hchoomi, BGomi, GC_slope]]
    E_new = (hchoomi - BGomi) / GC_slope
    # E_new is in atom C/cm2/s (same units as E_GC for slope)

    #print (np.nanmean(hcho))
    #print(BG)

    # loss rate of HCHO

    # loss rate of Isop

    #
    #TODO: store GC_background for comparison
    GC_BG=np.array([np.NaN])

    ## Calculate in kg/s for each grid box:
    # newE in atom C / cm2 / s  |||  * 1/5 * cm2/km2 * km2 * kg/atom_isop
    # = isoprene kg/s
    # kg/atom_isop = grams/mole * mole/molec * kg/gram
    kg_per_atom = util.isoprene_grams_per_mole * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * SA * kg_per_atom
    E_isop_kgs=E_new*conversion

    # Set up attributes for our file
    attrs["E_isop"]={"units":"atom C/cm2/s"}
    attrs["E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}
    attrs["lats"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    attrs["lons"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    attrs["background"]={"units":"molec HCHO/cm2",
        "desc":"background from OMI HCHO swathes"}
    attrs["GC_slope"]={"units":"s",
        "desc":"slope between HCHO_GC (molec/cm2) and E_Isop_GC (atom c/cm2/s)"}

    return {'E_isop':E_new, 'E_isop_kg':E_isop_kgs,
            'lats':latsomi, 'lons':lonsomi, 'background':BGomi,
            'GC_background':GC_BG, 'GC_slope':GC_slope,
            'lati':latiomi,'loni':loniomi,
            'attributes':attrs}

def Emissions_series(day0=datetime(2005,1,1), dayn=datetime(2005,2,1),
                     GC = None, OMI = None, region=pp.__AUSREGION__):
    '''
        Emissions over time
    '''
    if __VERBOSE__: print("Running Inversion.Emissions_series()")
    ## Read data for this date unless it's been passed in
    ##
    if GC is None:
        GC=GC_output(date=day0) # gets one month of GC.
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn)

    days=util.list_days(day0,dayn)
    Eomi=[]
    Egc=[]
    for day in days:
        E=Emissions(day0=day, dayn=None, GC=GC, OMI=OMI)


    #return {'E_isop':E_new, 'lats':lats, 'lons':lons, 'background':BG,
    #        'GC_background':GC_BG, 'GC_slope':GC_slope}


def store_emissions(day0=datetime(2005,1,1), dayn=None, GC=None, OMI=None,
                    region=pp.__GLOBALREGION__, ignorePP=True):
    '''
        Store a month of new emissions estimates into an he5 file
    '''
    mstr=day0.strftime("%Y%m")
    ddir="Data/Isop/E_new"
    fname=ddir+"/emissions_%s.h5"%mstr
    if __VERBOSE__:
        print("Reading %s Estimated Emissions over %s to file %s"%(mstr,str(region),fname))

    # If just a day is input, then save the whole month
    if dayn is None:
        dayn=util.last_day(day0)
    days=util.list_days(day0,dayn)

    if GC is None:
        GC=GC_output(date=day0)
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=ignorePP)

    Emiss=[]
    # Read each day then save the month
    for day in days:
        # Get a day of new emissions estimates
        Emiss.append(Emissions(day0=day, dayn=day, GC=GC, OMI=OMI, region=region))

    outattrs=Emiss[0]['attributes'] # get one copy of attributes is required

    # Adding time dimension (needs to be utf8 for h5 files)
    #dates = np.array([d.strftime("%Y%m%d").encode('utf8') for d in days])
    #outattrs["time"]={"format":"%Y%m%d", "desc":"year month day string"}
    dates = np.array([int(d.strftime("%Y%m%d")) for d in days])
    outdata={"time":dates}
    outattrs["time"]={"format":"%Y%m%d", "desc":"year month day as integer (YYYYMMDD)"}
    fattrs={'region':"SWNE: %s"%str(region)}

    # Save lat,lon
    outdata['lats']=Emiss[0]['lats']
    outdata['lons']=Emiss[0]['lons']
    outdata['lats_e']=util.edges_from_mids(outdata['lats'])
    outdata['lons_e']=util.edges_from_mids(outdata['lons'])

    # Save data into month of daily averages
    # TODO: keep OMI counts from earlier...
    keys_to_save=['E_isop', 'E_isop_kg','background', 'GC_background', 'GC_slope']
    #if not ignorePP: keys_to_save.append("") save PP based new emissions also..
    for key in keys_to_save:
        outdata[key]=np.array([E[key] for E in Emiss]) # time will be first dim

    # Save file, with attributes
    fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)

if __name__=='__main__':
    print('Inversion has been run')

    store_emissions()