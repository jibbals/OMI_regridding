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

def Emissions_1day(day, GC, OMI, region=pp.__AUSREGION__):
    '''
        Return one day of emissions estimates
            Uses one month of GEOS-Chem (GC) estimated 'slope' for Y_hcho
            uses one day of OMI HCHO
    '''
    dstr=day.strftime("%Y%m%d")
    attrs={} # attributes dict for meta data
    if __VERBOSE__:
        # Check the dims of our stuff
        print("GC data %s"%str(GC.hcho.shape))
        print("OMI data %s"%str(OMI.VCC.shape))
        print("Calculating emissions for %s"%dstr)

    omilats0, omilons0=OMI.lats,OMI.lons
    omi_lats, omi_lons= omilats0.copy(), omilons0.copy()
    omi_SA=OMI.surface_areas # in km^2

    # Get GC_isoprene for this day also
    GC_isop=GC.get_field(keys=['E_isop_bio',],region=region)['E_isop_bio']
    GC_isop=GC_isop[GC.date_index(day)] # only want one day of E_isop_GC
    attrs['GC_isop']={'units':'atom C/cm2/s',
                      'desc' :'biogenic isoprene emissions from MEGAN/GEOS-Chem'}

    # model slope between HCHO and E_isop:
    # This also returns the lats and lons for just this region.
    S_model=GC.model_slope(region=region) # in seconds I think
    GC_slope=S_model['slope']
    GC_lats, GC_lons=S_model['lats'], S_model['lons']

    if __VERBOSE__:
        print("%d lats, %d lons for GC(region)"%(len(GC_lats),len(GC_lons)))
        print("%d lats, %d lons for OMI(global)"%(len(omi_lats),len(omi_lons)))

    # Get OMI corrected vertical columns, averaged over time
    # And with reduced resolution if desired
    omi_day=OMI.time_averaged(day0=day,keys=['VCC'])
    omi_hcho=omi_day['VCC']
    gridentries=omi_day['gridentries']
    attrs['gridentries']={'desc':'OMI satellite pixels used in each gridbox'}
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
    if __VERBOSE__:
        print('%d lats, %d lons in region'%(len(omi_lats),len(omi_lons)))
        print('HCHO shape: %s'%str(omi_hcho.shape))

    ## map GC stuff onto same lats/lons as OMI
    slope_before=np.nanmean(GC_slope)
    GC_slope0=np.copy(GC_slope)
    GC_slope=util.regrid(GC_slope, GC_lats, GC_lons,omi_lats,omi_lons)
    GC_isop=util.regrid(GC_isop, GC_lats,GC_lons,omi_lats,omi_lons)
    slope_after=np.nanmean(GC_slope)
    attrs["GC_slope"]={"units":"s",
        "desc":"\"VC_H=S*E_i+B\" slope (S) between HCHO_GC (molec/cm2) and E_Isop_GC (atom c/cm2/s)"}

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
        print()
        print("Mean slope = %1.3e"%np.nanmean(GC_slope))

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
    kg_per_atom = util.isoprene_grams_per_mole * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * omi_SA * kg_per_atom
    E_isop_kgs=E_new*conversion
    GC_isop_kgs=GC_isop*conversion
    attrs["E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}
    attrs["GC_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}

    return {'E_isop':E_new, 'E_isop_kg':E_isop_kgs,
            'GC_isop':GC_isop, 'GC_isop_kg':GC_isop_kgs,
            'lats':omi_lats, 'lons':omi_lons, 'background':omi_background,
            'GC_background':GC_background, 'GC_slope':GC_slope,
            'lati':omi_lati,'loni':omi_loni,'gridentries':gridentries,
            'attributes':attrs}

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
    attrs["lats"]={"units":"degrees",
        "desc":"gridbox midpoint"}
    attrs["lons"]={"units":"degrees",
        "desc":"gridbox midpoint"}

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
    attrs["background"]={"units":"molec HCHO/cm2",
        "desc":"background from OMI HCHO swathes"}

    ## Calculate Emissions from these
    ##

    # \Omega_{HCHO} = S \times E_{isop} + B
    # E_isop = (Column_hcho - B) / S
    #   Determine S from the slope bewteen E_isop and HCHO
    #[print(np.nanmean(x)) for x in [hchoomi, BGomi, GC_slope]]
    E_new = (hchoomi - BGomi) / GC_slope
    attrs["E_isop"]={"units":"atom C/cm2/s"}

    #print (np.nanmean(hcho))
    #print(BG)

    # loss rate of HCHO

    # loss rate of Isop

    #
    #TODO: store GC_background for comparison
    GC_BG=np.array([np.NaN])

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
    kg_per_atom = util.isoprene_grams_per_mole * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * SA * kg_per_atom
    E_isop_kgs=E_new*conversion
    attrs["E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}


    return {'E_isop':E_new, 'E_isop_kg':E_isop_kgs,
            'lats':latsomi, 'lons':lonsomi, 'background':BGomi,
            'GC_background':GC_BG, 'GC_slope':GC_slope,
            'lati':latiomi,'loni':loniomi,
            'attributes':attrs}

def Emissions_series(day0=datetime(2005,1,1), dayn=datetime(2005,2,1), region=pp.__AUSREGION__):
    '''
        Emissions over time
    '''
    if __VERBOSE__: print("Running Inversion.Emissions_series()")
    ## Read data for these dates
    ##
    # TODO: update to use read_E_new_range()
    E_new=fio.read_E_new()



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

    # If just a day is input, then save a month
    if dayn is None:
        dayn=util.last_day(day0)
    days=util.list_days(day0,dayn)

    d0str=day0.strftime("%Y%m%d")
    dnstr=dayn.strftime("%Y%m%d")
    ddir="Data/Isop/E_new"
    fname=ddir+"/emissions_%s-%s.h5"%(d0str,dnstr)
    if __VERBOSE__:
        print("Calculating %s-%s estimated emissions over %s to file %s"%(d0str,dnstr,str(region),fname))

    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=ignorePP)
    if GC is None:
        GC=GC_output(date=day0)

    Emiss=[]
    # Read each day then save the month
    #
    priorday=days[0]
    for day in days:
        # Read GC month if necessary:
        if day.month != priorday.month:
            GC=GC_output(date=day)
        # Get a day of new emissions estimates
        Emiss.append(Emissions_1day(day=day, GC=GC, OMI=OMI, region=region))
        priorday=day

    outattrs=Emiss[0]['attributes'] # get one copy of attributes is required

    # Adding time dimension (needs to be utf8 for h5 files)
    #dates = np.array([d.strftime("%Y%m%d").encode('utf8') for d in days])
    #outattrs["time"]={"format":"%Y%m%d", "desc":"year month day string"}
    dates = np.array([int(d.strftime("%Y%m%d")) for d in days])
    outdata={"time":dates}
    outattrs["time"]={"format":"%Y%m%d", "desc":"year month day as integer (YYYYMMDD)"}
    fattrs={'region':"SWNE: %s"%str(region)}
    fattrs['date range']="%s to %s"%(d0str,dnstr)

    # Save lat,lon
    outdata['lats']=Emiss[0]['lats']
    outdata['lons']=Emiss[0]['lons']
    outdata['lats_e']=util.edges_from_mids(outdata['lats'])
    outdata['lons_e']=util.edges_from_mids(outdata['lons'])

    # Save data into month of daily averages
    # TODO: keep OMI counts from earlier...
    keys_to_save=['E_isop', 'E_isop_kg','background',
                  'GC_isop', 'GC_isop_kg', 'GC_background',
                  'GC_slope']
    #if not ignorePP: keys_to_save.append("") save PP based new emissions also..
    for key in keys_to_save:
        outdata[key]=np.array([E[key] for E in Emiss]) # time will be first dim

    # Save file, with attributes
    fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)

if __name__=='__main__':
    print('Inversion has been run')

    store_emissions()