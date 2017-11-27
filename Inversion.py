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


def Emissions_1day(day, GC_biog, OMI, region=pp.__AUSREGION__):
    '''
        Return one day of emissions estimates
            Uses one month of GEOS-Chem (GC) estimated 'slope' for Y_hcho
            uses one day of OMI HCHO
            Use biogenic run for GC_biog
    '''
    dstr=day.strftime("%Y%m%d")
    attrs={} # attributes dict for meta data
    GC=GC_biog.sat_out

    if __VERBOSE__:
        # Check the dims of our stuff
        print()
        print("Calculating emissions for %s"%dstr) 
        print("GC data %s "%str(GC.hcho.shape)) # [t,lat,lon,lev]
        print("nanmean:",np.nanmean(GC.hcho),GC.attrs['hcho']) # should be molecs/cm2
        print("OMI data %s"%str(OMI.VCC.shape)) # [t, lat, lon]
        print("nanmean:",np.nanmean(OMI.VCC),'molecs/cm2')# should be molecs/cm2

    omilats0, omilons0 = OMI.lats, OMI.lons
    omi_lats, omi_lons= omilats0.copy(), omilons0.copy()
    omi_SA=OMI.surface_areas # in km^2

    # Get GC_isoprene for this day also
    GC_days, GC_E_isop = GC_biog.hemco.daily_LT_averaged(hour=13)

    #GC_E_isop=GC.get_field(keys=['E_isop_bio',],region=region)['E_isop_bio']
    if __DEBUG__:
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
        GC=GC_sat(date=day0)
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
    kg_per_atom = util.__grams_per_mole__['isop'] * 1.0/N_avegadro * 1e-3
    conversion= 1./5.0 * 1e10 * SA * kg_per_atom
    E_isop_kgs=E_new*conversion
    attrs["E_isop_kg"]={"units":"kg/s",
        "desc":"emissions/cm2 multiplied by area"}


    return {'E_isop':E_new, 'E_isop_kg':E_isop_kgs,
            'lats':latsomi, 'lons':lonsomi, 'background':BGomi,
            'GC_background':GC_BG, 'GC_slope':GC_slope,
            'lati':latiomi,'loni':loniomi,
            'attributes':attrs}

def store_emissions_month(month=datetime(2005,1,1), GC=None, OMI=None,
                          region=pp.__GLOBALREGION__, ignorePP=True):
    '''
        Store a month of new emissions estimates into an he5 file
    '''
    day0=datetime(month.year,month.month,1)

    # Get last day in month:
    dayn=util.last_day(day0)
    days=util.list_days(day0,dayn)

    mstr=dayn.strftime('%Y%m')
    d0str=day0.strftime("%Y%m%d")
    dnstr=dayn.strftime("%Y%m%d")
    ddir="Data/Isop/E_new/"
    fname=ddir+"emissions_%s.h5"%(mstr)
    if __VERBOSE__:
        print("Calculating %s-%s estimated emissions over %s"%(d0str,dnstr,str(region)))
        print("will save to file %s"%(fname))

    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn, ignorePP=ignorePP)
    if GC is None:
        GC=GC_class.GC_biogenic(day0) # data like [time,lat,lon,lev]

    # Read each day then save the month
    #
    Emiss=[]
    for day in days:
        Emiss.append(Emissions_1day(day=day, GC_biog=GC, OMI=OMI, region=region))

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

    outdata['smearing']=smearing(month,plot=True,region=region)
    outattrs['smearing']={'desc':'smearing for %s'%mstr}
    # Save data into month of daily averages
    # TODO: keep OMI counts from earlier...
    keys_to_save=['E_isop', 'E_isop_kg','background',
                  'GC_E_isop', 'GC_E_isop_kg', 'GC_background',
                  'GC_slope']
    #if not ignorePP: keys_to_save.append("") save PP based new emissions also..
    for key in keys_to_save:
        outdata[key]=np.array([E[key] for E in Emiss]) # time will be first dim

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
        print('mean HCHO column from omhchorp',np.nanmean(OMI.VCC))
        
        # Read GC month:
        GC=GC_class.GC_biogenic(month)
        print('mean surface hcho from GC_biogenic run:',np.nanmean(GC.sat_out.hcho[:,:,:,0]))
        
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
