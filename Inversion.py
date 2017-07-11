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

# local imports
from utilities.JesseRegression import RMA
from classes.GC_class import GC_output # Class reading GC output
from classes.omhchorp import omhchorp # class reading OMHCHORP
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

def Emissions(day0=datetime(2005,1,1), dayn=datetime(2005,2,1), GC = None, OMI = None, region=pp.__AUSREGION__, ReduceOmiRes=0):
    '''
        Determine emissions of isoprene averaged over some length of time.
        1) Calculates model slope E_isop -> Tropcol_HCHO
        2) Use Biogenic OMI HCHO product to calculate E_isoprene.
        Notes:
            Up to 1 month at a time, OMI is averaged over day0 -> dayn-1

        HCHO = S * E_isop + b
    '''
    ## Read data for this date unless it's been passed in
    ##
    if GC is None:
        GC=GC_output(date=day0)
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn)

    if __VERBOSE__:
        # Check the dims of our stuff
        print("GC data %s"%str(GC.hcho.shape))
        print("Lats from %.2f to %.2f"%(GC.lats[0],GC.lats[-1]))
        print("Lons from %.2f to %.2f"%(GC.lons[0],GC.lons[-1]))
        print("OMI data %s"%str(OMI.VCC.shape))
        print("Lats from %.2f to %.2f"%(OMI.lats[0],OMI.lats[-1]))
        print("Lons from %.2f to %.2f"%(OMI.lons[0],OMI.lons[-1]))

    # model slope between HCHO and E_isop:
    S_model=GC.model_slope(region)

    # get OMI corrected vertical columns, averaged over time
    hcho=OMI.time_averaged(day0=day0,dayn=dayn,keys=['VCC'])['VCC']

    lats,lons=OMI.lats,OMI.lons
    if ReduceOmiRes > 0 :
        omilow=OMI.lower_resolution('VCC',factor=ReduceOmiRes,dates=[day0,dayn])
        hcho=omilow['VCC']
        lats,lons=omilow['lats'], omilow['lons']

    # subset over region of interest
    lati, loni = util.lat_lon_range(lats, lons, region)
    #inds = OMI.region_subset(region=region, maskocean=False, maskland=False)
    lats, lons = lats[lati], lons[loni]
    if __VERBOSE__:
        print('lats, lons in region:')
        print((lats,lons))
        print(type(lati))
        print(lati)
        print(type(hcho))
        print(hcho.shape)
    hcho    = hcho[lati,:]
    hcho    = hcho[:,loni]
    # Determine background using region latitude bounds
    BG      = OMI.background_HCHO(lats=[region[0],region[2]])


    ## Calculate Emissions from these
    ##

    # map GC slope onto same lats/lons as OMI
    print("slope shape, mean, and hcho shape:")
    print(np.shape(S_model['slope']))
    print(np.nanmean(S_model['slope']))
    print(np.shape(hcho))

    GC_slope=util.regrid(S_model['slope'], S_model['lats'],S_model['lons'],
                    lats,lons)

    print("slope shape and mean after regrid:")
    print (np.shape(GC_slope))
    print(np.nanmean(GC_slope))

    # \Omega_{HCHO} = S \times E_{isop} + B
    # E_isop = (Column_hcho - B) / S
    #   Determine S from the slope bewteen E_isop and HCHO
    E_new = (hcho - BG) / GC_slope

    #print (np.nanmean(hcho))
    #print(BG)

    # loss rate of HCHO

    # loss rate of Isop

    #
    #TODO: store GC_background for comparison
    GC_BG=np.NaN
    return {'E_isop':E_new, 'lats':lats, 'lons':lons, 'background':BG,
            'GC_background':GC_BG, 'GC_slope':GC_slope}

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

def check_regridding():
    #TODO: implement
    print('check_regridding TODO')

def check_against_MEGAN(month=datetime(2005,1,1)):
    dstr=month.strftime("%Y%m%d")
    GC=GC_output(date=month)
    OMI=omhchorp(day0=month,dayn=util.last_day(month),ignorePP=True)
    region=pp.__AUSREGION__

    E_new=Emissions(day0=month, GC=GC, OMI=OMI, region=region)
    E_new_lowres=Emissions(day0=month,GC=GC, OMI=OMI, region=region, ReduceOmiRes=8)
    E_GC=GC.get_field(keys=['E_isop'], region=region)
    E_GC['E_isop'] = np.mean(E_GC['E_isop'],axis=2) # average of the monthly values
    for k,v in E_GC.items():
        print ((k, v.shape))
        print(np.nanmean(v))

    Eomi=E_new['E_isop']
    Egc=E_GC['E_isop']
    latsgc=E_GC['lats']
    lonsgc=E_GC['lons']
    latsom=E_new['lats']
    lonsom=E_new['lons']
    for l,e in zip(['E_omi','E_gc'],[Eomi,Egc]):
        print("%s: %s"%(l,str(e.shape)))
        print("Has %d nans"%np.sum(np.isnan(e)))
        print("Has %d negatives"%np.sum(e<0))

    vmin,vmax=1e9,5e12 # map min and max
    amin,amax=-1e12, 5e12 # absolute diff min and max
    rmin,rmax=10, 1000 # relative difference min and max
    #pp.createmap(Eomi, latsom, lonsom, aus=True, vmin=vmin,vmax=vmax,
    #             linear=True, pname="Figs/GC/E_new_200501.png")
    #pp.createmap(Egc, latsgc, lonsgc, aus=True, vmin=vmin,vmax=vmax,
    #             linear=True, pname="Figs/GC/E_GC_200501.png")

    Eomi_nn = np.copy(Eomi)
    Eomi_nn[Eomi_nn < 0] = 0.0 # np.NaN makes average too high
    arrs=[Egc, Eomi_nn]
    lats=[latsgc, latsom]
    lons=[lonsgc, lonsom]
    titles=['E_gc', 'E_omi']
    suptitle='GEOS-Chem (gc) vs OMI: 1st, Jan, 2005'
    fname='Figs/GC/E_Comparison_%s.png'%dstr
    pp.compare_maps(arrs, lats, lons, pname=fname,
                    suptitle=suptitle,titles=titles,
                    vmin=vmin,vmax=vmax,linear=False,rlinear=False,
                    amin=amin,amax=amax,rmin=rmin,rmax=rmax)
    # again with degraded omi emissions:
    arrs[1]=E_new_lowres['E_isop']
    lats[1],lons[1]=E_new_lowres['lats'],E_new_lowres['lats']
    fname='Figs/GC/E_Comparison_lowres_%s.png'%dstr
    pp.compare_maps(arrs,lats,lons,pname=fname,
                    suptitle=suptitle, titles=titles,
                    vmin=vmin, vmax=vmax, linear=False, rlinear=False,
                    amin=amin, amax=amax, rmin=rmin, rmax=rmax)

    print("New estimate: %.2e"%np.nanmean(Eomi))
    print("Old estimate: %.2e"%np.nanmean(Egc))
    print("New estimate (no negatives): %.2e"%np.nanmean(Eomi_nn))
    print("New estimate (low resolution): %.2e"%np.nanmean(E_new_lowres['E_isop']))
    # corellation

    #Convert both arrays to same dimensions for correllation?
    #pp.plot_corellation()


if __name__=='__main__':
    # check the regridding function:
    check_regridding()
    # try running
    month=datetime(2005,1,1)
    check_against_MEGAN(month)
    month=datetime(2005,2,1)
    check_against_MEGAN(month)
