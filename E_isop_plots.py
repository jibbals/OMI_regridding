#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:25:42 2017

Plot various things and print diagnostics on Isoprene outputs

@author: jesse
"""
###############
### Modules ###
###############
# python modules
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')# don't plot on screen, send straight to file
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from classes.GC_class import GC_output # GC trac_avg class
from classes.omhchorp import omhchorp # OMI product class
import Inversion

###############
### Globals ###
###############
__VERBOSE__=True

###############
### Methods ###
###############

def map_E_gc(month, GC, clims=[1e11,5e12], region=pp.__AUSREGION__,
             cmapname='PuBuGn',linear=True, contourf=False):
    #GEOS-Chem over our region:
    E_GC_sub=GC.get_field(keys=['E_isop_bio'], region=region)
    Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values
    lats_e=E_GC_sub['lats_e']
    lons_e=E_GC_sub['lons_e']

    title=r'E$_{GC}$ %s'%month.strftime("%Y, %b")
    # We need to specify the edges since GC is not fully regular
    pp.createmap(Egc, lats_e, lons_e, edges=True, title=title,
                 vmin=clims[0], vmax=clims[1], contourf=contourf,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$',
                 cmapname=cmapname, linear=linear, region=region)

def map_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
              contourf=False, linear=True,
              clims=[2e11,2e12], region=pp.__AUSREGION__,
              cmapname='PuBuGn', pname=None):
    '''
        Plot calculated emissions
    '''
    day0=month
    dstr=month.strftime("%Y %b") # YYYY Mon
    dayn=util.last_day(day0)
    em=Inversion.Emissions(day0=day0, dayn=dayn, GC=GC, OMI=OMI, region=region)
    E=em['E_isop']
    lats=em['lats']; lons=em['lons']

    title=r'E$_{isop}$ %s'%dstr

    pp.createmap(E, lats, lons, edges=False, title=title, pname=pname,
                 contourf=contourf,
                 vmin=clims[0], vmax=clims[1], linear=linear, aus=True,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$', cmapname=cmapname)


def E_gc_VS_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
                  ReduceOmiRes=0, contourf=False,
                  region=pp.__AUSREGION__):
    '''
        Plot E_gc, E_new, diff, ratio
    '''
    dstr=month.strftime("%Y%m")
    yyyymon=month.strftime("%Y, %b")
    day0=month; dayn=util.last_day(month)
    if __VERBOSE__:
        print("running E_isop_plots.E_gc_VS_E_new from %s and %s"%(day0, dayn))

    ## READ DATA
    if GC is None:
        GC=GC_output(date=month)
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn,ignorePP=True)

    ## Inversion
    # based on OMI using GC calculated yield (slope)
    E_new=Inversion.Emissions(day0=day0, dayn=dayn, GC=GC, OMI=OMI,
                              ReduceOmiRes=ReduceOmiRes, region=region)
    newE=E_new['E_isop']
    omilats=E_new['lats']
    omilons=E_new['lons']

    # GEOS-Chem over our region:
    E_GC_sub=GC.get_field(keys=['E_isop_bio'], region=region)
    Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values
    latsgc=E_GC_sub['lats']
    lonsgc=E_GC_sub['lons']

    # map the lower resolution data onto the higher resolution data:
    Egc_up=Egc
    if len(omilats) > len(latsgc):
        Egc_up = util.regrid(Egc,latsgc,lonsgc,omilats,omilons)


    ## First plot maps of emissions:
    ##
    plt.figure(figsize=(16,12))
    vlinear=True # linear flag for plot functions
    clims=[2e11,2.5e12] # map min and max
    amin,amax=-1e12, 3e12 # absolute diff min and max
    rmin,rmax=0, 10 # ratio min and max

    # start with E_GC:
    plt.subplot(221)
    map_E_gc(month=month, GC=GC, clims=clims, region=region,
             linear=vlinear, contourf=contourf)

    # then E_new
    plt.subplot(222)
    map_E_new(month=month,GC=GC,OMI=OMI,clims=clims,
              region=region, linear=vlinear, contourf=contourf)

    ## Difference and ratio:
    ##

    ## Diff map:
    plt.subplot(223)
    title=r'E$_{GC} - $E$_{omi}$ '
    args={'region':region, 'clabel':r'atoms C cm$^{-2}$ s$^{-1}$',
          'linear':True, 'lats':omilats, 'lons':omilons,
          'contourf':contourf, 'title':title, 'cmapname':'cool',
          'vmin':amin, 'vmax':amax}
    pp.createmap(Egc_up - newE, **args)

    ## Ratio map:
    plt.subplot(224)
    args['title']=r"$E_{GC} / E_{omi}$"
    args['vmin']=rmin; args['vmax']=rmax
    args['clabel']="ratio"
    pp.createmap(Egc_up / newE, **args)


    # SAVE THE FIGURE:
    #
    suptitle='GEOS-Chem (gc) vs OMI for %s'%yyyymon
    plt.suptitle(suptitle)
    fname='Figs/GC/E_Comparison_%s%s%s.png'%(dstr,
                                             ['','_contourf'][contourf],
                                             ['','_lowres'][ReduceOmiRes>0])
    plt.savefig(fname)
    print("SAVED FIGURE: %s"%fname)

    ## PRINT EXTRA INFO
    #
    if __VERBOSE__:
        print("GC calculations:")
        for k,v in E_GC_sub.items():
            print ("    %s, %s, mean=%.3e"%(k, str(v.shape), np.nanmean(v)))
        print("OMI calculations:")
        for k,v in E_new.items():
            print ("    %s, %s, mean=%.3e"%(k, str(v.shape), np.nanmean(v)))

    # Print some diagnostics here.
    for l,e in zip(['E_new','E_gc'],[newE,Egc]):
        print("%s: %s    (%s)"%(l,str(e.shape),dstr))
        print("    Has %d nans"%np.sum(np.isnan(e)))
        print("    Has %d negatives"%np.sum(e<0))

    ## Get the non-negative version of our new emissions estimate:
    newE_nn = np.copy(newE)
    newE_nn[newE_nn < 0] = 0.0 # np.NaN makes average too high

    # Print the average estimates:
    print("New estimate: %.2e"%np.nanmean(newE))
    print("Old estimate: %.2e"%np.nanmean(Egc))
    print("New estimate (no negatives): %.2e"%np.nanmean(newE_nn))
    #print("New estimate (low resolution): %.2e"%np.nanmean(E_new_lowres['E_isop']))
    # corellation

    #Convert both arrays to same dimensions for correllation?
    #pp.plot_corellation()


def All_maps(month=datetime(2005,1,1), ignorePP=True, region=pp.__AUSREGION__):
    '''
        Plot Emissions from OMI over region for averaged month
    '''
    dstr=month.strftime("%Y%m")   # YYYYMM
    if __VERBOSE__: print("Running E_isop_plots.All_E_omi() on %s"%dstr)

    day0=month
    dayn=util.last_day(day0)

    ## Read data
    ##
    GC=GC_output(date=day0) # gets one month of GC.
    OMI=omhchorp(day0=day0,dayn=dayn,ignorePP=ignorePP)

    ## Plot E_new
    ##
    clims=[2e11,2e12]
    cmapname='YlGnBu'
    pname='Figs/GC/E_new_%s.png'%dstr
    plt.figure(figsize=[10,8])
    map_E_new(month=month, GC=GC, OMI=OMI, clims=clims, cmapname=cmapname)
    plt.savefig(pname)

    ## Plot E_GC vs E_new, low and high res.
    ##
    for ReduceOmiRes in [0,8]:
        for contourf in [False, True]:
            E_gc_VS_E_new(month=month, GC=GC, OMI=OMI,
                          ReduceOmiRes=ReduceOmiRes, contourf=contourf)

if __name__=='__main__':

    # try running
    for month in [datetime(2005,1,1), datetime(2005,2,1)]:
        All_maps(month=month)

