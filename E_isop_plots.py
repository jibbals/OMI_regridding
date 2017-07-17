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
             right='purple', cmap='PuBuGn'):
    #GEOS-Chem over our region:
    E_GC_sub=GC.get_field(keys=['E_isop_bio'], region=region)
    Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values
    latsgc=E_GC_sub['lats']
    lonsgc=E_GC_sub['lons']

    title=r'E$_{GC}$ %s'%month.strftime("%Y, %b")
    pp.createmap(Egc, latsgc, lonsgc, title=title,
                 vmin=clims[0], vmax=clims[1],
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$',
                 cmap=cmap, right=right, left='white',
                 linear=False, region=region)

def map_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
              clims=[1e11,3e12], region=pp.__AUSREGION__,
              right='purple', cmap='PuBuGn'):
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

    pp.createmap(E,lats,lons, vmin=clims[0], vmax=clims[1], title=title,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$',cmap=cmap,
                 right=right, left='white',
                 linear=False, aus=True)
                 #pname=pname)


def E_gc_VS_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
                  ReduceOmiRes=0, smoothed=False,
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

    right='purple' # color for values over max
    vlims=[1e11,5e12] # map min and max
    amin,amax=-1e12, 3e12 # absolute diff min and max
    rmin,rmax=.5, 10 # ratio min and max

    ## First plot maps of emissions:
    ##
    plt.figure(figsize=(16,12))


    # start with E_GC:
    plt.subplot(221)
    map_E_gc(month=month, GC=GC, clims=vlims, region=region, right=right)

    # then E_new
    plt.subplot(222)
    map_E_new(month=month,GC=GC,OMI=OMI,clims=vlims,region=region, right=right)

    ## Difference and ratio:
    ##

    # map the lower resolution data onto the higher resolution data:
    Egc_up=Egc
    if len(omilats) > len(latsgc):
        Egc_up = util.regrid(Egc,latsgc,lonsgc,omilats,omilons)

    ## Diff map:
    plt.subplot(223)
    title=r'E$_{GC} - $E$_{omi}$ '
    args={'region':region, 'clabel':r'atoms C cm$^{-2}$ s$^{-1}$',
          'linear':True, 'lats':omilats, 'lons':omilons,
          'contourf':smoothed, 'title':title, 'cmap':'YlGnBu',
          'vmin':amin, 'vmax':amax, 'right':right}
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
                                             ['','_smoothed'][smoothed],
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


def All_maps(month=datetime(2005,1,1), clims=[1e12,3e12],
             ignorePP=True, region=pp.__AUSREGION__):
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
    pname='Figs/GC/E_new_%s.png'%dstr
    plt.figure(figsize=[10,8])
    map_E_new(month=month, GC=GC, OMI=OMI)
    plt.savefig(pname)

    ## Plot E_GC vs E_new, low and high res.
    ##
    for ReduceOmiRes in [0,8]:
        E_gc_VS_E_new(month=month, GC=GC, OMI=OMI,
                  ReduceOmiRes=ReduceOmiRes, smoothed=False)

if __name__=='__main__':

    # try running
    for month in [datetime(2005,1,1), datetime(2005,2,1)]:
        All_maps(month=month)

