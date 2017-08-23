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
from utilities import GMAO
from utilities import fio
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

def E_new_time_series(region=pp.__AUSREGION__):
    '''
        Plot the time series of E_new, eventually compare against MEGAN, etc..
    '''

    # Read data, attributes
    #date=datetime(2005,1,1)
    E,Ea=fio.read_E_new()
    lats=E['lats']; lons=E['lons']
    dates=E['dates']
    dnums = matplotlib.dates.date2num(dates)
    E_new=E['E_isop']
    units=Ea['E_isop']['units']

    # Subset to region
    lati,loni=util.lat_lon_range(lats,lons,region)
    E_new=E_new[:,lati,:]
    E_new=E_new[:,:,loni]

    # average down to time series
    E_new=np.nanmean(E_new,axis=(1,2)) # mean over lat/lon

    # Plot time series
    print(dnums)
    print(E_new)
    plt.plot_date(dnums,E_new)
    plt.title('E_new time series over %s'%str(region))
    # set up better labels
    plt.ylabel("E_isop [%s]"%units)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b %y'))


    # save figure
    pname='Figs/E_new_series.png'
    plt.savefig(pname)
    print("Saved %s"%pname)

def map_E_gc(month, GC, clims=[1e11,5e12], region=pp.__AUSREGION__,
             cmapname='PuBuGn',linear=True, smoothed=False):
    #GEOS-Chem over our region:
    E_GC_sub=GC.get_field(keys=['E_isop_bio'], region=region)
    Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values
    lats_e=E_GC_sub['lats_e']
    lons_e=E_GC_sub['lons_e']

    title=r'E$_{GC}$ %s'%month.strftime("%Y, %b")
    # We need to specify the edges since GC is not fully regular
    pp.createmap(Egc, lats_e, lons_e, edges=True, title=title,
                 vmin=clims[0], vmax=clims[1], smoothed=smoothed,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$',
                 cmapname=cmapname, linear=linear, region=region)

def map_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
              smoothed=False, linear=True,
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
                 smoothed=smoothed,
                 vmin=clims[0], vmax=clims[1], linear=linear, aus=True,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$', cmapname=cmapname)


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
             linear=vlinear, smoothed=smoothed)

    # then E_new
    plt.subplot(222)
    map_E_new(month=month,GC=GC,OMI=OMI,clims=clims,
              region=region, linear=vlinear, smoothed=smoothed)

    ## Difference and ratio:
    ##

    ## Diff map:
    plt.subplot(223)
    title=r'E$_{GC} - $E$_{omi}$ '
    args={'region':region, 'clabel':r'atoms C cm$^{-2}$ s$^{-1}$',
          'linear':True, 'lats':omilats, 'lons':omilons,
          'smoothed':smoothed, 'title':title, 'cmapname':'cool',
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
                                             ['','_smoothed'][smoothed],
                                             ['','_lowres'][ReduceOmiRes>0])
    plt.savefig(fname)
    print("SAVED FIGURE: %s"%fname)
    plt.close()

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


    # corellation

    #Convert both arrays to same dimensions for correllation?
    #pp.plot_corellation()


def All_maps(month=datetime(2005,1,1),GC=None, OMI=None, ignorePP=True, region=pp.__AUSREGION__):
    '''
        Plot Emissions from OMI over region for averaged month
    '''
    dstr=month.strftime("%Y%m")   # YYYYMM
    if __VERBOSE__: print("Running E_isop_plots.All_E_omi() on %s"%dstr)

    day0=month
    dayn=util.last_day(day0)

    ## Read data
    ##
    ## READ DATA
    if GC is None:
        GC=GC_output(date=month)
    if OMI is None:
        OMI=omhchorp(day0=day0,dayn=dayn,ignorePP=ignorePP)

    ## Plot E_new
    ##
    clims=[2e11,2e12]
    cmapname='YlGnBu'
    for smoothed in [True,False]:
        pname='Figs/GC/E_new_%s%s.png'%(dstr,['','_smoothed'][smoothed])
        map_E_new(month=month, GC=GC, OMI=OMI, clims=clims, cmapname=cmapname,
                  smoothed=smoothed,pname=pname)

    ## Plot E_GC vs E_new, low and high res.
    ##
    for ReduceOmiRes in [0,8]:
        for smoothed in [True, False]:
            E_gc_VS_E_new(month=month, GC=GC, OMI=OMI,
                          ReduceOmiRes=ReduceOmiRes, smoothed=smoothed)

def print_megan_comparison(month=datetime(2005,1,1), GC=None, OMI=None,
                           ReduceOmiRes=0, region=pp.__AUSREGION__):
    ''' look at number differences between E_new and MEGAN output from GEOS_Chem'''
    dstr=month.strftime("%Y%m")
    yyyymon=month.strftime("%Y, %b")
    day0=month; dayn=util.last_day(month)
    if __VERBOSE__:
        print("running Inversion.print_megan_comparison for %s"%(yyyymon))

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
    E_isop_kgs=E_new['E_isop_kg_per_second']
    omilats=E_new['lats']
    omilons=E_new['lons']

    # GEOS-Chem over our region:
    E_GC_sub=GC.get_field(keys=['E_isop_bio','E_isop_bio_kgs'], region=region)
    Egc_kg=np.mean(E_GC_sub['E_isop_bio_kgs'], axis=0)
    Egc = np.mean(E_GC_sub['E_isop_bio'],axis=0) # average of the monthly values

    ## Get the non-negative version of our new emissions estimate:
    newE_nn = np.copy(newE)
    E_isop_kgs_nn=np.copy(E_isop_kgs)
    # lets ignore that nans don't compare to numbers, nan<0 gives false anyway
    with np.errstate(divide='ignore', invalid='ignore'):
        newE_nn[newE_nn < 0] = 0.0 # np.NaN makes average too high
        E_isop_kgs_nn[E_isop_kgs_nn<0]=0.0

    # Print the average estimates:
    print("For %s, in %s"%(str(region),yyyymon))
    print("   units     | E_New     | E_MEGAN ")
    print("atom C/cm2/s | %.2e  | %.2e "%(np.nanmean(newE),np.nanmean(Egc)))
    print("   isop kg/s | %.2e  | %.2e "%(np.nansum(E_isop_kgs),np.nansum(Egc_kg)))
    print("---Then with negatives set to zero---")
    print("atom C/cm2/s | %.2e  |  "%(np.nanmean(newE_nn)))
    print("   isop kg/s | %.2e  |  "%(np.nansum(E_isop_kgs_nn)))

    #print("New estimate (low resolution): %.2e"%np.nanmean(E_new_lowres['E_isop']))

def plot_comparison_table():
    """
        Currently: Demo of table function to display a table within a plot.
    """
    data = [[  66386,  174296,   75131,  577908,   32015],
            [  58230,  381139,   78045,   99308,  160454],
            [  89135,   80552,  152558,  497981,  603535],
            [  78415,   81858,  150656,  193263,   69638],
            [ 139361,  331509,  343164,  781380,   52269]]

    columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x/1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    pname='Figs/GC/Table.png'
    plt.savefig(pname)
    print('SAVED FIGURE %s'%pname)
    plt.close()

if __name__=='__main__':

    # try running
    JennySEA=[-38, 145, -30, 153]
    JennySEA_fixed=GMAO.edges_containing_region(JennySEA) # [-37, 143.75,-29, 153.75]
    SEAus=[-41,138.75,-25,156.25]
    regions=pp.__AUSREGION__, SEAus, JennySEA_fixed

    E_new_time_series(region=pp.__AUSREGION__)

#    for region in regions:
#        print("REGION = %s"%str(region))
#
#        for month in [datetime(2005,1,1), datetime(2005,2,1)]:
#            # Read month of data
#            GC=GC_output(date=month)
#            OMI=omhchorp(day0=month,dayn=util.last_day(month),ignorePP=True)
#
#            # Run plots and print outputs
#            print_megan_comparison(month, GC=GC, OMI=OMI, region=region,)
#            All_maps(month=month, GC=GC, OMI=OMI, region=region)

