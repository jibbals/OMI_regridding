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
#from utilities import fio
from classes.GC_class import GC_tavg # GC trac_avg class
from classes.omhchorp import omhchorp # OMI product class
from classes.E_new import E_new # E_new class

import Inversion

###############
### Globals ###
###############
__VERBOSE__=True

NA     = GMAO.edges_containing_region([-21,115,-11,150])
SWA    = GMAO.edges_containing_region([-36,114,-29,128])
SEA    = GMAO.edges_containing_region([-39,144,-29,153])
subs   = [SWA,NA,SEA]
labels = ['SWA','NA','SEA']
colours = ['chartreuse','magenta','aqua']

###############
### Methods ###
###############
def check_E_new(d0=datetime(2005,1,1),dn=datetime(2005,12,1),region=pp.__AUSREGION__, plotswaths=False):
    '''
        Print out averages and anomalies in time series
    '''
    # Read data
    Enew=E_new(d0,dn)
    dates,E_isop=Enew.get_series('E_isop',region=region, testplot=True)

    negs=np.where(E_isop<0)[0]
    highs=np.where(E_isop>6e11)[0]
    print("Negative emissions, date")
    pargs={'vmin':-1e16,'vmax':2e16,'clabel':'molec/cm2'}
    for i in negs:
        neg=dates[i]
        negstr=neg.strftime("%Y%m%d")
        print("%.4e    , %s"%(E_isop[i],negstr))
        if plotswaths:
            title="OMI Swath %s (E_new=%.4e)"%(negstr,E_isop[i])
            #plot the maps:
            pp.plot_swath(neg,title=title,
                          pname="Figs/Checks/swath_%s.png"%negstr,
                          **pargs)
    print("Super high emissions, date")
    for i in highs:
        high=dates[i]
        highstr=high.strftime("%Y%m%d")
        print("%.4e    , %s"%(E_isop[i],str(dates[i])))
        if plotswaths:
            title="OMI Swath %s (E_new=%.4e)"%(highstr,E_isop[i])
            # plot the maps:
            pp.plot_swath(high,title=title,
                          pname="Figs/Checks/swath_%s.png"%highstr,
                          **pargs)


def E_new_time_series(d0=datetime(2005,1,1),dn=datetime(2005,12,1),
                      drawmap=True, pname='Figs/E_new_series.png'):
    '''
        Plot the time series of E_new, eventually compare against MEGAN, etc..
    '''
    # Regions I'm averaging over:
    #Aus=[-40,112,-11,153]



    linewidths=[2,2,2]

    # Draw them if you want
    if drawmap:
        plt.figure()
        region=pp.__AUSREGION__

        pp.displaymap(region=region, subregions=subs,
                      labels=labels, linewidths=linewidths, colors=colours)

        regionname='Figs/regionmap.png'
        plt.title("Regions for E_isop analysis")
        plt.savefig(regionname)
        print("Saved %s"%regionname)
        plt.close()

    pp.InitMatplotlib() # set up plotting defaults
    f,ax=plt.subplots(1,3,figsize=(16,7),sharex=True,sharey=True,squeeze=True)

    # Read data
    Enew=E_new(d0,dn,dkeys=['E_isop','GC_E_isop'])

    for i,rc in enumerate(zip(subs,colours)):
        plt.sca(ax[i])  # ax.append(plt.subplot(131+i))
        region,colour=rc
        ptsargs={'dfmt':"%b",'color':'k'}

        dates, E_isop=Enew.get_series('E_isop',region=region)
        dates, GC_E_isop=Enew.get_series('GC_E_isop',region=region)
        units=Enew.attributes['E_isop']['units']

        # Plot time series (dots with colour of region)
        pp.plot_time_series(dates,E_isop,
                            linestyle='None', marker='.',
                            label=[None,'daily estimate'][i==1],
                            color=colour)

        # Add monthly average line
        monthly=util.monthly_averaged(dates,E_isop)
        GC_monthly=util.monthly_averaged(dates,GC_E_isop)
        GC_E_monthly=GC_monthly['data']
        GC_mstd=GC_monthly['std']
        mdates=monthly['middates']; E_monthly=monthly['data']
        xticks=mdates[0::2]
        mstd=monthly['std'];

        print(region)
        print(monthly)
        print(GC_E_monthly)
        
        # Plot monthly average and std:
        pp.plot_time_series(mdates,E_monthly, linewidth=2.0,
                            label=[None,'monthly avg.'][i==1],
                            **ptsargs)

        # add +- 1 std.
        pp.plot_time_series(mdates,E_monthly+mstd, linestyle='--',
                            label=[None,'1 std.'][i==1],
                            **ptsargs)
        pp.plot_time_series(mdates,E_monthly-mstd, linestyle='--',
                            **ptsargs)

        # Plot E_isop average
        pp.plot_time_series(mdates,GC_E_monthly,linestyle=':',xticks=xticks,
                            markeredgewidth=3, marker='x', linewidth=2,
                            label=[None,'monthly avg. (MEGAN)'][i==1],
                            **ptsargs)


        plt.ylim([-0.5e12, 3e12])
        plt.title(labels[i])
        if i==1: plt.legend(loc='best',prop={'size': 10})

    ax[0].set_ylabel("E_isop [%s]"%units)
    plt.suptitle("Emissions over 2005")




    # save figure
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
    pname='Figs/E_MEGAN_%s.png'%month.strftime("%Y%m")
    # We need to specify the edges since GC is not fully regular
    pp.createmap(Egc, lats_e, lons_e, edges=True, title=title,
                 vmin=clims[0], vmax=clims[1], smoothed=smoothed,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$',
                 cmapname=cmapname, linear=linear, region=region,
                 pname=pname)

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
        GC=GC_tavg(date=month)
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
    #pp.plot_regression()


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
        GC=GC_tavg(date=month)
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
        GC=GC_tavg(date=month)
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

    # Get some pastel shades for the colours
    colours = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colours[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x/1000.0) for x in y_offset])
    # Reverse colours and text labels to display the last value at the top.
    colours = colours[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colours,
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

def megan_monthly_regression():
    '''
    monthly regression of my product vs MEGAN E_isop_biog
    '''
    d0=datetime(2005,1,1)
    dn=datetime(2005,12,1)
    # do each month
    for month in util.list_months(d0,dn):
        d0=month; dn=util.last_day(d0)
        dstr=month.strftime('%b, %Y')
        ymd=month.strftime('%Y%m%d')
        Enew=E_new(day0=d0,dayn=dn)
        f,axes=plt.subplots(1,3,figsize=(18,7),sharey=True,squeeze=True)
        ii=0
        ppargs={'legendfont':14}
        for reg,c in zip(subs,colours):
            plt.sca(axes[ii])
            ppargs['colour']=c
            ppargs['linecolour']=c
            Enew.plot_regression(d0,dn,region=reg,**ppargs)
            plt.title(labels[ii])

            ii=ii+1
            if ii==1:
                plt.ylabel('E_isop')
            if ii==2:
                plt.xlabel('MEGAN')
        plt.suptitle(dstr,fontsize=24)
        pname='Figs/Regression_'+ymd+'.png'
        plt.savefig(pname)
        print('Saved',pname)
        plt.close()

def megan_SEA_regression():
    '''
    Close look at SEA region vs MEGAN
    '''
    ds0=datetime(2005,1,1); ds1=util.last_day(datetime(2005,2,1))
    dw0=datetime(2005,6,1); dw1=datetime(2005,8,31)
    region=SEA
    # summer
    E_summer=E_new(ds0,ds1)
    E_winter=E_new(dw0,dw1)
    plt.figure(figsize=(14,14))#,sharex=True,squeeze=True)
    for ds in [True,False]:
        ppargs={'colour':'red','linecolour':'red','diag':False,'legend':False}
        E_summer.plot_regression(ds0,ds1,region=region,deseasonalise=ds,**ppargs)
        ppargs['colour']='aqua'
        ppargs['linecolour']='aqua'
        E_winter.plot_regression(dw0,dw1,region=region,deseasonalise=ds,**ppargs)
        plt.title('SEA Summer (JF) vs Winter (JJA), 2005')
        plt.legend(loc='best')
        plt.xlabel('MEGAN')
        plt.ylabel('Satellite based')
        pname='Figs/Regression_SEA_2005%s.png'%['','_DS'][ds]
        plt.savefig(pname)
        plt.close()
        print("Saved: ",pname)


if __name__=='__main__':

    # try running
    JennySEA=[-38, 145, -30, 153]
    JennySEA_fixed=GMAO.edges_containing_region(JennySEA) # [-37, 143.75,-29, 153.75]
    SEAus=[-41,138.75,-25,156.25]
    regions=pp.__AUSREGION__, SEAus, JennySEA_fixed

    d0=datetime(2005,1,1); dn=datetime(2005,12,31)
    megan_SEA_regression()
    E_new_time_series(d0,dn) # Takes a few minuts (use qsub)
    #map_E_gc(month=d0,GC=GC_tavg(d0))
    #check_E_new(dn=datetime(2005,2,1))

#    for region in regions:
#        print("REGION = %s"%str(region))
#
#        for month in [datetime(2005,1,1), datetime(2005,2,1)]:
#            # Read month of data
#            GC=GC_tavg(date=month)
#            OMI=omhchorp(day0=month,dayn=util.last_day(month),ignorePP=True)
#
#            # Run plots and print outputs
#            print_megan_comparison(month, GC=GC, OMI=OMI, region=region,)
#            All_maps(month=month, GC=GC, OMI=OMI, region=region)

