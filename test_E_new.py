#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:22:36 2018

    Run tests on E_new outputs

@author: jesse
"""

## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
from utilities import fio
from utilities import plotting as pp
from utilities.JesseRegression import RMA
from utilities import utilities as util

from classes.GC_class import GC_tavg, GC_sat
from classes.E_new import E_new

import numpy as np
from datetime import datetime #, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
from matplotlib.ticker import FormatStrFormatter # tick formats

#import matplotlib.patches as mpatches
import seaborn # kdeplots

import random # random number generation
import timeit

##############################
######### GLOBALS ############
##############################


Ohcho='VC$_{HCHO}$'
Ovc='VC'
Ovcgc='VC$_{GC}$'
Ovcc='VCC'
Ovccgc='VCC$_{GC}$'
Ovccpp='VCC$_{PP}$' # Paul Palmer VCC
Ovccomi='VCC$_{OMI}$'
Ovcomi='VC$_{OMI}$'
Ovccomi="VCC$_{OMI}$" #corrected original product
Ovcgc='VC$_{GC}$'
Ovcpp='VC$_{PP}$'


##############################
########## TESTS     #########
##############################

def Filter_affects(d0=datetime(2005,1,1),dn=None,region=pp.__AUSREGION__):
    '''
        Look at affects of filters to both VCC and E_new
    '''
    if dn is None:
        dn=util.last_day(d0)
    dstr="%s-%s"%(d0.strftime("%Y%m%d"),dn.strftime("%Y%m%d"))
    suptitles="%%s with and without filtering over %s"%dstr
    pnames="Figs/Emiss/filtered_%%s_%%s_%s.png"%dstr
    titles="%s"
    masktitles="masked by %s"

    Enew=E_new(d0,dn)
    dates=Enew.dates
    lats,lons=Enew.lats,Enew.lons
    masks=[mask.astype(np.bool) for mask in [Enew.firefilter,Enew.anthrofilter,Enew.firefilter+Enew.anthrofilter]]
    mask_names=['fire','anthro','fire+anthro']
    arr_names=['VCC_GC','VCC_OMI','E_VCC_GC','E_VCC_OMI']
    arr_vmins=[ 1e14   ,  1e14   ,  1e9     ,   1e9     ]
    arr_vmaxs=[ 1e16   ,  1e16   ,  5e12    ,   5e12    ]
    for arr_name,vmin,vmax in zip(arr_names,arr_vmins,arr_vmaxs):
        arr = getattr(Enew,arr_name)
        units= Enew.attributes[arr_name]['unit'] # TODO change to units when implemented
        for mask,mask_name in zip(masks,mask_names):
            #(data, dates, lats, lons, mask=None, subzones=__subzones_AUS__,
            #pname=None,title=None,suptitle=None, masktitle=None,
            #clabel=None, vmin=None, vmax=None, linear=False,
            #maskoceans=True,
            #colors=__subzones_colours__):
            title=titles%arr_name
            suptitle=suptitles%arr_name
            pname=pnames%(arr_name,mask_name)
            masktitle=masktitles%mask_name
            pp.subzones(arr,dates,lats,lons, mask=mask,
                        vmin=vmin,vmax=vmax,
                        title=title,suptitle=suptitle,masktitle=masktitle,
                        pname=pname,clabel=units)

def Summary_E_new(month=datetime(2005,1,1)):
    '''
        Check E_new and stdev and time series
    '''
    # Read month of data
    day0=datetime(month.year,month.month,1)
    dayn=util.last_day(month)
    Enew=E_new(day0,dayn)

    pp.createmap(np.nanmean(Enew.E_VCC_GC,axis=0),Enew.lats,Enew.lons,
                 region=pp.__AUSREGION__,
                 pname='test_Enew.png')


def VCC_comparison(month=datetime(2005,1,1),region=pp.__AUSREGION__):
    '''
        Plot columns with different amf bases
        Differences and Correlations
              |  VCC   |  VCC_gc   |  VCC_pp
       abs dif|
       rel dif|
       distrs |

    '''

    # start by reading all the VCC stuff
    # useful strings
    ymstr=month.strftime('%Y%m')
    d0=datetime(month.year,month.month,1)
    dN=util.last_day(month)
    pname='Figs/VCC_comparison_%s.png'%ymstr
    linear=True # linear colour scale?

    start_time=timeit.default_timer()
    # read in omhchorp
    om=omrp(d0,dayn=dN, keylist=['VCC_OMI','VCC_GC','VCC_PP','gridentries','ppentries'])
    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to read omhchorp"%elapsed)

    start_time2=timeit.default_timer()
    # Subset the data to our region
    subsets=util.lat_lon_subset(om.lats,om.lons,region,[om.VCC_OMI,om.VCC_GC,om.VCC_PP,om.gridentries,om.ppentries],has_time_dim=True)
    lats,lons=subsets['lats'],subsets['lons']
    VCC_OM,VCC_GC,VCC_PP,pix,pix_pp = subsets['data']

    elapsed = timeit.default_timer() - start_time2
    print("TIMEIT: Took %6.2f seconds to subset the VCC arrays"%elapsed)

    oceanmask=util.oceanmask(lats,lons)

    # firemask is 3dimensional: [days,lats,lons]
    fstart=timeit.default_timer()
    firemask,fdates,flats,flons=fio.make_fire_mask(d0,dN=dN,region=region) # use defaults
    smokemask,sdates,slats,slons=fio.make_smoke_mask(d0,dN=dN,region=region)
    anthromask,adates,alats,alons=fio.make_anthro_mask(d0,dN,region=region)
    fullmask=firemask+smokemask+anthromask
    felapsed = timeit.default_timer() - fstart
    print ("TIMEIT: Took %6.2f seconds to get fire,smoke, and anthro masks"%(felapsed))

    # Plot rows,cols,size:
    f=plt.figure(figsize=[18,18])

    # first line is maps of VCC, VC_GC, VCC_PP
    titles=[[Oomi,Ogc+" (S(z) updated)",Opp+" (S(z)+$\omega$(z) updated)"],
            [Ogc+'-'+Oomi,Opp+"-"+Ogc,Oomi+"-"+Opp],
            ['','','']
            ]
    maps = [[VCC_OM, VCC_GC, VCC_PP], # orig
            [VCC_GC-VCC_OM, VCC_PP-VCC_GC, VCC_OM-VCC_PP], # abs diff
            [100*(VCC_GC-VCC_OM)/VCC_OM, 100*(VCC_PP-VCC_GC)/VCC_GC, 100*(VCC_OM-VCC_PP)/VCC_PP] # rel diff
            ]
    vmins,vmaxs=[4e15,None,-120],[9e15,None,120] # min,max for colourbars
    cmapnames=['plasma','plasma','seismic']
    cbarlabels=['molec/cm2','molec/cm2','%']
    area_list=[]
    ts_list=[]
    for j in range(3): #col
        for i in range(3): #row
            plt.subplot(5,3,i*3+j+1)

            arr = maps[i][j]
            arr[fullmask]=np.NaN # Remove fire,smoke,anthro...
            arr[:,oceanmask]=np.NaN # nanify the ocean
            if i==0:
                ts_list.append(np.nanmean(arr,axis=(1,2)))
            arr=np.nanmean(arr,axis=0) # average over time
            if i == 0:
                area_list.append(arr) # save each map for distribution plot
            m,cs,cb= pp.createmap(arr,lats,lons,linear=True,
                                  region=region,
                                  vmin=vmins[i],vmax=vmaxs[i],
                                  cmapname=cmapnames[i],
                                  colorbar=j==1, # colorbar in middle column only
                                  clabel=cbarlabels[i])
            plt.title(titles[i][j])

            # add a little thing showing entries and mean and max
            # entries for normal or ppamf
            entries=np.copy([om.gridentries,om.ppentries][j==2])
            entries=entries.astype(np.float) # so I can Nan the ocean/non-aus areas
            entries=np.nansum(entries,axis=0) # how many entries
            entries[oceanmask]=np.NaN

            txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries),np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr), 'max = %.2e'%np.nanmax(arr)]
            for txt, yloc in zip(txt,[0.01,0.07,0.13]):
                plt.text(0.01, yloc, txt,
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=plt.gca().transAxes,
                     color='k', fontsize=10)

    # Add time series for each VCC
    plt.subplot(5,1,4)
    labels=[Oomi,Ogc,Opp]
    plt.plot(ts_list,label=labels)

    # Finally add density plots for each map
    plt.subplot(5,1,5)

    area_list= np.transpose([vec for vec in area_list]) # list of np arrays to array of vectors..
    plt.hist(area_list,bins=np.linspace(vmins[0], vmaxs[0], 20), label=labels)

    #
    #ticks=[np.logspace(np.log10(vmin),np.log10(vmax),5),np.linspace(vmin,vmax,5)][linear]
    #pp.add_colourbar(f,cs,ticks=ticks,label='molec/cm$^2$')
    #pp.add_colourbar(f2,cs2,ticks=np.linspace(vmin2,vmax2,5),label='pixels')

    f.savefig(pname)
    plt.close(f)
    print("Saved ",pname)


    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to run plot_VCC_rsc_gc_pp()"%elapsed)

    #createmap(data, lats, lons, make_edges=False, GC_shift=True,
    #          vmin=None, vmax=None, latlon=True,
    #          region=__GLOBALREGION__, aus=False, linear=False,
    #          clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
    #          cbarorient='bottom',
    #          pname=None,title=None,suptitle=None, smoothed=False,
    #          cmapname=None, fillcontinents=None):


if __name__ == '__main__':

    #Summary_E_new() # Not yet run?
    Filter_affects() # Run 23/5/18