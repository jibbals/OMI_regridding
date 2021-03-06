#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:35:08 2019

Plots shown in chapter 2 should be here, or moved here eventually

@author: jesse
"""

#################
### IMPORTS
################

# plotting libraries
import matplotlib
matplotlib.use('Agg')  # don't show plots interactively
import matplotlib.pyplot as plt
plt.ioff() # plot_date fix

from datetime import datetime, timedelta
import numpy as np

# local modules
from utilities import GMAO,GC_fio,fio, masks
from utilities import utilities as util
from utilities import plotting as pp
from utilities.JesseRegression import RMA, OLS

import Inversion
import tests
from tests import utilities_tests, test_new_emissions
import reprocess
import new_emissions
import Analyse_E_isop

from classes.E_new import E_new # E_new class
from classes import GC_class, campaign
from classes.omhchorp import omhchorp


import xbpch
import xarray
import pandas as pd
import seaborn as sns

import warnings
import timeit


###############
### Globals ###
###############
__VERBOSE__=True

## LABELS
# total column HCHO from GEOS-Chem

__Ogc__ = "$\Omega_{GC}$"
__Ogca__ = "$\Omega_{GC}^{\\alpha}$" # need to double escape the alpha for numpy plots for some reason
__Ogc__units__ = 'molec cm$^{-2}$'

# total column HCHO from OMI (recalculated using PP code)
__Oomi__= "$\Omega_{OMI}$"
__Oomi__units__ = __Ogc__units__
# a priori
__apri__ = "$E_{GC}$"
__apri__units__ = "atom C cm$^{-2}$ s$^{-1}$"
__apri__label__ = r'%s [ %s ]'%(__apri__, __apri__units__)
# a posteriori
__apost__ = "$E_{OMI}$"

# Plot size and other things...
def fullpageFigure(*kvpair):
    """set some Matplotlib stuff."""
    matplotlib.rcParams["text.usetex"]      = False     #
    matplotlib.rcParams["legend.numpoints"] = 1         # one point for marker legends
    matplotlib.rcParams["legend.fontsize"]  = 10        # legend font size
    matplotlib.rcParams["figure.figsize"]   = (12, 14)  #
    matplotlib.rcParams["font.size"]        = 18        # font sizes:
    matplotlib.rcParams["axes.titlesize"]   = 18        # title font size
    matplotlib.rcParams["axes.labelsize"]   = 13        #
    matplotlib.rcParams["xtick.labelsize"]  = 13        #
    matplotlib.rcParams["ytick.labelsize"]  = 13        #
    matplotlib.rcParams['image.cmap'] = 'plasma' #'PuRd' #'inferno_r'       # Colormap default
    # set extra key values if wanted
    for k,v in kvpair:
        matplotlib.rcParams[k] = v

labels=util.__subregions_labels__
colors=util.__subregions_colors__
regions=util.__subregions__
n_regions=len(regions)


####################
### FUNCTIONS
#######################

############################################################################
####################################### OMI RECALC #########################
############################################################################

def AMF_distributions(d0=datetime(2005,1,1),d1=datetime(2005,12,31), VCCs=False):
    '''
        Top row: averaged OMI Satellite AMF for 2005, from the OMHCHO data set (left, $AMF_{OMI}$), recalculated using GEOS-Chem shape factors  (middle, $AMF_{GC}$), and recalculated using GEOS-Chem shape factors and scattering weights (right, $AMF_{PP}$).
        Middle row: AMF time series over 2005 for each recalculation.
        Bottom row: AMF distibutions over January and February against a normalised Y axis.
        FIGURE: 
        maps :   AMF OMI, AMF GC, AMF PP 
        TS   :   all threeeeeeeeeeeeeee   : monthly resampled?
        distr:   all three Emissions      : monthly resampled?
    '''
    
    ystr=d0.strftime('%Y')
    if not VCCs:
        pname='Figs/AMF_distributions_%s.png'%(ystr)
        
        # read in omhchorp
        omkeys= [ #  'VCC_GC',           # The vertical column corrected using the RSC
                  #  'VCC_PP',        # Corrected Paul Palmer VC
                  #  'VCC_OMI',       # OMI VCCs from original satellite swath outputs
                  #  'VCC_OMI_newrsc', # OMI VCCs using original VC_OMI and new RSC corrections
                    'AMF_GC',        # AMF calculated using by GEOS-Chem
                  #  'AMF_GCz',       # secondary way of calculating AMF with GC
                    'AMF_OMI',       # AMF from OMI swaths
                    'AMF_PP',        # AMF calculated using Paul palmers code
                    ]
        om=omhchorp(d0,d1, keylist=omkeys)
        lats,lons=om.lats,om.lons
        dates=om.dates
        
        # AMF Subsets
        subsets=util.lat_lon_subset(lats,lons,pp.__AUSREGION__,data=[om.AMF_OMI,om.AMF_GC,om.AMF_PP],has_time_dim=True)
        lats,lons=subsets['lats'],subsets['lons']
        for i,istr in enumerate(['AMF (OMI)', 'AMF (GC) ', 'AMF (PP) ']):
            dat=subsets['data'][i]
            print("%s mean : %7.4f, std: %7.4f"%(istr, np.nanmean(dat),np.nanstd(dat)))
        
        amf_titles= ['AMF$_{OMI}$', 'AMF$_{GC}$', 'AMF$_{PP}$']
        amf_min, amf_max = .4,2.5
        bins=np.linspace(0.1,3,30)
    else:
        pname='Figs/VCC_distributions_%s.png'%(ystr)
        
        # read in omhchorp
        omkeys= [   'VCC_GC',           # The vertical column corrected using the RSC
                    'VCC_PP',        # Corrected Paul Palmer VC
                    'VCC_OMI',       # OMI VCCs from original satellite swath outputs
                  #  'VCC_OMI_newrsc', # OMI VCCs using original VC_OMI and new RSC corrections
                  #  'AMF_GC',        # AMF calculated using by GEOS-Chem
                  #  'AMF_GCz',       # secondary way of calculating AMF with GC
                  #  'AMF_OMI',       # AMF from OMI swaths
                  #  'AMF_PP',        # AMF calculated using Paul palmers code
                  ]
        om=omhchorp(d0,d1, keylist=omkeys)
        lats,lons=om.lats,om.lons
        dates=om.dates
        
        # AMF Subsets
        subsets=util.lat_lon_subset(lats,lons,pp.__AUSREGION__,data=[om.VCC_OMI,om.VCC_GC,om.VCC_PP],has_time_dim=True)
        lats,lons=subsets['lats'],subsets['lons']
        for i,istr in enumerate(['VCC (OMI)', 'VCC (GC) ', 'VCC (PP) ']):
            dat=subsets['data'][i]
            print("%s mean : %.2e, std: %.2e"%(istr, np.nanmean(dat),np.nanstd(dat)))
        
        
        amf_titles= ['VCC$_{OMI}$', 'VCC$_{GC}$', 'VCC$_{PP}$']
        amf_min, amf_max = 1e14, 7e15
        bins=np.linspace(5e12,3e16,30)
    
    # Mean for row of maps
    OMP = subsets['data'] # OMI, My, Palmer AMFs
    amf_colours=['orange','saddlebrown','salmon']    
    
    # Mask oceans for AMFs
    oceanmask = util.oceanmask(lats,lons)
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],om.n_times,axis=0)
    
    fullpageFigure() # Figure stuff
    plt.close()
    f=plt.figure()
    
    ax1=plt.subplot(3,1,2)
    ax2=plt.subplot(3,1,3)
    
    distrs=[]
    dlabels=[]
    for i, (amf, title,color) in enumerate(zip(OMP,amf_titles,amf_colours)):
        plt.subplot(3,3,i+1)
        # map the average over time
        m,cs,cb = pp.createmap(np.nanmean(amf,axis=0),lats,lons,
                               vmin=amf_min,vmax=amf_max, aus=True, 
                               linear=True, colorbar=False, title=title)
        
        # For time series we just want land
        amf[oceanmask3d] = np.NaN
        
        # AMF time series:
        plt.sca(ax1)
        
        # EXPAND out spatially, then get monthly means
        seasonal = util.monthly_averaged(dates,amf)
        smean = seasonal['mean']
        
        # lower and upper quantile?
        #slq = seasonal['lq']
        #suq = seasonal['uq']
        
        #yerr=np.zeros([2,len(smean)])
        #yerr[0,:] = slq
        #yerr[1,:] = suq    
        X = np.arange(len(smean))
        plt.plot(X,smean,label=title,color=color,linewidth=2)
        #plt.fill_between(mdates,mmean+mstd,mmean-mstd, color=color, alpha=0.35)
        
        # finally plot distributions in Jan/Feb:
        summerinds = np.array([d.month in [1,2] for d in dates])
        summer=amf[summerinds,:,:].flatten()
        infs = ~np.isfinite(summer)
        n_pix = np.nansum(~np.isnan(summer[~infs]))
        distrs.append(summer[~infs])
        dlabels.append(title+"(N=%d)"%n_pix)
    
    # Add colour bar at right edge for all three maps
    pp.add_colourbar(f,cs,label=["AMF",'VCC'][VCCs], axes=[0.9, 0.7, 0.02, 0.2])
    
    plt.sca(ax1)
    plt.legend(loc='best',ncol=3)
    plt.xlim([-.5,11.5])
    plt.xticks(X,['J','F','M','A','M','J','J','A','S','O','N','D'][0:len(smean)])
    plt.title("mean land-only %s"%(['AMF','VCC'][VCCs]))
    
    # Finally plot time series of emissions australian land average
    plt.sca(ax2)
    
    plt.hist(distrs,bins=bins, label=dlabels, color=amf_colours)
    
    plt.legend(loc='best', ncol=1)
    plt.title("distributions Jan-Feb, 2005")
    
    # save plot
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)



################################################
###### OMI RECALC ######################
#################################################

def N_pixels_comparison(d0=datetime(2005,1,1), dn=datetime(2005,12,31)):
    '''
        monthly australian land pixel counts for GC and PP AMF recalculations (also filtered amounts shown)
    '''
    pname="Figs/N_pix_comparison.png"
    Enew = E_new(d0,dn,dkeys=['pixels_PP_u','pixels_PP','pixels','pixels_u'])
    #lats = Enew.lats;  lons=Enew.lons
    dates= Enew.dates
    
    #smokesum = np.nansum(Enew.smokemask,axis=0).astype(np.float)
    Ep      = Enew.pixels
    Epu     = Enew.pixels_u
    Epp     = Enew.pixels_PP
    Eppu    = Enew.pixels_PP_u
    OM      = Enew.oceanmask3d
    del Enew
    
    arrays = Epu, Ep, Eppu, Epp
    labels = ['N$_{GC}$ (unfiltered)','N$_{GC}$','N$_{PP}$ (unfiltered)','N$_{PP}$',]
    colors = ['red','orange','blue','magenta']
    plt.close()
    for arr,label,color in zip(arrays,labels,colors):
        # Set ocean to no pixels
        arr[OM]=0
        
        monthly     = util.monthly_averaged(dates,arr,)
        msum        = monthly['sum']
        
        
        X = np.arange(len(msum))
        plt.plot(X,msum,label=label,color=color,linewidth=2)
    
    plt.xlim([-.5,11.5])
    plt.xticks(X,['J','F','M','A','M','J','J','A','S','O','N','D'][0:len(msum)])
    plt.xlabel("2005")
    plt.title("(N)umber of pixels over land")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(pname)
    print("SAVED ",pname)



###############################
########  Filtering data  #####
###############################

def plot_VCC_firefilter_vs_days(month=datetime(2005,1,1),region=pp.__AUSREGION__):
    '''
        Plot columns with different amf bases
        also different fire filtering strengths
              |  VCC_pp | pixel counts
        fire0 |
        fire2 |
        fire4 |
        fire8 |
    '''
    
    d0 = util.first_day(month)
    dn = util.last_day(month)
    # start by reading all the VCC stuff
    # useful strings
    ymstr=d0.strftime('%Y%m')
    pname='Figs/VCC_fires_%s.png'%ymstr
    #pname2='Figs/VCC_entries_%s.png'%ymdstr
    vmin,vmax=1e15,6e15 # min,max for colourbar
    linear=True # linear colour scale?
    vmin2,vmax2=5,60
    
    # read in omhchorp
    om=omhchorp(d0,dayn=dn, keylist=['VCC_PP','ppentries'])
    VCC = om.VCC_PP
    pix = om.ppentries
    subsets=util.lat_lon_subset(om.lats,om.lons,region=region,data=[VCC,pix], 
                                has_time_dim=True)
    VCC,pix = subsets['data'] # subsetted to region
    lats=subsets['lats']
    lons=subsets['lons']
    
    #print(vars(om).keys()) # [3, 720, 1152] data arrays returned, along with lats/lons etc.
    oceanmask=util.oceanmask(lats,lons)# lets just look at land squares
    oceanmask3d = np.repeat(oceanmask[np.newaxis,:,:], np.shape(VCC)[0], axis=0)
    VCC[oceanmask3d] = np.NaN
    pix[oceanmask3d] = 0
    
    
    # clear some ram
    del om
    
    # Will plot VCC at low resolution
    lats_lr, _      = GMAO.GMAO_lats(2.0)
    lons_lr, _      = GMAO.GMAO_lons(2.5)
    lat_lri,lon_lri = util.lat_lon_range(lats_lr,lons_lr,region)
    lats_lr         = lats_lr[lat_lri]
    lons_lr         = lons_lr[lon_lri]
    print("lats_lr:",lats_lr[0],'...',lats_lr[-1])
    print("lons_lr:",lons_lr[0],'...',lons_lr[-1])
    
    
    # Plot rows,cols,size:
    priordayslist=[0,2,4,8]
    f,axes=plt.subplots(len(priordayslist),2,figsize=[12,14])
    
    # first line is maps of VCC, VC_GC, VCC_PP
    titles=["$VCC_{PP}$", "$N_{pixels}$"]
    
    # Now loop over the same plots after NaNing our different fire masks
    
    for j, N in enumerate(priordayslist):
        #for j, N in enumerate([0,]):
        # first make the new active fire mask
        # firemask is 3dimensional: [days,lats,lons]
        fstart=timeit.default_timer()
        if j==0:
            firemask=np.zeros(np.shape(VCC)).astype(np.bool)
        else:
            firemask, fdays, flats, flons, fires= \
                fio.make_fire_mask(d0,dN=dn, prior_days_masked=N,
                                   region=region, max_procs=4)
        felapsed = timeit.default_timer() - fstart
        print ("TIMEIT: Took %6.2f seconds to make_fire_mask(%d days)"%(felapsed,N))
        # 10 minutes for 1 day, probably similar for 2-N days
        
        # appply mask to VCC and pixels
        VCCj = np.copy(VCC)
        pixj = np.copy(pix)
        VCCj[firemask] = np.NaN
        pixj[firemask] = 0
        
        # Then plot VCC and entries maps
        plt.sca(axes[j,0])
        
        # FIRST CONVERT TO LOW RES
        VCCj_lr = np.nanmean(VCCj,axis=0) # Average over time
        print(np.shape(VCCj),np.shape(VCCj_lr),np.shape(lats),np.shape(lons),np.shape(lats_lr),np.shape(lons_lr))
        VCCj_lr= util.regrid_to_lower(VCCj_lr,lats,lons,lats_lr,lons_lr)
        print(np.shape(VCCj),np.shape(VCCj_lr))
        # VCC first:
        m,cs,cb= pp.createmap(VCCj_lr,lats_lr,lons_lr,
                              region=region, linear=linear,
                              vmin=vmin,vmax=vmax,
                              cmapname='rainbow',colorbar=False)
        if j==0:
            plt.title(titles[0])

        # add a little thing showing entries and mean and max
        txt=['N=%d'%(np.nansum(pixj)), 'mean = %.2e'%np.nanmean(VCCj), 'max = %.2e'%np.nanmax(VCCj)]
        for txt, yloc in zip(txt,[0.005,0.07,0.14]):
            plt.text(0.01, yloc, txt,
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='k', fontsize=10)

        # also plot entries
        plt.sca(axes[j,1])
        m2, cs2, cb2 = pp.createmap(np.nansum(pixj,axis=0),lats,lons,
                     region=region, linear=True,
                     cmapname='hot_r', colorbar=False,
                     vmin=vmin2,vmax=vmax2)
 
        if j==0:
            plt.title(titles[1])

    # Add row labels
    rows = ['%d days'%fdays for fdays in priordayslist]
    rows[0]='no filter'
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='small')

    # Need to add colour bar for left column and right column
    f.tight_layout()
    f.subplots_adjust(bottom=0.1)
    # left bottom width height
    axes0=[0.125,0.05,0.3,0.03]
    axes1=[0.575,0.05,0.3,0.03]
    
    # left column:
    ticks=[np.logspace(np.log10(vmin),np.log10(vmax),5),np.linspace(vmin,vmax,5)][linear]
    
    cbar_ax = f.add_axes(axes0)
    cb=f.colorbar(cs,cax=cbar_ax,orientation='horizontal')
    cb.set_ticks(ticks)
    cb.set_label('molec cm$^{-2}$')
    
    # right column:
    cbar_ax = f.add_axes(axes1)
    cb=f.colorbar(cs2,cax=cbar_ax,orientation='horizontal')
    #cb.set_ticks(ticks)
    cb.set_label('pixel count')
    
    f.savefig(pname)
    plt.close(f)
    print("Saved ",pname)

    

def pyrogenic_filter():
    '''
        Show pyrogenic filter pixels removed for 2005 and 2012, and time series
    '''
    pname = "Figs/Filters/Pyrogenic_filter.png"
    d0=datetime(2005,1,1)
    dN=datetime(2005,12,31)
    
    # Also want to look at 2012
    d12 = datetime(2012,1,1)
    d12N= datetime(2012,12,31)
    
    firemask05, dates05, lats, lons  = fio.get_fire_mask(d0,dN,region=pp.__AUSREGION__)
    firemask12, dates12, lats, lons  = fio.get_fire_mask(d12,d12N,region=pp.__AUSREGION__)
    smokemask05, _, _, _             = fio.get_smoke_mask(d0,dN,region=pp.__AUSREGION__)
    smokemask12, _, _, _             = fio.get_smoke_mask(d12,d12N,region=pp.__AUSREGION__)
    
    pyro05 = firemask05+smokemask05
    pyromap05 = np.nansum(pyro05,axis=0).astype(np.float)
    pyro12 = firemask12+smokemask12
    pyro12 = pyro12[0:-1] # cut off that leap year day
    pyromap12 = np.nansum(pyro12,axis=0).astype(np.float)
    
    
    plt.close()
    #plt.figure(figsize=[16,16])
    plt.subplot(2,2,1)
    m,cs,cb    = pp.createmap(pyromap05,lats,lons,title='days filtered 2005',
                              aus=True,linear=True, set_under='grey',vmin=1, vmax=300,colorbar=False)
    plt.subplot(2,2,2)
    m2,cs2,cb2 = pp.createmap(pyromap12,lats,lons,title='days filtered 2012',
                              aus=True,linear=True, set_under='grey',vmin=1, vmax=300, colorbar=False)
    
    plt.subplot(2,1,2)
    
    # Take out ocean squares
    om = util.oceanmask(lats,lons)
    om = np.repeat(om[np.newaxis,:,:],365,axis=0)
    squares = np.ones(om.shape).astype(np.float)
    squares[om] = np.NaN
    for arr, label, color in zip([pyro05,pyro12],['2005','2012'],['saddlebrown','magenta']):
        arr = arr.astype(np.float)
        arr[om]=np.NaN
        # portion of gridsquares filtered out as a percentage of land squares available
        ts = 100 * np.nansum(arr,axis=(1,2))/np.nansum(squares,axis=(1,2))
        plt.plot(np.arange(1,len(ts)+1), ts, linewidth=2, label=label, color=color)
    plt.title('land area filtered')
    plt.legend(loc='best')
    plt.ylabel('%')
    plt.xlabel('Day of the year')
    
    # add axis to middle area just below maps
    # left, bottom, width, height
    axes=[0.33,0.54,0.33,0.03]
    f=plt.gcf()
    cbar_ax = f.add_axes(axes)
    cb=f.colorbar(cs,cax=cbar_ax,orientation='horizontal')
    #cb.set_ticks(ticks)
    cb.set_label('days masked')
    
    plt.savefig(pname)
    print('Saved ',pname)
            



def HCHO_vs_temperature(d0=datetime(2005,1,1), d1=datetime(2005,2,28),
                        hcholims=[-0.4e16,3e16], surfhcholims=[0.1,8]):
    '''
        Exponential RMA regression of Sydney, SEA HCHO vs CPC and GC temperatures, with and without fires
        
        FIGURE:  scatter of HCHO (Y axis) vs temperature (X axis)
                  With fires    |    Without fires
        Sydney
        SEA
            legend: colour = GC or CPC temperatures
            inset: text: slope and regression coefficient, and N
        
        
    '''
    ymd=d0.strftime('%Y%m%d') + '-' + d1.strftime('%Y%m%d')
    pname='Figs/Filters/HCHO_vs_temp_%s.png'%(ymd)
    pname2='Figs/Filters/HCHO_vs_temp_surface_%s.png'%(ymd)
    
    Tcolor="saddlebrown"
    cpccolor="magenta"
    
    Syd = [-33.87,151.21] # Sydney lat, lon
    SEA = util.__subregions__[1]
    
    ## ENEW COLUMNS AND FIRE MASK
    # read fire filter, VCC_PP, and pixel counts?
    Enew=E_new(d0,d1,dkeys=['firemask','VCC_PP_u','VCC_PP','pixels_u','pixels'])
    ntimes=len(Enew.dates)
    
    # VCC HCHO
    VCC_u = Enew.VCC_PP_u
    VCC   = Enew.VCC_PP
    
    # convert some things to lower resolution
    firemask_lr = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)])
    VCC_lr_u = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)])
    VCC_lr   = np.zeros([ntimes,len(Enew.lats_lr),len(Enew.lons_lr)])
    for ti in range(ntimes):
        firemask_lr[ti,:,:] = util.regrid(Enew.firemask[ti,:,:],
                                            Enew.lats,Enew.lons,
                                            Enew.lats_lr,Enew.lons_lr,
                                            groupfunc=np.sum)
        VCC_lr_u[ti,:,:]   = util.regrid_to_lower(VCC_u[ti,:,:],
                                                     Enew.lats,Enew.lons,
                                                     Enew.lats_lr,Enew.lons_lr,
                                                     pixels=Enew.pixels_u[ti,:,:])
        VCC_lr[ti,:,:]     = util.regrid_to_lower(VCC[ti,:,:],
                                                     Enew.lats,Enew.lons,
                                                     Enew.lats_lr,Enew.lons_lr,
                                                     pixels=Enew.pixels[ti,:,:])
    
    ### GC HCHO SURFACE AND TEMPERATURE
    # read modelled hcho, and temperature at midday
    gc=GC_class.GC_sat(d0,d1,keys=['IJ-AVG-$_CH2O','DAO-FLDS_TS'],run='tropchem')
    
    # Surface HCHO
    HCHO_u  = gc.hcho[:,:,:,0] # surface HCHO ppbv?
    T_gc        = gc.surftemp[:,:,:,0] # surface temp in Kelvin
    
    # Regrid fire filter to apply to GC surface HCHO
    # First need to subset HCHO and firemask to the same area
    SEA_sub     = util.lat_lon_subset(gc.lats,gc.lons,region=SEA, data=[T_gc,HCHO_u],
                                      has_time_dim=True)
    firemask_sea_lr = util.lat_lon_subset(Enew.lats_lr,Enew.lons_lr,region=SEA,
                                          data=[firemask_lr],has_time_dim=True)['data'][0]
    HCHO_u  = SEA_sub['data'][1] # subset to SEA
    HCHO    = np.copy(HCHO_u)
    HCHO[firemask_sea_lr>0] = np.NaN
    syd_lati, syd_loni  = util.lat_lon_index(Syd[0],Syd[1],SEA_sub['lats'],SEA_sub['lons'])
    
    HCHO_syd_u  = HCHO_u[:,syd_lati,syd_loni]
    HCHO_syd    = HCHO[:,syd_lati,syd_loni]
    HCHO_sea_u  = np.nanmean(HCHO_u, axis=(1,2))
    HCHO_sea    = np.nanmean(HCHO,axis=(1,2))
    
    # GC output subset and locational time series:
    syd_lati, syd_loni = util.lat_lon_index(Syd[0],Syd[1],gc.lats,gc.lons)
    T_gc_syd    = T_gc[:,syd_lati,syd_loni]
    T_gc_sea    = np.nanmean(SEA_sub['data'][0], axis=(1,2))
    
    
    
    ### CPC DAILY MAX TEMPERATURES
    # Read the CPC temperatures
    tmax, _d, lats_cpc, lons_cpc = fio.read_CPC_temp(d0,d1,regrid=False)
    tmax=tmax+273.15 # to kelvin
    T_cpc = np.zeros(np.shape(VCC_lr))+np.NaN
    # Subset over AUS and change to low resolution using avg maxtemp for each grid square
    for i in range(tmax.shape[0]):
        T_cpc[i,:,:] = util.regrid_to_lower(tmax[i], lats_cpc,lons_cpc,
                                            Enew.lats_lr,Enew.lons_lr)
    #plt.close()
    #plt.figure()
    #plt.subplot(2,1,1)
    #pp.createmap(T_cpc[0],Enew.lats_lr,Enew.lons_lr,aus=True,
    #             colorbar=False,linear=True,vmin=270,vmax=320)
    #plt.subplot(2,1,2)
    #pp.createmap(tmax[0],lats_cpc,lons_cpc,aus=True,
    #             colorbar=False,linear=True,vmin=270,vmax=320)
    #plt.savefig("test_temp.png")
    # LOOKS FINE
    
    
    
    #T_cpc = ... # surface temp in Kelvin
    
    # PULL OUT Sydney time series, and SEA area averaged time series for HCHO and Temperature
    syd_lati,syd_loni   = util.lat_lon_index(Syd[0],Syd[1],Enew.lats_lr,Enew.lons_lr)
    VCC_syd_lr_u        = VCC_lr_u[:,syd_lati,syd_loni]
    VCC_syd_lr          = VCC_lr[:,syd_lati,syd_loni]
    SEA_sub             = util.lat_lon_subset(Enew.lats_lr,Enew.lons_lr,
                                       region=SEA,data=[VCC_lr_u,VCC_lr,T_cpc],
                                       has_time_dim=True)
    VCC_sea_u           = np.nanmean(SEA_sub['data'][0],axis=(1,2))
    VCC_sea             = np.nanmean(SEA_sub['data'][1],axis=(1,2))
    T_cpc_syd           = T_cpc[:,syd_lati,syd_loni]
    T_cpc_sea           = np.nanmean(SEA_sub['data'][2],axis=(1,2))
    
    
    # FIGURE: corellations in following subplots
    #
    plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(20,18), sharex='all', sharey='all')
    '''          With fires    |    Without fires
        Sydney
        SEA
            legend: colour = GC or CPC temperatures
            inset: text: slope and regression coefficient, and N
    '''
    titles=['Fires remain','Fires filtered out','','']
    ylabels=['Sydney','','SEA','']
    #       temp,  column,  surface
    for i, (temp, hcho, cpc) in enumerate(zip([T_gc_syd, T_gc_syd, T_gc_sea, T_gc_sea],
                                              [VCC_syd_lr_u,VCC_syd_lr,VCC_sea_u,VCC_sea],
                                              [T_cpc_syd, T_cpc_syd, T_cpc_sea, T_cpc_sea])):
        
        plt.sca(ax[i//2, i%2])
        
        # Plot temperature vs hcho scatter
        plt.scatter(temp, hcho, color=Tcolor)
        
        # add regression (exponential)
        pp.add_regression(temp, hcho, addlabel=True, labelprefix="T$_{GC}$ :", 
                          exponential=True, color=Tcolor, linewidth=1)
        
        # ALSO overplot HCHO vs CPC
        
        plt.scatter(cpc, hcho, color=cpccolor)
        pp.add_regression(cpc,hcho, addlabel=True, labelprefix="T$_{CPC}$: ",
                          exponential=True, color=cpccolor, linewidth=1)
        
        plt.legend(loc='upper center',fontsize=17)
        
        plt.title(titles[i])
        plt.ylabel(ylabels[i])
        plt.ylim(hcholims)
        
    plt.suptitle('Temperature vs $\Omega_{HCHO}$ [molec cm$^{-2}$] %s'%ymd,fontsize=36)
    for ax in [ax[1,0],ax[1,1]]:
        plt.sca(ax)
        plt.xlabel('Kelvin')
    
    plt.savefig(pname)
    plt.close()
    print('Saved ',pname)
    
    
    ### SAME AGAIN JUST USING SURFACE HCHO
    fig, ax = plt.subplots(2, 2, figsize=(20,18), sharex='all', sharey='all')
    titles=['Fires remain','Fires filtered out','','']
    ylabels=['Sydney','','SEA','']
    #       temp,  column,  surface
    for i, (temp, hcho, cpc) in enumerate(zip([T_gc_syd, T_gc_syd, T_gc_sea, T_gc_sea],
                                              [HCHO_syd_u,HCHO_syd,HCHO_sea_u,HCHO_sea],
                                              [T_cpc_syd, T_cpc_syd, T_cpc_sea, T_cpc_sea])):
        
        plt.sca(ax[i//2, i%2])
        
        # Plot temperature vs hcho scatter
        plt.scatter(temp, hcho, color=Tcolor)
        
        # add regression (exponential)
        pp.add_regression(temp, hcho, addlabel=True, labelprefix="T$_{GC}$:", 
                          exponential=True, color=Tcolor, linewidth=1)
        
        # ALSO overplot HCHO vs CPC
        
        plt.scatter(cpc, hcho, color=cpccolor)
        pp.add_regression(cpc,hcho, addlabel=True, labelprefix="T$_{CPC}$: ",
                          exponential=True, color=cpccolor, linewidth=1)
        
        plt.legend(loc='upper center',fontsize=17)
        
        plt.title(titles[i])
        plt.ylabel(ylabels[i])
        plt.ylim(surfhcholims)
        
    plt.suptitle('Temperature vs surface HCHO [ppbv] %s'%ymd,fontsize=36)
    for ax in [ax[1,0],ax[1,1]]:
        plt.sca(ax)
        plt.xlabel('Kelvin')
    
    
    plt.savefig(pname2)
    plt.close()
    print('Saved ',pname2)


def no2_thresh(no2_orig,dates,lats,lons,
               threshd=fio.__Thresh_NO2_d__, threshy=fio.__Thresh_NO2_y__,
               subzones=pp.__subzones_AUS__,colors=pp.__subregions_colors__):
    '''
        Look at effect of applying threshold
        no2_orig should be [t,lats,lons] for a particular year
    '''
    print("IMPLEMENTATION REQUIRED")



###########################
#### MAIN
###########################
if __name__=="__main__":
    
    fullpageFigure()
    ##############################
    #### OMI RECALC PLOTS ########
    
    # AMF distribution summary 
    #AMF_distributions()
    #AMF_distributions(VCCs=True)
    
    
    ##############################
    ### FIltering stuff 3#########
    #N_pixels_comparison()
    #plot_VCC_firefilter_vs_days()
    #pyrogenic_filter() # 2019/5/24
    
    #HCHO_vs_temperature()
    #HCHO_vs_temperature(d0=datetime(2005,12,1),d1=datetime(2006,2,28))
    from tests.test_filters import no2_map, no2_timeseries
    
    # READ NO2, dates,lats,lons
    day0,dayN=datetime(2005,1,1),datetime(2005,12,31)
    data = fio.get_anthro_data(day0,dayN,region=pp.__AUSREGION__,)
    no2_orig=data['no2']
    dates=data['dates']
    lats=data['lats']
    lons= data['lons']
    threshd=fio.__Thresh_NO2_d__
    threshy=fio.__Thresh_NO2_y__
    subzones=pp.__subzones_AUS__
    colors=pp.__subzones_colours__

    #    '''
    #        Look at effect of applying threshold
    #        no2_orig should be [t,lats,lons] for a particular year
    #    '''
    
    year=dates[0].year
    # first subzone is AUSTRALIA
    region=subzones[0]
    
    
    # Subset to region:
    subset=util.lat_lon_subset(lats,lons,region,data=[np.copy(no2_orig)],has_time_dim=True)
    no2=subset['data'][0]
    lats,lons=subset['lats'],subset['lons']


    # Mask ocean
    oceanmask=util.get_mask(no2[0],lats,lons,masknan=False,maskocean=True)
    #print(no2.shape, lats.shape, lons.shape)
    no2[:,oceanmask] = np.NaN

    # Also remove negatives
    negmask=no2 < 0
    #no2[negmask]=np.NaN

    # Filtered copy:
    no2_f=np.copy(no2)
    no2_mean=np.nanmean(no2,axis=0)

    # Filter for yearly threshold, first count how many we will remove
    n_filtered_y = 0
    n_days=len(no2_f[:,0,0])
    n_filtered_y = n_days*np.nansum(no2_mean>threshy)
    # if day is nan in square removed by threshy then don't count it as removed
    n_filtered_y = n_filtered_y - np.sum(np.isnan(no2_f[:,no2_mean>threshy]))
    # apply filter
    no2_f[:,no2_mean>threshy]=np.NaN

    # Apply daily filter
    no2_f[no2_f>threshd] = np.NaN

    ## Printing table as follows:
    #\textbf{Region} & \textbf{NO$_2$} & \textbf{NO$_2$ after filtering} & \textbf{\% Data lost}    \\ 
    #Aus & 1.5e15 & 2e15 & 20\% \\
    formstr = "%s & %.2f & %.2f & %5.1f\\%% \\\\"
    print("MAKING TABLE FOR TEXT (1e14 * molec/cm2)")
    for region, label in zip(subzones,pp.__subzones_labels__):
        subset = util.lat_lon_subset(lats,lons,region,data=[no2,no2_f],has_time_dim=True)
        sub_mean = np.nanmean(subset['data'][0])*1e-14
        sub_mean_f = np.nanmean(subset['data'][1])*1e-14
        ngoods=np.nansum(subset['data'][0]>0)
        ngoods_f=np.nansum(subset['data'][1]>0)
        portion_filtered = (1-ngoods_f/ngoods)*100
        print(formstr%(label, sub_mean,sub_mean_f,portion_filtered))
    
    
    
    
    
    #### FIGURE
    ###
    pname='Figs/OMNO2_thresheffect_%d.png'%year
    fig, axes = plt.subplots(2,2,figsize=[16,16])

        
    # plot stuff:
    titles = ['NO$_2$ Mean', 'After filtering']
    vmin = 1e14
    vmax = 2e15
    
    for i,arr in enumerate([no2,no2_f]):

        # Plot map with and without threshold filter
        mean=np.nanmean(arr,axis=0)
        plt.sca(axes[0,i])
        # only show subzones in first plot
        bmap,_cs,_cb = no2_map(mean, lats, lons, vmin, vmax, [subzones,None][i==1], colors)
        plt.title(titles[i])

        # Hatch for yearly threshold
        #pp.hatchmap(bmap, mean, lats, lons, threshy, region=region)

        # Also time series with and without filter
        plt.sca(axes[1,i])
        no2_timeseries(arr,dates,lats,lons,subzones,colors,print_values=False)

    # Add thresholds to last timeseries
    plt.plot([0,395],[threshd,threshd], '--k',linewidth=1)
    plt.plot([0,395],[threshy,threshy], ':k',linewidth=1)
    for ax in axes[1,:]:
        plt.sca(ax)
        plt.yticks([2e14,5e14,1e15,1.5e15,2e15,4e15],['2e14','5e14','1e15','1.5e15','2e15','4e15'])
    plt.suptitle("OMNO2d threshold effects %d"%year, fontsize=24)
    plt.savefig(pname)
    print('saved ',pname)
    plt.close()
