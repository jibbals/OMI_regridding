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

def AMF_distributions(d0=datetime(2005,1,1),d1=datetime(2005,12,31)):
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
    
    
    # Mask oceans for AMFs
    oceanmask = util.oceanmask(lats,lons)
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],om.n_times,axis=0)
    
    # Mean for row of maps
    OMP = subsets['data'] # OMI, My, Palmer AMFs
    amf_titles= ['AMF$_{OMI}$', 'AMF$_{GC}$', 'AMF$_{PP}$']
    amf_colours=['orange','saddlebrown','salmon']
    
    
    
    fullpageFigure() # Figure stuff
    plt.close()
    f=plt.figure()
    
    ax1=plt.subplot(3,1,2)
    ax2=plt.subplot(3,1,3)
    
    amf_min, amf_max = .4,2.5
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
        
        #to get quantiles need different method
        slq = seasonal['lq']
        suq = seasonal['uq']
        
        #yerr=np.zeros([2,len(smean)])
        #yerr[0,:] = slq
        #yerr[1,:] = suq    
        X = np.arange(len(smean))
        plt.plot(X,smean,label=title,color=color,linewidth=2)
        #plt.fill_between(mdates,mmean+mstd,mmean-mstd, color=color, alpha=0.35)
        
        # finally plot distributions in Jan/Feb:
        summerinds = np.array([d.month in [1,2] for d in dates])
        summer=amf[summerinds,:,:].flatten()
        n_pix = np.nansum(~np.isnan(summer))
        distrs.append(summer)
        dlabels.append(title+"(N=%d)"%n_pix)
    
    # Add colour bar at right edge for all three maps
    pp.add_colourbar(f,cs,label="AMF",axes=[0.9, 0.7, 0.02, 0.2])
    
    plt.sca(ax1)
    plt.legend(loc='best',ncol=3)
    plt.xlim([-.5,11.5])
    plt.xticks(X,['J','F','M','A','M','J','J','A','S','O','N','D'][0:len(smean)])
    plt.title("mean land-only AMF")
    
    # Finally plot time series of emissions australian land average
    plt.sca(ax2)
    
    plt.hist(distrs,bins=np.linspace(0.1,3,30), label=dlabels, color=amf_colours)
    
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







###########################
#### MAIN
###########################


if __name__=="__main__":
    
    
    ##############################
    #### OMI RECALC PLOTS ########
    
    # AMF distribution summary 
    AMF_distributions()    
    