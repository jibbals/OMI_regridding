#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:18:17 2019

@author: jesse
"""

from datetime import datetime
import csv
import numpy as np
import os.path, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


# fio and data classes
from utilities import fio, GC_fio
from utilities import plotting as pp
from classes import GC_class, campaign

##################
### GLOBALS ######
##################

def ftir_method_plots():
    '''
        look at ftir profile vs a priori
        plot averaging kernal summary
            mean AK for column and profiles
    '''
    pname_prof = 'Figs/FTIR_apriori.png'
    pname_AK = 'Figs/FTIR_midday_AK.png'
        
    # Read FTIR output
    ftir=campaign.Wgong()
    
    # Resample FTIR to just midday averages
    middatas=ftir.resample_middays()
    
    ### Mean profile and apriori
    
    # Plot mean profile vs mean a priori
    plt.close()
    alts = ftir.alts
    for i, (prof, c) in enumerate(zip([middatas['VMR'], middatas['VMR_apri']],['k','teal'] )):
        # Convert ppmv to ppbv and plot profiles
        mean = 1000*np.nanmean(prof,axis=0)
        lq = 1000*np.nanpercentile(prof, 25, axis=0)
        uq = 1000*np.nanpercentile(prof, 75, axis=0)
        plt.fill_betweenx(alts, lq, uq, alpha=0.5, color=c)
        plt.plot(mean,alts,label=['x$_{ret}$','x$_{apri}$'][i],linewidth=2,color=c)
    plt.ylim([0,50])
    plt.ylabel('altitude [km]')
    plt.xlabel('HCHO [ppbv]')
    plt.legend(fontsize=14)
    plt.title('FTIR mean profile')
    plt.savefig(pname_prof)
    print("Saved ",pname_prof)
    plt.close()
    
    
    ### Mean averaging kernal summarised
    
    # check plot of VC_AK
    # [dates, levels]
    plt.close()
    plt.figure(figsize=(12,12))
    ax0=plt.subplot(1,2,1)
    OAK=middatas['VC_AK']
    mean = np.nanmean(OAK,axis=0)
    lq = np.nanpercentile(OAK,25, axis=0)
    uq = np.nanpercentile(OAK,75, axis=0)
    plt.fill_betweenx(ftir.alts,lq,uq, label='IQR')
    plt.plot(mean, ftir.alts,color='k',linewidth=2, label='mean')
    plt.title("$\Omega$ sensitivity to HCHO")
    plt.legend()
    plt.ylabel('altitude [km]')
    
    # also check average AK
    AAK = np.nanmean(middatas['VMR_AK'],axis=0)    
    colors=pp.get_colors('gist_ncar',48) # gist_ncar
    plt.subplot(1,2,2, sharey=ax0)
    for i in np.arange(0,48,1):
        sample = int((i%6)==0)
        label=[None,ftir.alts[47-i]][sample]
        linestyle=['--','-'][sample]
        linewidth=[1,2][sample]
        alpha=[.5,1][sample]
        plt.plot(AAK[47-i],ftir.alts, color=colors[i], alpha=alpha,
                 label=label,linestyle=linestyle,linewidth=linewidth)
    plt.legend(title='altitude')
    plt.title('Mean averaging kernal')
    #plt.colorbar()
    plt.ylim([-1,81])
    
    plt.savefig(pname_AK)
    print('Saved ',pname_AK)