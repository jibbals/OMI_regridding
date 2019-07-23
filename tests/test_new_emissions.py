#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:33:05 2018

@author: jesse
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import timeit

import new_emissions
from utilities import utilities as util
from classes.E_new import E_new
from classes import GC_class
from utilities import GC_fio

# plotting
from utilities import plotting as pp
import matplotlib.pyplot as plt

####
## GLOBALS
####

__VERBOSE__=True

####
## TESTS
####

def alpha_creation():
    '''
        Create isoprene scaling factors monthly over Australia
          using difference from top-down and MEGAN emissions at midday

        Create yearly plots and compare with multi-year avg version
    '''
    label_meg='a priori'
    label_topd='a posteriori'

    yearly_alphas=[]
    yearly_megan=[]
    yearly_topd=[]
    # first loop over year and create yearly alpha plots
    for year in range(2005,2013):
        dat = new_emissions.calculate_alpha(year,mya=False)
        topd=dat['Enew']
        megan=dat['Emeg']
        alpha=dat['alpha']
        yearly_alphas.append(alpha)
        yearly_topd.append(dat['Enewm'])
        yearly_megan.append(dat['Emegm'])
        lats=dat['lats']
        lons=dat['lons']
        dates=dat['dates']
        months=dat['months']
        months=[m+timedelta(days=15) for m in months]
        region=[-50,105,-7,155]
        vmin,vmax = 0, 2
        sydlat,sydlon = pp.__cities__['Syd']

        lati,loni = util.lat_lon_index(sydlat,sydlon,lats,lons)

        plt.figure(figsize=[15,13])
        # first plot alpha in jan, then alpha in
        plt.subplot(221)
        pp.createmap(alpha[0],lats, lons, linear=True, region=region, title='January',vmin=vmin,vmax=vmax)
        # then plot alpha in June
        plt.subplot(222)
        pp.createmap(alpha[6],lats, lons, linear=True, region=region, title='July',vmin=vmin,vmax=vmax)
        #finally plot time series at sydney of alpha, megan, and topdown emissions
        plt.subplot(212)
        plt.title('Sydney',fontsize=22)
        plt.plot_date(dates, megan[:,lati,loni], 'm-', label=label_meg)
        plt.plot_date(dates, topd[:,lati,loni], '-', label=label_topd, color='cyan')
        plt.ylim(1e11,2e13)
        plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]')
        plt.legend()
        plt.sca(plt.twinx())
        plt.plot_date(months, alpha[:,lati,loni], 'k-', linewidth=3, label='$\\alpha$')
        plt.ylim(vmin,vmax)
        plt.ylabel('$\\alpha$')
        plt.suptitle('$\\alpha$ for %4d'%year)
        plt.savefig('Figs/new_emiss/alpha_%4d.png'%year)
        plt.close()

    allalpha = np.concatenate([yearly_alphas[i] for i in range(8)], axis=0)
    allmegan = np.concatenate([yearly_megan[i] for i in range(8)], axis=0)
    alltopd  = np.concatenate([yearly_topd[i] for i in range(8)], axis=0)

    # finally create/plot mya alpha
    dat = new_emissions.calculate_alpha(mya=True)
    topd=dat['Enewm']
    megan=dat['Emegm']
    alpha=dat['alpha']
    lats=dat['lats']
    lons=dat['lons']
    dates=dat['dates']
    months=dat['months']
    months=[m+timedelta(days=15) for m in months]
    allmonths=util.list_months(dates[0],dates[-1])
    print(np.shape(allmonths), np.shape(allalpha), np.shape(megan))

    region=[-45,110,-10,155]
    vmin,vmax = 0, 2
    sydlat,sydlon = pp.__cities__['Syd']

    lati,loni = util.lat_lon_index(sydlat,sydlon,lats,lons)

    f=plt.figure(figsize=[15,13])
    # first plot alpha in jan, then alpha in
    plt.subplot(221)
    fixed=np.copy(alpha)
    fixed[fixed == 1] =np.NaN
    pp.createmap(fixed[0],lats, lons, linear=True, region=region, 
                 title='January',vmin=vmin,vmax=vmax, colorbar=False)
    # then plot alpha in June
    plt.subplot(222)
    m,cs,cb = pp.createmap(fixed[6],lats, lons, linear=True, region=region, 
                 title='July',vmin=vmin,vmax=vmax, colorbar=False)
    
    # add colourbar between maps
    cbar_ax1 = f.add_axes([0.485, 0.55, .0225, 0.3])
    cb = f.colorbar(cs,cax=cbar_ax1, orientation='vertical')
    cb.set_label('$\\alpha$')
    
    #finally plot time series at sydney of alpha, megan, and topdown emissions
    plt.subplot(212)
    X=range(12)
    plt.plot(X, megan[:,lati,loni], 'm-', label=label_meg)
    plt.plot(X, topd[:,lati,loni], '-', label=label_topd, color='cyan')
    plt.ylim(1e11,1.5e13)
    plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]',fontsize=17)
    plt.legend(loc='best',fontsize=14)
    plt.title('Sydney', fontsize=22)
    plt.sca(plt.twinx())
    plt.plot(X, alpha[:,lati,loni], 'k-', linewidth=3, label='$\\alpha$')
    plt.plot([X[0],X[-1]], [1,1], 'k--', linewidth=1) # dotted line
    plt.xlim(-0.5,11.5)
    plt.xticks(X)
    plt.gca().set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

    plt.ylim(vmin,vmax)
    plt.ylabel('$\\alpha$')
    plt.suptitle('$\\alpha$ multi-year monthly average 2005-2012',fontsize=22)
    plt.savefig('Figs/new_emiss/alpha_mya.png')
    print("SAVED ", 'Figs/new_emiss/alpha_mya.png')
    plt.close()
    
    plt.figure()
    plt.plot_date(allmonths, allmegan[:,lati,loni], 'm-', label=label_meg)
    plt.plot_date(allmonths, alltopd[:,lati,loni], '-', label=label_topd, color='cyan')
    plt.ylim(1e11,1.5e13)
    plt.ylabel('Emissions [atom C cm$^{-2}$ s$^{-1}$]',fontsize=17)
    plt.legend(loc='best',fontsize=14)
    plt.title('monthly average $\\alpha$ over grid square with Sydney')
    plt.sca(plt.twinx())
    plt.plot_date(allmonths,allalpha[:,lati,loni], 'k-', linewidth=3)
    plt.plot_date([allmonths[0],allmonths[-1]], [1,1], 'k--',linewidth=1) # dotted line
    plt.ylabel('$\\alpha$')
    plt.ylim(vmin,vmax)

    plt.savefig('Figs/new_emiss/alpha_monthly_sydney.png')
    print("SAVED ", 'Figs/new_emiss/alpha_monthly_sydney.png')
    plt.close()

def check_old_vs_new_emission_diags(month=datetime(2005,1,1)):
    # compare new_emissions hemco output
    d0=datetime(month.year,month.month,1)
    d1=util.last_day(d0)

    # old emissions are hourly
    old, oldattrs = GC_fio.read_Hemco_diags(d0,d1, new_emissions=False)
    # new emissions are daily
    new, newattrs = GC_fio.read_Hemco_diags(d0,d1, new_emissions=True)

    lats=old['lat']
    lons=old['lon']

    old_isop = np.mean(old['ISOP_BIOG'],axis=0) # hourly -> 1 month avg
    new_isop = np.mean(new['ISOP_BIOG'],axis=0) # daily -> 1 month avg

    pname=month.strftime('Figs/new_emiss/check_old_new_emission_diags_%Y%m.png')
    pp.compare_maps([old_isop,new_isop], [lats,lats],[lons,lons],linear=True,
                    rmin=-400,rmax=400, vmin=0,vmax=0.8e-9, titles=['old','new'],
                    region=pp.__GLOBALREGION__, pname=pname)

def isop_biog_time_series(d0=datetime(2005,1,1), d1=None):

    # Read whole year and do time series for both
    old, oldattrs = GC_fio.read_Hemco_diags(d0,None, new_emissions=False)
    new, newattrs = GC_fio.read_Hemco_diags(d0,None, new_emissions=True)
    old_isop = old['ISOP_BIOG']
    new_isop = new['ISOP_BIOG']
    old_dates= old['dates']
    new_dates= new['dates']
    lats,lons= new['lats'],new['lons']

    # remove last index from old data, and old dates
    lastindex=len(old_dates)-1
    old_dates = np.delete(old_dates,[lastindex])
    old_isop  = np.delete(old_isop, [lastindex],axis=0)

    # pull out regions and compare time series
    r_olds, r_lats, r_lons = util.pull_out_subregions(old_isop,
                                                     lats, lons,
                                                     subregions=pp.__subregions__)
    r_news, r_lats, r_lons = util.pull_out_subregions(new_isop,
                                                     lats, lons,
                                                     subregions=pp.__subregions__)
    f,axes = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    for i, [label, color] in enumerate(zip(pp.__subregions_labels__, pp.__subregions_colors__)):
        # set current axis
        plt.sca(axes[i])

        # plot time series for each subregion
        r_old_hourly = np.nanmean(r_olds[i],axis=(1,2)) # hourly regional avg
        # change hourly into daily time series
        r_old = np.array(pd.Series(r_old_hourly,index=old_dates).resample('D').mean())
        r_new = np.nanmean(r_news[i],axis=(1,2)) # daily

        pp.plot_time_series(new_dates,r_old, label='a priori', linestyle='--', color=color, linewidth=2)
        pp.plot_time_series(new_dates,r_new, label='a posteriori', linestyle='-', color=color, linewidth=2)
        plt.title(label,fontsize=20, color=color)
        if i==0:
            plt.ylabel('kgC m$^{-2}$ s$^{-1}$')
            plt.legend(loc='best')

    pname = 'Figs/new_emiss/E_isop_series_2005.png'
    plt.ylabel('kgC m$^{-2}$ s$^{-1}$')
    plt.suptitle('Daily mean biogenic isoprene emissions',fontsize=26)

    plt.savefig(pname)
    print('SAVED FIGURE ',pname)
    plt.close(f)


def GC_Profile_check(d0=datetime(2005,1,1),d1=datetime(2005,1,31), title=None):
    """
        Compare satellite output hcho profiles
    """
    LatWol, LonWol = pp.__cities__['Wol']
    # Read GC output    
    trop = GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__)
    tropa= GC_class.GC_sat(d0,d1, keys=['IJ-AVG-$_CH2O']+GC_class.__gc_tropcolumn_keys__, run='new_emissions')
    # make sure pedges and pmids are created
    trop.add_pedges()
    dates=trop.dates
    
    # colours for trop and tropa
    c = 'r'
    ca= 'm'
    
    # grab wollongong square
    Woli, Wolj = util.lat_lon_index(LatWol,LonWol,trop.lats,trop.lons) # lat, lon indices
    GC_VMR  = trop.hcho[:,Woli,Wolj,:]
    GCa_VMR = tropa.hcho[:,Woli,Wolj,:]
    
    GC_zmids=trop.zmids[:,Woli,Wolj,:]
    
    # check profile
    # TODO: split into summer and winter
    plt.figure(figsize=[10,10])
    #ax0=plt.subplot(1,2,1)
    for i,prof in enumerate([GC_VMR,GCa_VMR]):
        zmids = np.nanmean(GC_zmids,axis=0)/1000.0
        #pmids = np.nanmean(GC_pmids[0:20,:],axis=0)
        
        mean = np.nanmean(prof,axis=0)
        lq = np.nanpercentile(prof, 25, axis=0)
        uq = np.nanpercentile(prof, 75, axis=0)
        plt.fill_betweenx(zmids, lq, uq, alpha=0.5, color=[c,ca][i])
        plt.plot(mean,zmids,label=['VMR','VMR$^{\\alpha}$'][i],linewidth=2,color=[c,ca][i])
    #plt.yscale('log')
    plt.ylim([0, 40])
    plt.ylabel('altitude [km]')
    plt.legend(fontsize=20)
    plt.xlabel('HCHO [ppbv]')
    if title is None:
        title="Wollongong midday HCHO profile Jan 2005"
    plt.title(title)
    pname_checkprof='Figs/check_GC_profile.png'
    plt.savefig(pname_checkprof)
    print("Saved ", pname_checkprof)



def hcho_ozone_timeseries(d0,d1):
    '''
    '''
    suptitle_prefix='Daily'
    dstr = d0.strftime("%Y%m%d_") + d1.strftime("%Y%m%d")
    pname1 = 'Figs/new_emiss/HCHO_total_columns_%s.png'%dstr
    pname2 = 'Figs/new_emiss/O3_surface_%s.png'%dstr

    satkeys = ['IJ-AVG-$_ISOP', 'IJ-AVG-$_CH2O',
               #'IJ-AVG-$_NO2',     # NO2 in ppbv
               'IJ-AVG-$_O3',      # O3 in ppbv
               ] + GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('GEOS-Chem satellite outputs read')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)

    new_sat_tc  = new_sat.get_total_columns(keys=['hcho'])
    tropchem_sat_tc  = tropchem_sat.get_total_columns(keys=['hcho'])
    # TOTAL column HCHO
    new_hcho_tc = new_sat_tc['hcho']
    tropchem_hcho_tc = tropchem_sat_tc['hcho']

    # Surface O3 in ppb
    new_o3_surf = new_sat.O3[:,:,:,0]
    tropchem_o3_surf = tropchem_sat.O3[:,:,:,0]


    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates

    ## read old satellite hcho columns...
    # OMI total columns, PP corrected total columns
    Enew = E_new(d0, d1, dkeys=['VCC_OMI','VCC_PP'])

    # grab total columns
    vcc_omi     = Enew.VCC_OMI
    vcc_pp      = Enew.VCC_PP
    lats2, lons2= Enew.lats, Enew.lons
    # Enew lats,lons are in high resolution

    # pull out regions and compare time series
    new_sat_o3s, r_lats, r_lons = util.pull_out_subregions(new_o3_surf,
                                                           lats, lons,
                                                           subregions=pp.__subregions__)
    tropchem_sat_o3s, r_lats, r_lons = util.pull_out_subregions(tropchem_o3_surf,
                                                                lats, lons,
                                                                subregions=pp.__subregions__)

    new_sat_hchos, r_lats, r_lons = util.pull_out_subregions(new_hcho_tc,
                                                         lats, lons,
                                                         subregions=pp.__subregions__)

    tropchem_sat_hchos, r_lats, r_lons = util.pull_out_subregions(tropchem_hcho_tc,
                                                         lats, lons,
                                                         subregions=pp.__subregions__)

    vcc_omis, r_lats2, r_lons2 = util.pull_out_subregions(vcc_omi,
                                                         lats2, lons2,
                                                         subregions=pp.__subregions__)

    vcc_pps, r_lats3, r_lons3 = util.pull_out_subregions(vcc_pp,
                                                         lats2, lons2,
                                                         subregions=pp.__subregions__)

    # for plotting we may want daily, weekly, or monthly averages
    def baseresample(datain, bins='M'):
        return pd.Series(datain, index=dates).resample(bins)

    # by default do nothing
    resample = lambda datain : datain
    newdates = dates
    suptitle_prefix = 'Daily'

    if (d1-d0).days > 100: # after 100 days switch to weekly averages
        suptitle_prefix='Weekly'
        bins='7D'
        if (d1-d0).days > 500: # use monthly for > 500 days
            suptitle_prefix='Monthly'
            bins='M'
        resample = lambda datain: np.array(baseresample(datain, bins).mean())
        newdates = baseresample(np.arange(len(dates)), bins).mean().index.to_pydatetime()


    # will be printing mean difference between estimates
    print('area,   new_emiss hcho,   tropchem hcho,   OMI hcho,       OMI$_{PP}$ hcho, new_emiss O3, tropchem O3')


    f1,axes1 = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    f2,axes2 = plt.subplots(6, figsize=(14,16), sharex=True, sharey=True)
    for i, [label, color] in enumerate(zip(pp.__subregions_labels__, pp.__subregions_colors__)):

        # plot time series for each subregion
        hcho_new_emiss = np.nanmean(new_sat_hchos[i], axis=(1,2)) # daily
        hcho_tropchem = np.nanmean(tropchem_sat_hchos[i],axis=(1,2))
        o3_new_emiss = np.nanmean(new_sat_o3s[i], axis=(1,2))
        o3_tropchem = np.nanmean(tropchem_sat_o3s[i], axis=(1,2))


        # change hourly into daily time series
        #r_old = np.array(pd.Series(r_old_hourly,index=old_dates).resample('D').mean())
        hcho_omi = np.nanmean(vcc_omis[i],axis=(1,2)) # daily
        hcho_pp  = np.nanmean(vcc_pps[i],axis=(1,2)) # daily


        # resample daily into something else:
        hcho_new_emiss = resample(hcho_new_emiss)
        hcho_tropchem = resample(hcho_tropchem)
        hcho_omi = resample(hcho_omi)
        hcho_pp = resample(hcho_pp)
        o3_new_emiss = resample(o3_new_emiss)
        o3_tropchem = resample(o3_tropchem)

        arr1 = np.array([np.nanmean(hcho_new_emiss), np.nanmean(hcho_tropchem), np.nanmean(hcho_omi), np.nanmean(hcho_pp)])
        print(label, arr1[0], arr1[1], arr1[2], arr1[3])
        print('   ,', 100*(arr1 - arr1[2])/arr1[2], 'difference from OMI orig')
        print('   ,', 100*(arr1 - arr1[3])/arr1[3], ' difference from OMI PP')
        
        print( 'Ozone: new, old, rel-diff', np.nanmean(o3_new_emiss), np.nanmean(o3_tropchem), (np.nanmean(o3_new_emiss)-np.nanmean(o3_tropchem))*100/np.nanmean(o3_tropchem))

        # Fig1: HCHO time series
        plt.sca(axes1[i])
        pp.plot_time_series(newdates,hcho_new_emiss, label='$\Omega_{GC}^{\\alpha}$', linestyle='-.', color=color, linewidth=2)
        pp.plot_time_series(newdates,hcho_tropchem, label='$\Omega_{GC}$', linestyle='--', color=color, linewidth=2)
        pp.plot_time_series(newdates,hcho_omi, dfmt='%Y%m%d', label='$\Omega_{OMI}$', linestyle='-', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_pp, label='OMI recalculated', linestyle=':', color=color, linewidth=2)
        plt.title(label,fontsize=20)
        if i==0:
            plt.ylabel('HCHO cm$^{-2}$')
            plt.legend(loc='best',ncol=3)

        # Fig2: Ozone timeseries
        plt.sca(axes2[i])
        pp.plot_time_series(newdates,o3_new_emiss, label='new_emiss run', linestyle=':', color=color, linewidth=2)
        pp.plot_time_series(newdates,o3_tropchem, label='tropchem run', linestyle='--', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_omi, dfmt='%Y%m%d', label='OMI', linestyle='-', color=color, linewidth=2)
        #pp.plot_time_series(newdates,hcho_pp, label='OMI recalculated', linestyle=':', color=color, linewidth=2)
        plt.title(label,fontsize=20)
  


    # final touches figure 1
    plt.legend(loc='best',)
    for ii in [0,i]:
        plt.sca(axes1[ii])
        plt.ylabel('HCHO cm$^{-2}$')
    plt.suptitle('%s mean $\Omega_{HCHO}$'%suptitle_prefix, fontsize=26)

    plt.savefig(pname1)
    print('SAVED FIGURE ',pname1)
    plt.close(f1)

    # final touches figure 1
    for ii in [0,i]:
        plt.sca(axes2[ii])
        plt.ylabel('O$_3$ ppb')
    plt.suptitle('%s mean O$_3$ tropospheric column'%suptitle_prefix, fontsize=26)

    plt.savefig(pname2)
    print('SAVED FIGURE ',pname2)
    plt.close(f2)

def print_ozone_isop_table_summary():
    '''
      Metric & AUS & SEA & NEA & NA & SWA & MID \\
      \midrule
      MEGAN & 43(2) & blah &  &  & & \\
      Ozone & 9.70 & 11.17 & 11.03 & 11.19 & 11.69 & 9.09 \\
      \midrule
      Top-Down & 19(2) & & & & & \\
      Ozone & 9.64 & 11.11 & 10.99 & 11.12 & 11.63 & 9.02 \\
      \bottomrule
    \end{tabular}
    '''
    from utilities import GMAO
    from scipy.constants import N_A # avegaadro's number
    
    d0=datetime(2005,1,1)
    #d1=datetime(2012,12,31)
    d1=datetime(2005,1,31)
    satkeys = [#'IJ-AVG-$_ISOP', # ppbv 
               #'IJ-AVG-$_CH2O', # ppbv
               #'IJ-AVG-$_NO2',     # NO2 in ppbv
               'IJ-AVG-$_O3',      # O3 in ppbv
               ]# + GC_class.__gc_tropcolumn_keys__
    new_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    tropchem_sat = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('GEOS-Chem satellite outputs read')
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)

    # Surface O3 in ppb
    new_o3_surf = new_sat.O3[:,:,:,0]
    tropchem_o3_surf = tropchem_sat.O3[:,:,:,0]

    # dims for GEOS-Chem outputs
    lats=new_sat.lats
    lons=new_sat.lons
    dates=new_sat.dates
    
    del new_sat
    del tropchem_sat 
    
    # Read a priori and a posteriori
    # Read megan (3-hourly)
    MEGAN = GC_class.Hemco_diag(d0,d1)
    meglats,meglons = MEGAN.lats, MEGAN.lons
    aprior = MEGAN.E_isop_bio # hours, lats, lons
    aprior = aprior / MEGAN.kgC_per_m2_to_atomC_per_cm2 # convert back to kgC/m2/s
    del MEGAN # free up some ram
    aprior = np.nansum(aprior*3*60*60, axis=0) # sum over the 3 hourly kgC/m2/s to get kgC/m2
    assert np.all(np.shape(aprior)==np.shape(GMAO.area_m2)), "Area from GMAO does not match shape of emissions from MEGAN"
    aprior = aprior * GMAO.area_m2 # now aprior is in kg total over the length of time
    aprior = aprior/8.0 # now is TgC/a
    
    enew=E_new(d0,d1,dkeys=['E_PP_lr'])
    enewlats,enewlons=enew.lats_lr, enew.lons_lr
    
    apost = enew.E_PP_lr # [days,lats,lons] atomC/cm2/s 
    # convert from peak emissions to daily sin wave integrated
    # 0.637 x peak x sunlight seconds
    daily_seconds = util.daylengths_matched(enew.dates) * 60 # daylight seconds 
    daily_seconds = np.repeat(daily_seconds[:,np.newaxis],len(enewlats),axis=1)
    daily_seconds = np.repeat(daily_seconds[:,:,np.newaxis],len(enewlons),axis=2)
    apost = apost * 0.637 * daily_seconds # now in daily atomC/cm2
    apost = np.nansum(apost, axis=0) # sum over daily atomC/cm2 to get atomC/cm2
    apost = apost * enew.SA_lr * 1e10 # multiply by cm2  (SA in km2) to get atomC emitted
    # atomC * mol/atom * g/mol = g C 
    apost = (apost / N_A) * util.__grams_per_mole__['carbon']
    # we have 8 years totalled, divide to get TgC/a
    apost = (apost/8.0) /1e12 # 1e12 g/Tg
    
    # pull out regions and compare time series
    apriors, r_lats, r_lons = util.pull_out_subregions(aprior, meglats, meglons, subregions=pp.__subregions__)
    aposts, r_lats, r_lons = util.pull_out_subregions(apost, enewlats, enewlons, subregions=pp.__subregions__)
    
    new_sat_o3s, r_lats, r_lons = util.pull_out_subregions(new_o3_surf,
                                                           lats, lons,
                                                           subregions=pp.__subregions__)
    tropchem_sat_o3s, r_lats, r_lons = util.pull_out_subregions(tropchem_o3_surf,
                                                                lats, lons,
                                                                subregions=pp.__subregions__)

    # Build up an array with 6 columns, 4 rows:
    #                        AUS,  SEA, NEA, NA, SWA, MID
    #  isop TgC/a tropchem   ...
    #  O3   ppb   tropchem   ...
    #  isop TgC/a top-down   ...
    #  O3   ppb   top-down
    #  
    table = np.zeros(4,6)
    
    # O3 ppb surface from tropchem and top-down
    table[3,:] = [np.nanmean(new_sat_o3s[i]) for i in range(6)]
    table[1,:] = [np.nanmean(tropchem_sat_o3s[i]) for i in range(6)]
    
    #isop TgC/a megan
    table[0,:] = [np.nansum(apriors[i]) for i in range(6)]
    #isop TgC/a top-down
    table[2,:] = [np.nansum(aposts[i]) for i in range(6)]
    
    print("Isoprene emissions in TgC/a, O3 in ppb")
    print(pp.__subregions_labels__)
    formstring = "%%5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f \\\\"
    print("TABLE=============")
    print("tropchem &&&&&&")
    print("isoprene & "+formstring%table[0,:])
    print("O3       & "+formstring%table[1,:])
    print("scaled   &&&&&&")
    print("isoprene & "+formstring%table[2,:])
    print("O3       & "+formstring%table[3,:])
    
    

def spatial_comparisons(d0, d1, dlabel):
    ''' Compare HCHO, O3, NO columns between runs over Australia averaged over input dates '''

    #dstr = d0.strftime("%Y%m%d")
    pname1 = 'Figs/new_emiss/HCHO_total_columns_map_%s.png'%dlabel
    pname2 = 'Figs/new_emiss/O3_surf_map_%s.png'%dlabel
    pname3 = 'Figs/new_emiss/NO_surf_map_%s.png'%dlabel

    satkeys = ['IJ-AVG-$_ISOP', 'IJ-AVG-$_CH2O',
               'IJ-AVG-$_NO2',     # NO2 in ppbv
               'IJ-AVG-$_O3', ] + GC_class.__gc_tropcolumn_keys__
    
    
    GCnew = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='new_emissions')
    GCtrop = GC_class.GC_sat(day0=d0,dayN=d1, keys=satkeys, run='tropchem')
    print('GEOS-Chem satellite outputs read 2005')
    lats=GCnew.lats
    lons=GCnew.lons
    # new_sat.hcho.shape #(31, 91, 144, 47)
    # new_sat.isop.shape #(31, 91, 144, 47)

    # TOTAL column HCHO
    new_hcho  = GCnew.get_total_columns(keys=['hcho'])['hcho']
    trop_hcho  = GCtrop.get_total_columns(keys=['hcho'])['hcho']
    # Average temporally
    new_hcho_map = np.nanmean(new_hcho,axis=0)
    trop_hcho_map = np.nanmean(trop_hcho,axis=0)

    # surface O3 in ppb
    new_o3 = GCnew.O3[:,:,:,0]
    trop_o3 = GCtrop.O3[:,:,:,0]
    new_o3_map = np.nanmean(new_o3, axis=0)
    trop_o3_map = np.nanmean(trop_o3, axis=0)
    
    print(GCnew.attrs['NO2'])
    new_NO2 = GCnew.NO2[:,:,:,0]
    trop_NO2 = GCtrop.NO2[:,:,:,0]
    new_NO2_map = np.nanmean(new_NO2, axis=0)
    trop_NO2_map = np.nanmean(trop_NO2, axis=0)
    
    
    ## read old satellite hcho columns...
    # OMI total columns, PP corrected total columns
    Enew = E_new(d0, d1, dkeys=['VCC_OMI','VCC_PP','pixels_PP_u']) # unfiltered pixel counts

    # grab total columns
    vcc_omi     = Enew.VCC_OMI
    vcc_pp      = Enew.VCC_PP
    pixels_pp   = Enew.pixels_PP_u
    lats2, lons2= Enew.lats, Enew.lons
    lats_lr     = Enew.lats_lr
    lons_lr     = Enew.lons_lr

    # Get VCC in lower resolution
    vcc_pp_lr=np.zeros([len(Enew.dates), len(lats_lr), len(lons_lr)])+np.NaN
    pixels_pp_lr=np.zeros([len(Enew.dates), len(lats_lr), len(lons_lr)])
    for i in range(vcc_pp.shape[0]):
        vcc_pp_lr[i]    = util.regrid_to_lower(vcc_pp[i],lats2,lons2,lats_lr,lons_lr,pixels=pixels_pp[i])
        pixels_pp_lr[i] = util.regrid_to_lower(pixels_pp[i],lats2,lons2,lats_lr,lons_lr,func=np.nansum)

    omi_pp_map, omi_pp_map_pixels  = util.satellite_mean(vcc_pp_lr, pixels_pp_lr, spatial=False, temporal=True)

    # plot some test maps
    # one plot for hcho trop columns, similar for surface O3, and then for NO
    # order: hcho, O3, NO
    vmins = [1e15, 10, 0]
    vmaxs = [1.8e16, 40, 0.4]
    amins = [-5e14, -5, -0.05]
    amaxs = [5e14, 5, .05]
    rlims = [[-40,40],[-10,10],[-10,10]]
    units = ['molec cm$^{-2}$', 'ppbv', 'ppbv']
    linears= [False,True,True]
    comparison_plots = [ omi_pp_map, None, None ]
    comparison_titles= ['$\Omega_{OMI}$', '', '']
    comparison_lats  = [lats_lr, None, None]
    comparison_lons  = [lons_lr, None, None]
    first_maps      = [trop_hcho_map, trop_o3_map, trop_NO2_map]
    first_title     ='tropchem run' 
    second_maps       = [new_hcho_map, new_o3_map, new_NO2_map]
    second_title   = 'scaled run'
    pnames = [pname1,pname2,pname3]
    stitles = ['Midday total column HCHO: %s'%dlabel,
                'GEOS-Chem surface ozone %s'%dlabel,
                'GEOS-Chem surface NO$_2$ %s'%dlabel]

    for i in range(3):
        vmin=vmins[i]; vmax=vmaxs[i]; unit=units[i]; pname=pnames[i]
        amin=amins[i]; amax=amaxs[i]
        rmin,rmax = rlims[i]
        
        linear=linears[i]
        f=plt.figure(figsize=[14,14])

        plt.subplot(3,2,1)
        pp.createmap(first_maps[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, clabel=unit, linear=linear)
        plt.title(first_title)

        plt.subplot(3,2,2)
        pp.createmap(second_maps[i],lats,lons,aus=True, vmin=vmin,vmax=vmax, clabel=unit, linear=linear)
        plt.title(second_title)
        
        # add rel abs diff
        plt.subplot(3,2,3)
        pp.createmap(first_maps[i] - second_maps[i],lats,lons,aus=True, 
                     vmin=amin,vmax=amax, clabel=unit, linear=True,
                     cmapname='bwr')
        plt.title('tropchem - scaled')
    
        plt.subplot(3,2,4)
        pp.createmap(100*(first_maps[i] - second_maps[i])/second_maps[i],lats,lons,aus=True, 
                     vmin=rmin,vmax=rmax, clabel='%', linear=True,
                     cmapname='bwr')
        plt.title('tropchem - scaled [%]')
        
        if comparison_plots[i] is not None:
            plt.subplot(3,2,5)
            pp.createmap(comparison_plots[i], comparison_lats[i], comparison_lons[i], aus=True, vmin=vmin,vmax=vmax, clabel=unit, linear=linear)
            plt.title(comparison_titles[i])
        
            plt.subplot(3,2,6)
        else:
            plt.subplot(3,1,3)
        # three way regression if possible
        subsets = util.lat_lon_subset(lats,lons,pp.__AUSREGION__,[first_maps[i],second_maps[i]],has_time_dim=False)
        X=subsets['data'][0].flatten() # new hcho map
        Y=subsets['data'][1].flatten() # trop hcho map
        #Z= scatter coloured by value of OMI PP 
        plt.scatter(X,Y)
        pp.add_regression(X,Y)
        plt.legend(loc='best',fontsize=18)
        
        plt.plot([vmin,vmax],[vmin,vmax], 'k--')# black dotted 1-1 line
        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)
        plt.title('Surface O$_3$')
        plt.ylabel('scaled run [%s]'%unit)
        plt.xlabel('tropchem run [%s]'%unit)
        #pp.createmap(new_hcho_map,lats,lons,aus=True,title='scaled run', vmin=vmin,vmax=vmax)

        plt.suptitle(stitles[i])
        plt.subplots_adjust(hspace=0.29)
        plt.savefig(pname)
        print('Saved ', pname)
        plt.close(f)

def trend_emissions(d0 = datetime(2005,1,1), d1=datetime(2012,12,31)):
    '''
        plot emissions and look for trends
    '''
    # plot name, titles, labels ...
    pname='Figs/Emiss/trend_E.png'
    titles=['isoprene a priori anomaly', 'isoprene a postiori anomaly']
    unitss=['atom C cm$^{-2}$ s$^{-1}$','atom C cm$^{-2}$ s$^{-1}$']
    
    # look at each region
    regions   = util.__subregions__
    colors    = util.__subregions_colors__
    labels    = util.__subregions_labels__
    n_regions = len(regions)
    
    # read apriori and postiori emissions
    enew    = E_new(d0,d1,dkeys=['E_MEGAN','E_PP_lr'])
    lats    = enew.lats_lr
    lons    = enew.lons_lr
    dates   = enew.dates
    months  = util.list_months(d0,d1)
    
    # Grab the emissions priori and postiori
    apri = enew.E_MEGAN
    apos = enew.E_PP_lr
    
    # plot anomaly and regression for emissions
    f, axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    
    for j, (arr,title,units) in enumerate(zip([apri, apos],titles,unitss)):
        #mya_df = util.multi_year_average_regional(arr, dates, lats, lons, grain='monthly', regions=regions)
        #mya = [np.squeeze(np.array(mya_df['df'][i].mean())) for i in range(n_regions)]
        # pull out subregions, keeping lats and lons
        regional, lats_regional, lons_regional = util.pull_out_subregions(arr,lats,lons,subregions=regions)
        
        # average spatial dims into monthly time series
        regional_ts = [ np.nanmean(regional[i], axis=(1,2)) for i in range(n_regions) ]
        
        # get trend
        trends = [ util.trend(regional_ts[i], dates, remove_mya=True, resample_monthly=True, remove_outliers=True) for i in range(n_regions) ]
        
        #anomaly = [ trends[i]['anomaly'] for i in range(n_regions) ]
        #monthly = [ trends[i]['monthly'] for i in range(n_regions) ]
        
        #monthly = [ np.array(util.resample(np.nanmean(regional[i],axis=(1,2)),dates,bins='M').mean()).squeeze() for i in range(len(regions))]
        #anomaly = [ np.array([ old_monthly[k][i] - mya[k][i%12] for i in range(len(months)) ]) for k in range(len(regions)) ]
        
        plt.sca(axes[j])
        print(title)
        print('region, [ slope, regression coeff, p value for slope non zero]')
        for i in range(n_regions):
        #for monthly_anomaly, monthly_data, color, label in zip(anomaly, monthly, colors, labels):
            color=colors[i]; label=labels[i]
            
            trendi=trends[i]
            #monthly = trendi['monthly']
    
            # Calculate with all data (for interest)
            #m, b, r, cir, cijm = RMA(np.arange(len(months)), monthly_anomaly)
            #print("%s (has outliers) &  [ %.2e,  %.2e ]   & \\"%(label,cir[0][0], cir[0][1]))
            
            # Y = mX+b
            anomaly = trendi['anomaly']
            m=trendi['slope']
            b=trendi['intercept']
            r=trendi['r']
            p=trendi['p'] # two sided p value against slope being zero as H0
            outliers=trendi['outliers']
            
            print("%s &  [ m=%.2e,  r=%.3f, p=%.3f ]   & \\"%(label,m, r, p))
            
            # once more with outliers removed
            #std=np.nanstd(monthly_anomaly)
            #mean=np.nanmean(monthly_anomaly)
            #outliers = ( monthly_anomaly > mean + 3 * std ) + ( monthly_anomaly < mean - 3*std )
            #Xin = np.arange(len(months))[~outliers]
            #Yin = monthly_anomaly[~outliers]
            
            #m, b, r, cir, cijm = RMA(Xin, Yin)
            #print("%s (no outliers)  &  [ %.2e,  %.2e ]   & \\"%(label,cir[0][0], cir[0][1]))
            #print(m/np.nanmean(monthly_data) * 12)
            #if cir[0][0] * cir[0][1] > 0: # if slope doesn't surround zero then we plot line
            if p < 0.05:
                pp.plot_time_series( [months[0], months[-1]], [b, m*(len(months)-1)], alpha=0.5, color=color ) # regression line
            pp.plot_time_series( months, anomaly, color=color, label=label, marker='.', markersize=6, linewidth=0) # regression line
            pp.plot_time_series( np.array(months)[outliers], anomaly[outliers], color=color, marker='x', markersize=8, linewidth=0)
        plt.title(title,fontsize=24)
        plt.ylabel(units)
        plt.xticks(util.list_years(d0,d1))
    
    plt.sca(axes[0])
    xywh=(.975, 0.15, 0.1, .7)
    plt.legend(bbox_to_anchor=xywh, loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=12)
    
    plt.savefig(pname)
    print('SAVED ',pname)
    plt.close()

if __name__ == '__main__':

    
    print("Testing alpha creation")

    start=timeit.default_timer()
    alpha_creation()
    end=timeit.default_timer()
    print("TIME: %6.2f minutes for alpha_creation"%((end-start)/60.0))
