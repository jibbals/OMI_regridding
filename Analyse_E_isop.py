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
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')# don't plot on screen, send straight to file
import matplotlib.pyplot as plt
plt.ioff()

from PIL import Image # paste together some plots


# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO
#from utilities import fio
from classes import GC_class, campaign # GC trac_avg class
from classes.omhchorp import omhchorp # OMI product class
from classes.E_new import E_new # E_new class

import Inversion

###############
### Globals ###
###############
__VERBOSE__=True

NA     = util.NA
SWA    = util.SWA
SEA    = util.SEA
subs   = [SWA,NA,SEA]
labels = ['SWA','NA','SEA']
colours = ['chartreuse','magenta','aqua']

###############
### Methods ###
###############

def yearly_megan_cycle(d0=datetime(2005,1,1), dn=datetime(2009,12,31)):
    '''
    '''
    # Read megan
    MEGAN = GC_class.Hemco_diag(d0,dn)
    data=MEGAN.E_isop_bio # hours, lats, lons
    dates=np.array(MEGAN.dates)

    Enew = E_new(d0,dn,dkeys=['E_PP_lr'])
    
    # average over some regions
    regions=pp.__subregions__
    region_colours=pp.__subregions_colors__
    region_labels=pp.__subregions_labels__
    region_means=[]
    region_stds=[]
    topd_means=[]
    topd_stds=[]
    # UTC offsets for x axis
    # region_offsets=[11,10,10,11,12,11]
    offset=10 # utc + 10

    for region in regions:
        # subset MEGAN data and serialise
        subset=util.lat_lon_subset(MEGAN.lats, MEGAN.lons, region, [data], has_time_dim=True)
        series=np.nanmean(subset['data'][0],axis=(1,2))
        
        # subset topd data
        topdsub=util.lat_lon_subset(Enew.lats_lr, Enew.lons_lr, region, [Enew.E_PP_lr], has_time_dim=True)
        topdser=np.nanmean(topdsub['data'][0],axis=(1,2))
        
        # group by month, and hour to get the multi-year monthly averaged diurnal cycle
        monthly_hours=util.multi_year_average(series,dates,grain='hourly')
        monthly_topd=util.multi_year_average(topdser, Enew.dates, grain='monthly')
        # save mean and std into [month, hour] array
        region_means.append(monthly_hours.mean().squeeze().values.reshape([12,24]))
        region_stds.append(monthly_hours.std().squeeze().values.reshape([12,24]))
        topd_means.append(monthly_topd.mean().squeeze().values)
        topd_stds.append(monthly_topd.std().squeeze().values)


    ## set up the plots
    # monthly day cycles : 4 rows 3 columns with shared axes
    f, axes = plt.subplots(4,3, sharex=True, sharey=True, figsize=(16,16))
    axes[3,1].set_xlabel('Hour (UTC+%d)'%offset)
    xlim=[6,22]
    axes[3,1].set_xlim(xlim)
    axes[1,0].set_ylabel('Emission (molec/cm2/s)')
    ylim=[1e10,1e14]
    axes[1,0].set_ylim(ylim)
    axes[1,0].set_yscale('log')
    titles=np.array([['Dec','Jan','Feb'],['Mar','Apr','May'],['Jun','Jul','Aug'],['Sep','Oct','Nov']])

    for r,region in enumerate(regions):
        means = region_means[r]
        stds  = region_stds[r]
        topdm = topd_means[r]
        topds = topd_stds[r]
        #offset= region_offsets[r]

        # plot the daily cycle and std range
        for i in range(4): # 4 rows
            for j in range(3): # 3 columns
                # shift forward by one month to get dec as first entry
                ii, jj = (i+int((j+1)%3==0))%4, (j+1)%3
                # grab month (map i,j onto (0-11)*24)
                #mi=i*3*24 + j*24
                mi=i*3+j #month index
                mip=(mi+1)%12
                # grab mean and std from dataset for this month in this region
                mdata = means[mip,:].squeeze()
                mstd  = stds[mip,:].squeeze()
                mtopd = topdm[mip].squeeze()
                mtopds= topds[mip].squeeze()
                
                # roll over x axis to get local time midday in the middle
                #high  = np.roll(data+std, offset)
                #low   = np.roll(data-std, offset)
                mdata  = np.roll(mdata, offset)

                #plot into monthly panel, and remove ticks
                ax   = axes[ii,jj]
                plt.sca(ax)

                # remove ticks from right and top edges
                plt.tick_params(
                    axis='both',      # changes apply to the x-axis
                    which='both',     # both major and minor ticks are affected
                    right=False,      # ticks along the right edge are off
                    top=False,       # ticks along the top edge are off
                    left=jj==0,
                    bottom=ii==3)

                # first highlight the 1300-1400 time window with soft grey
                #plt.fill_betweenx(ylim,[13,13],[14,14], color='grey', alpha=0.2)
                plt.plot([13,13], ylim, color='grey',alpha=0.5)
                plt.plot([14,14], ylim, color='grey',alpha=0.5)
                

                #plt.fill_between(np.arange(24), high, low, color='k')
                plt.plot(np.arange(24), mdata, color=region_colours[r])
                plt.title(titles[ii,jj])
        
                # also plot topd from 1300-1400
                plt.plot([12,15], [mtopd, mtopd], color=region_colours[r], linewidth=2)
                plt.plot([12,15], [mtopd+mtopds, mtopd+mtopds], color=region_colours[r], linewidth=1)
                plt.plot([12,15], [mtopd-mtopds, mtopd-mtopds], color=region_colours[r], linewidth=1)



    # remove gaps between plots
    f.subplots_adjust(wspace=0, hspace=0.1)
    pname='Figs/Emiss/MEGAN_monthly_daycycle.png'
    plt.savefig(pname)
    print('SAVED FIGURE ',pname)
    plt.close()


def multiyear_vs_campaigns(d0=datetime(2005,1,1),dn=datetime(2007,12,31)):
    '''
        Multiyear average peak emissions per day vs daily peak campaign concentrations

    '''
    pname='yearly_cycle_vs_campaigns.png'
    mumba = campaign.mumba()
    sps1  = campaign.sps(1)
    sps2  = campaign.sps(2)

    colors=['m','pink','orange']


    ## READ E_NEW
    #
    Enew  = E_new(d0,d1,enew_keys)
    lats,lons=Enew.lats_lr,Enew.lons_lr
    yi,xi = util.lat_lon_index(lat,lon,lats,lons)
    Eomi = Enew.E_PP_lr[:,yi,xi]
    Eomi[Eomi<0]=0
    dates=Enew.dates

    ## READ E_MEGAN
    #

    plt.figure(figsize=(16,8))
    # create map, with gridsquare of comparison, and dots for campaigns
    #    plt.subplot(2,1,1)
    #    m=pp.displaymap(region=[-45,130,-14,155])
    #    pp.add_grid_to_map(m,)
    #    # Add dot to map
    #    for i,[y,x] in enumerate([[mumba.lat,mumba.lon],[sps1.lat,sps1.lon],[sps2.lat,sps2.lon]]):
    #        mx,my = m(x, y)
    #        m.plot(mx, my, 'o', markersize=3, color=colors[i])

    #plt.subplot(2,1,2)
    ax=plt.gca()
    dates,isop=mumba.get_daily_hour(key='isop')
    d1,i1=sps1.get_daily_hour(key='isop')
    d2,i2=sps2.get_daily_hour(key='isop')


    pp.plot_yearly_cycle(isop,dates,color='m',label='MUMBA',linewidth=2)
    pp.plot_yearly_cycle(i1,d1,color='pink',label='SPS1',linewidth=2)
    pp.plot_yearly_cycle(i2,d2,color='orange',label='SPS2',linewidth=2)

    plt.legend()
    plt.title('isoprene yearly cycle')
    plt.tight_layout()
    plt.savefig(pname)

def campaign_vs_emissions():
    '''
        Compare campaign data to Enew and Egc in that one grid square
    '''
    # Read campaign data
    mumba = campaign.mumba()
    sps1  = campaign.sps(1)
    sps2  = campaign.sps(2)

    colors=['m','pink','orange']




def check_E_new(d0=datetime(2005,1,1),dn=datetime(2005,1,31),region=pp.__AUSREGION__, plotswaths=False):
    '''
        Print out averages and anomalies in time series
    '''
    # Read data
    Enew=E_new(d0,dn)
    dates,E_isop=Enew.get_series('E_OMI',region=region, testplot=True)

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

def E_time_series(d0=datetime(2005,1,1),dn=datetime(2006,12,31),
                  lat=pp.__cities__['Syd'][0],lon=pp.__cities__['Syd'][1]-0.5,
                  locname='Sydney',count=False, plot_error=False,
                  monthly=False, monthly_func='median',
                  ylims=None):
    '''
        Plot the time series of E_new, eventually compare against MEGAN, etc..
        Look at E_OMI, E_GC, and E_PP

        Add vertical lines for monthly uncertainty(stdev,)

    '''

    # Read in emissions from E_new
    enew_keys=['E_PP_lr']
    #allkeys=['E_OMI', 'E_GC', 'E_PP','E_OMI_lr','E_GC_lr','E_PP_lr',
    #         'E_MEGAN', 'pixels','pixels_PP','pixels_lr','pixels_PP_lr']
    Enew  = E_new(d0,dn,enew_keys)
    E_meg = GC_class.Hemco_diag(d0,dn)

    pp.InitMatplotlib() # set up plotting defaults
    # Time series plots, how displaying each line
    linewidths  = [2,1,1,1]
    colours     = ['k','m','orange','red']

    for lowres in [True,False]:
        # Low res or not changes plotname and other stuff
        lrstr=['','_lr'][lowres]
        pname='Figs/Emiss/E_new_series_%s%s.png'%(locname,lrstr)
        # key names for reading E_new
        ekeys = [ek+lrstr for ek in ['E_OMI', 'E_GC', 'E_PP']]
        labels      = ['MEGAN',ekeys[0],ekeys[1],ekeys[2] ]
        pixels = getattr(Enew, 'pixels'+lrstr)
        pixelspp = getattr(Enew, 'pixels_PP'+lrstr)

        # Plot four time series
        f,axs = plt.subplots(1+count,1,figsize=(16,8), sharex=True)
        ax0=axs
        if count:
            ax0=axs[0]
        # Grab desired E_new data
        E_omi, E_gc, E_pp = [getattr(Enew,ekeys[i]) for i in range(3)]
        lats,lons = [ getattr(Enew,s+str.lower(lrstr)) for s in ['lats','lons'] ]
        enewlati,enewloni = util.lat_lon_index(lat,lon,lats,lons)
        meglati, megloni  = util.lat_lon_index(lat,lon,Enew.lats_lr,Enew.lons_lr)

        for i,arr in enumerate([E_meg,E_omi,E_gc,E_pp]):
            dates=Enew.dates
            lati = [enewlati,meglati][i==0]
            loni = [enewloni,megloni][i==0]
            arr = arr[:,lati,loni]
            pix = [pixels, pixelspp][i==3]
            pix=pix[:,lati,loni]
            # plot time series
            plt.sca(ax0)
            monthly_data    = util.monthly_averaged(dates,arr)
            monthly_pix     = util.monthly_averaged(dates,pix)
            mpix            = monthly_pix['sum']
            marr            = monthly_data[monthly_func]
            std             = monthly_data['std']
            mdates          = monthly_data['middates']
            pp.plot_time_series(mdates,marr,
                                linewidth=linewidths[i],
                                color=colours[i],label=labels[i])

            if i>0:
                # error points
                if plot_error:

                    pp.plot_time_series(mdates,marr+std,
                                        color=colours[i],marker='^',linestyle='None',
                                        markerfacecolor='None',markersize=4, markeredgecolor=colours[i])
                    pp.plot_time_series(mdates,marr-std,
                                        color=colours[i],marker='v',linestyle='None',
                                        markerfacecolor='None',markersize=4, markeredgecolor=colours[i])
                # plot pixel counts
                if count:
                    plt.sca(axs[1])
                    pp.plot_time_series(mdates,mpix,color=colours[i])
                    #df_m['pix'].plot(color=colours[i])
                    #plt.plot(pix, color=colours[i])

        plt.sca(ax0)
        if ylims is not None:
            plt.ylim(ylims)
        plt.title(locname+ ' [%.2f$^{\circ}$N, %.2f$^{\circ}$E]'%(lat,lon))
        plt.ylabel(Enew.attributes[ekeys[0]]['units'])
        plt.legend(loc='best')
        if count:
            plt.sca(axs[1])
            plt.ylabel('pixels (sum)')
        plt.xlabel('time')
        plt.savefig(pname)
        plt.close()
        print("SAVED ",pname)


def Emissions_weight():
    '''
        Print out emissions in molec/cm2/s and kg/s for each season
    '''
    d0=datetime(2005,1,1)
    de=datetime(2007,12,31)
    print('TODO: extend to 2013 when ready')
    Enew=E_new(d0,de)
    Enew.conversion_to_kg


def E_regional_time_series(d0=datetime(2005,1,1),dn=datetime(2007,12,31),
                           etype='pp', lowres=True, showmaps=False,
                           force_monthly=True, force_monthly_func='median'):
    '''
        Plot the time series of E_new, compare against MEGAN, etc..
        Look at E_OMI, E_GC, and E_PP
        Averaged within some regions

        currently can't compare high-res E_new to low res E_MEGAN

        Plot:    MAP ENEW    |    MAP MEGAN
                 time series |    time series
                 differences one row per region
    '''
    # Low res or not changes plotname and other stuff
    lrstr=['','_lr'][lowres]
    etype=str.upper(etype)
    ekey= 'E_'+etype+lrstr
    mstr=['','_monthly'][force_monthly]
    pname='Figs/Emiss/E_zones_%s%s%s.png'%(etype,mstr,lrstr)
    vmin=0; vmax=1.2e13

    # Read in E_new and E_MEGAN
    Enew  = E_new(d0,dn,[ekey,'E_MEGAN'])
    E = getattr(Enew,ekey)
    E[E<0] = 0 # remove negative emissions.
    Emeg= Enew.E_MEGAN
    lats,lons = [ getattr(Enew,s+str.lower(lrstr)) for s in ['lats','lons'] ]
    dates=Enew.dates

    f=plt.figure(figsize=[16,12])

    # Plots map, and time series of E over some subzones
    # just want Austalia and sydney area
    subzones=pp.__subregions__
    colors=pp.__subregions_colors__
    labels=pp.__subregions_labels__
    axs,series,series2,dates=pp.subzones(E,dates,lats,lons, comparison=Emeg, linear=True,
                                         vmin=vmin, vmax=vmax, maskoceans=True,
                                         mapvmin=0, mapvmax=5e12,
                                         labelcities=False,labels=labels,
                                         subzones=subzones,colors=colors,
                                         showmaps=showmaps,
                                         force_monthly=force_monthly, force_monthly_func=force_monthly_func,
                                         title='E$_{OMI}$', comparisontitle='E$_{GC}$')
    axs[0].set_title('E$_{OMI}$',fontsize=30)
    axs[1].set_title('E$_{GC}$',fontsize=30)
    axs[0].set_ylabel('molec cm$^{-2}$ s$^{-1}$')

    # add row at bottom for differences between Enew and Emeg
    f.subplots_adjust(bottom=0.37)
    pos2=axs[0].get_position()
    pos3=axs[1].get_position()
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(20)
    for tick in axs[1].get_xticklabels():
        tick.set_rotation(20)

    axnew=f.add_axes([pos2.x0, 0.05, pos3.x0 + pos3.width - pos2.x0 , 0.2],) #sharey=axs[3]) # share x axis with time series
    axs[0].set_ylim([vmin,vmax])
    axs[1].set_ylim([vmin,vmax])
    plt.sca(axnew)
    # let's plot the difference for each subregion

    for i in range(len(series)):
        # plot megan -enew
        lw=[1,3][i==0] # wider black line
        absdif = series[i]-series2[i]
        plt.plot_date(dates, -1*absdif, fmt='-', color=colors[i], linewidth=lw)
        #reldif = 100*(series[i]-series2[i]) / series2[i]
        #plt.plot_date(dates, reldif, fmt='-', color=colors[i], linewidth=lw)

    plt.ylim([-0.3e13, 1e13])
    plt.ylabel('molec cm$^{-2}$ s$^{-1}$')
    plt.title('(E$_{GC}$-E$_{OMI}$)')
    plt.plot_date([dates[0],dates[-1]],[0,0],'--k')

    f.autofmt_xdate()

    # save figure
    plt.savefig(pname)
    print("Saved %s"%pname)
    plt.close()

def E_regional_multiyear(d0=datetime(2005,1,1),dn=datetime(2005,12,31),
                         etype='pp', lowres=True):
    '''
    Plot the time series of E_new, compare against MEGAN, using multi-year averages monthly averages and std's at midday
    Look at E_OMI, E_GC, and E_PP
    Averaged within some regions

    currently can't compare high-res E_new to low res E_MEGAN

    Plot:    MAP ENEW    |    MAP MEGAN
             time series |    time series
             differences one row per region
    '''
    # Low res or not changes plotname and other stuff
    lrstr=['','_lr'][lowres]
    etype=str.upper(etype)
    ekey= 'E_'+etype+lrstr
    pname='Figs/Emiss/E_zones_multiyear_%s%s.png'%(etype,lrstr)
    regions=GMAO.__subregions__
    colors=GMAO.__subregions_colors__
    labels=GMAO.__subregions_labels__

    # Read in E_new and E_MEGAN
    Enew  = E_new(d0,dn,[ekey,'E_MEGAN'])
    E = getattr(Enew,ekey)
    E[E<0] = np.NaN # remove negative and zero emissions.
    Emeg= Enew.E_MEGAN

    # remove ocean squares
    lats,lons       = [ getattr(Enew,s+str.lower(lrstr)) for s in ['lats','lons'] ]
    dates           = Enew.dates
    oceanmask       = util.oceanmask(lats,lons)
    oceanmask       = np.repeat(oceanmask[np.newaxis,:,:], len(dates), axis=0)
    Emeg[oceanmask] = np.NaN
    E[oceanmask]    = np.NaN

    # multi-year monthly averages
    myaE    = util.multi_year_average_regional(E,dates,lats,lons,grain='monthly',regions=regions)
    myaEmeg = util.multi_year_average_regional(Emeg,dates,lats,lons,grain='monthly',regions=regions)
    dfE     = myaE['df']
    dfEmeg  = myaEmeg['df']

    x=range(12)
    n=len(dfE)
    f,axes = plt.subplots(n,1,figsize=[16,12], sharex=True,sharey=True)
    for i in range(n):
        plt.sca(axes[i])
        mean        = dfE[i].mean().values.squeeze()
        uq          = dfE[i].quantile(0.75).values.squeeze()
        lq          = dfE[i].quantile(0.25).values.squeeze()
        meanmeg     = dfEmeg[i].mean().values.squeeze()
        uqmeg       = dfEmeg[i].quantile(0.75).values.squeeze()
        lqmeg       = dfEmeg[i].quantile(0.25).values.squeeze()

        plt.fill_between(x,lq,uq, color=colors[i], alpha=0.5)
        plt.plot(x, mean, color=colors[i], label='top-down')
        plt.fill_between(x, lqmeg,uqmeg, color=colors[i], alpha=0.5, facecolor=colors[i], hatch='X', linewidth=0)
        plt.plot(x, meanmeg, color=colors[i], linestyle='--', label='MEGAN')
        plt.ylabel(labels[i], color=colors[i], fontsize=24)
        if i==0:
            plt.legend(loc='best')
        if i%2 == 1:
            axes[i].yaxis.set_label_position("right")
            axes[i].yaxis.tick_right()
    plt.ylim([0, 1e13])
    plt.xlim([-0.5,11.5])
    plt.xticks(x)
    plt.gca().set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    plt.xlabel('month')
    plt.suptitle('E$_{GC}$ vs E$_{OMI}$; mean and IQR',fontsize=30)
    f.subplots_adjust(hspace=0)


    ## save figure
    plt.savefig(pname)
    print("Saved %s"%pname)
    plt.close()

def distributions_comparison_regional(d0=datetime(2005,1,1),dE=datetime(2012,12,31)):
    '''
    '''

    ## Read Emegan and Enew into dataframe for a region and season
    Enew=E_new(d0,dE, dkeys=['E_MEGAN','E_PP_lr'])
    lats,lons=Enew.lats_lr,Enew.lons_lr
    Em=Enew.E_MEGAN
    Em[Em<1] = np.NaN
    Eo=Enew.E_PP_lr
    Eo[Eo<1] = np.NaN
    dates=Enew.dates
    # summer date indices
    summer= [ d.month in [1,2,12] for d in dates ]
    # winter date indices
    winter= [ d.month in [6,7,8] for d in dates ]

    # also work on monthly datasets
    monthlym = util.monthly_averaged(dates, Em.copy(), keep_spatial=True)
    Em_m = monthlym['mean']
    dates_m = monthlym['dates']

    monthlyo = util.monthly_averaged(dates, Eo.copy(), keep_spatial=True)
    Eo_m = monthlyo['mean']

    summer_m = [d.month in [1,2,12] for d in dates_m]
    winter_m = [d.month in [6,7,8] for d in dates_m]

    pname1=[]
    pname2=[]
    for region,color,label in zip(pp.__subregions__, pp.__subregions_colors__, pp.__subregions_labels__):

        # pull out region:
        lati,loni = util.lat_lon_range(lats,lons,region)
        Emsub=Em[:,lati,:]
        Emsub=Emsub[:,:,loni]
        Eosub=Eo[:,lati,:]
        Eosub=Eosub[:,:,loni]
        Emsub=Emsub[summer]
        Eosub=Eosub[summer]
        # for monthly also
        Emsub_m=Em_m[:,lati,:]
        Emsub_m=Emsub_m[:,:,loni]
        Eosub_m=Eo_m[:,lati,:]
        Eosub_m=Eosub_m[:,:,loni]
        Emsub_m=Emsub_m[summer_m]
        Eosub_m=Eosub_m[summer_m]

        # set 95th percentile as axes limits
        xmax=np.nanpercentile(Eosub,99)
        ymax=np.nanpercentile(Emsub,99)

        # lets put summer data into a dataframe for easy plotting
        subdata=np.array([Emsub.flatten(), Eosub.flatten()]).T
        subdata_m=np.array([Emsub_m.flatten(), Eosub_m.flatten()]).T
        slope,intercept,reg,ci1,ci2 = RMA(subdata_m[:,1],subdata_m[:,0])
        legend = "y={0:.1f}x+{1:.1e}: r={2:.2f}".format(slope,intercept,reg)

        df = pd.DataFrame(data=subdata, columns=['MEGAN','OMI'])
        df_m = pd.DataFrame(data=subdata_m, columns=['MEGAN','OMI'])

        plt.figure(figsize=[15,15])
        with sns.axes_style('white'):
            g = sns.jointplot("OMI", "MEGAN", df, kind='hex',#kind='reg')
                              dropna=True, xlim=[0,xmax], ylim=[0,ymax],
                              color=color,)
            # halve the x axis limit
            #g.ax_marg_x.set_xlim(0,g.ax_marg_x.get_xlim()[1]/2.0)
        plt.suptitle(label,fontsize=20)
        pname1.append('%s_summer_daily.png'%label)
        plt.savefig(pname1[-1])
        print('SAVED ',pname1[-1])
        plt.close()

        plt.figure(figsize=[15,15])
        with sns.axes_style('white'):
            g = sns.jointplot("OMI", "MEGAN", df_m, kind='reg',
                              dropna=True, #xlim=[0,xmax], ylim=[0,ymax],
                              color=color,
                              label=legend,
                              )
            g.ax_joint.legend(handlelength=0, handletextpad=0, frameon=False,)


            # halve the x axis limit
            #g.ax_marg_x.set_xlim(0,g.ax_marg_x.get_xlim()[1]/2.0)
        plt.suptitle(label,fontsize=20)
        pname2.append('Figs/Emiss/%s_summer_monthly.png'%label)
        plt.savefig(pname2[-1])
        print('SAVED ',pname2[-1])
        plt.close()


    for combined, pnames in zip(['Figs/Emiss/daily_Egressions.png','Figs/Emiss/monthly_Egressions.png'],[pname1, pname2]):
        images = [Image.open(pname) for pname in pnames]
        width, height = images[0].size

        # 3 rows 2 cols
        total_width = width * 2 + 50
        total_height= height * 3 + 50
        #max_height = max(heights)

        new_im = Image.new('RGBA', (total_width, total_height))

        for i,im in enumerate(images):
            x_offset = width*(i%2)
            y_offset = height*(i//2)
            new_im.paste(im, [x_offset, y_offset])
        new_im.save(combined)
        print('SAVED ',combined)

def map_E_new(month=datetime(2005,1,1), GC=None, OMI=None,
              smoothed=False, linear=True,
              clims=[2e11,2e12], region=pp.__AUSREGION__,
              cmapname='PuBuGn'):
    '''
        Plot calculated emissions
    '''
    day0=month
    dstr=month.strftime("%Y %b") # YYYY Mon
    dayn=util.last_day(day0)
    em=E_new(day0, dayn)
    #hemco=GC_class.Hemco_diag(day0,dayn)
    #_mdates, megan = hemco.daily_LT_averaged(hour=13) # 1pm local time data
    megan=em.E_MEGAN
    megan=np.nanmean(megan,axis=0) # avg over time
    meglats=em.lats_lr
    meglons=em.lons_lr


    titles=['OMI','OMI_u','OMI_lr','GC','GC_u','GC_lr','PP','PP_u','PP_lr']
    plt.figure(figsize=(12,24))
    plt.subplot(4,1,1)
    pp.createmap(megan,meglats,meglons,title='MEGAN',
                 vmin=clims[0], vmax=clims[1]*2, linear=linear, aus=True,
                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$', cmapname=cmapname)

    for i,arr in enumerate([em.E_OMI, em.E_OMI_u , em.E_OMI_lr,
                            em.E_GC, em.E_GC_u, em.E_GC_lr,
                            em.E_PP, em.E_PP_u, em.E_PP_lr]):
        plt.subplot(4,3,i+4)
        lats=[em.lats, em.lats_lr][(i%3) == 2]
        lons=[em.lons, em.lons_lr][(i%3) == 2]
        arr = np.nanmean(arr,axis=0) # average over time
        bmap,cs,cb= pp.createmap(arr, lats, lons, title=titles[i], GC_shift=True,
                                 smoothed=smoothed,
                                 vmin=clims[0], vmax=clims[1], linear=linear, aus=True,
                                 clabel=r'Atoms C cm$^{-2}$ s$^{-1}$', cmapname=cmapname)

        # Add dot to map
        sydlat,sydlon=pp.__cities__['Syd']
        mx,my = bmap(sydlon, sydlat)
        bmap.plot(mx, my, 'o', markersize=6, color='r')
    pname='Figs/Emiss/E_new_maps_%s.png'%day0.strftime("%Y%m")
    plt.suptitle('Emissions %s'%dstr)
    plt.savefig(pname)

    print('SAVED ', pname)


def MEGAN_vs_E_new(d0=datetime(2005,1,1), d1=datetime(2007,12,31),
                  key='E_PP_lr',
                  region=pp.__AUSREGION__):
    '''
        Plot E_gc, E_new, diff, ratio
        Use biogenic GC output
    '''
    dstr=d0.strftime("%Y%m")
    yyyymon=d0.strftime("%Y, %b")
    if __VERBOSE__:
        print("running Analyse_E_isop.E_gc_VS_E_new from %s to %s"%(d0, d1))

    ## READ DATA
    #    GC=GC_class.Hemco_diag(day0=d0,dayn=d1,month=False)
    #    gcdays, Megan_isop=GC.daily_LT_averaged(hour=13) # atomC/cm2/s
    #    # subset to region
    #    lati,loni=util.lat_lon_range(GC.lats,GC.lons,region=region)
    #    latsgc=GC.lats[lati]
    #    lonsgc=GC.lons[loni]
    #    Megan_isop=Megan_isop[:,lati,:]
    #    Megan_isop=Megan_isop[:,:,loni]
    #    Megan_isop=np.mean(Megan_isop,axis=0) # average over time
    #    Megan_isop_compare=Megan_isop

    # based on OMI using GC calculated yield (slope)
    d0s=d0.strftime("%Y%m%d")
    d1s=d1.strftime("%Y%m%d")
    Enew=E_new(day0=d0, dayn=d1,dkeys=[key,'E_MEGAN'])
    New_isop=getattr(Enew,key) # atom c/cm2/s
    New_isop=np.nanmean(New_isop,axis=0) # average over time
    lats,lons = Enew.lats, Enew.lons
    lowres=False
    if '_lr' in key:
        lats=Enew.lats_lr
        lons=Enew.lons_lr
        lowres=True
    MEGAN = np.nanmean(Enew.E_MEGAN,axis=0)
    meglats,meglons=Enew.lats_lr,Enew.lons_lr


    # Compare megan and E_new
    pp.compare_maps([MEGAN, New_isop],[lats,meglats],[lons,meglons], pname='Figs/Emiss/MEGAN_vs_%s_%s-%s.png'%(key,d0s,d1s),
                    titles=['$E_{GC}$','$E_{NEW}$'], rmin=-500, rmax=500, suptitle='Emissions Averaged from %s to %s'%(d0s,d1s),
                    clabel=Enew.attributes['E_MEGAN']['units'],
                    lower_resolution=lowres, linear=True,
                    maskocean=True, normalise=False)

    # Compare megan and E_new after normalising
    pp.compare_maps([MEGAN, New_isop],[lats,meglats],[lons,meglons], pname='Figs/Emiss/MEGAN_vs_%s_%s-%s_normalised.png'%(key,d0s,d1s),
                    titles=['$E_{GC}$','$E_{NEW}$'], suptitle='Emissions Averaged from %s to %s, normalised by means'%(d0s,d1s),
                    clabel=Enew.attributes['E_MEGAN']['units'],
                    lower_resolution=lowres, linear=True,
                    maskocean=True, normalise=True)


    #    compare_maps(datas,lats,lons,pname=None,titles=['A','B'], suptitle=None,
    #                 clabel=None, region=__AUSREGION__, vmin=None, vmax=None,
    #                 rmin=-200.0, rmax=200., amin=None, amax=None,
    #                 axeslist=[None,None,None,None],
    #                 maskocean=False,
    #                 lower_resolution=False, normalise=False,
    #                 linear=False, alinear=True, rlinear=True):
    #    ## First plot maps of emissions:
    #    ##
    #    plt.figure(figsize=(16,12))
    #    vlinear=False # linear flag for plot functions
    #    clims=[1e12,2e14] # map min and max
    #    amin,amax=-1e12, 3.5e12 # absolute diff min and max
    #    rmin,rmax=0, 10 # ratio min and max
    #    cmapname='PuRd'
    #
    #    # start with E_GC:
    #    plt.subplot(221)
    #    pp.createmap(Megan_isop,latsgc,lonsgc,vmin=clims[0],vmax=clims[1],GC_shift=True,
    #                 linear=vlinear,region=region,smoothed=smoothed,
    #                 cmapname=cmapname)
    #
    #    # then E_new
    #    plt.subplot(222)
    #    pp.createmap(New_isop,omilats,omilons,vmin=clims[0],vmax=clims[1],
    #                 linear=vlinear,region=region,smoothed=smoothed,
    #                 cmapname=cmapname)
    #
    #    ## Difference and ratio:
    #    ##
    #    cmapname='jet'
    #    ## Diff map:
    #    plt.subplot(223)
    #    title=r'E$_{MEGAN} - $E$_{new}$ '
    #    args={'region':region, 'clabel':r'atoms C cm$^{-2}$ s$^{-1}$',
    #          'linear':True, 'lats':lats, 'lons':lons,
    #          'smoothed':smoothed, 'title':title, 'cmapname':cmapname,
    #          'vmin':amin, 'vmax':amax}
    #    pp.createmap(Megan_isop_compare - New_isop_compare, **args)
    #
    #    ## Ratio map:
    #    plt.subplot(224)
    #    args['title']=r"$E_{MEGAN} / E_{OMI}$"
    #    args['vmin']=rmin; args['vmax']=rmax
    #    args['clabel']="ratio"
    #    pp.createmap(Megan_isop_compare / New_isop_compare, **args)
    #
    #
    #    # SAVE THE FIGURE:
    #    #
    #    suptitle='GEOS-Chem (gc) vs OMI for %s'%yyyymon
    #    plt.suptitle(suptitle)
    #    fname='Figs/GC/E_Comparison_%s%s%s.png'%(dstr,
    #                                             ['','_smoothed'][smoothed],
    #                                             ['','_lowres'][lowres])
    #    plt.savefig(fname)
    #    print("SAVED FIGURE: %s"%fname)
    #    plt.close()

    ## PRINT EXTRA INFO
    #
    #if __VERBOSE__:
        #print("GC calculations:")
        #for k,v in :
        #    print ("    %s, %s, mean=%.3e"%(k, str(v.shape), np.nanmean(v)))
        #print("OMI calculations:")
        #for k,v in Enew.items():
        #    print ("    %s, %s, mean=%.3e"%(k, str(v.shape), np.nanmean(v)))

    #    # Print some diagnostics here.
    #    for l,e in zip(['Enew','E_gc'],[New_isop_compare,Megan_isop_compare]):
    #        print("%s: %s    (%s)"%(l,str(e.shape),dstr))
    #        print("    Has %d nans"%np.sum(np.isnan(e)))
    #        print("    Has %d negatives"%np.sum(e<0))


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
        GC=GC_class.GC_tavg(day0, dayn)
    if OMI is None:
        OMI=omhchorp(day0,dayn,ignorePP=ignorePP)

    ## Plot E_new
    ##
    clims=[2e11,2e12]
    cmapname='YlGnBu'
    for smoothed in [True,False]:
        pname='Figs/Emiss/E_new_%s%s.png'%(dstr,['','_smoothed'][smoothed])
        map_E_new(month=month, GC=GC, OMI=OMI, clims=clims, cmapname=cmapname,
                  smoothed=smoothed,pname=pname)

    ## Plot E_GC vs E_new, low and high res.
    ##
    for smoothed in [True, False]:
        E_gc_VS_E_new(d0=day0,d1=dayn, smoothed=smoothed)

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
        GC=GC_class.GC_tavg(date=month)
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
    regions=[SEA,SWA,NA]
    rnames=['SEA','SWA','NA']
    # summer
    E_summer=E_new(ds0,ds1)
    E_winter=E_new(dw0,dw1)
    for region,rname in zip(regions,rnames):
        plt.figure(figsize=(14,14))#,sharex=True,squeeze=True)
        for ds in [True,False]:
            ppargs={'colour':'red','linecolour':'red','diag':False,'legend':False}
            E_summer.plot_regression(ds0,ds1,region=region,deseasonalise=ds,**ppargs)
            ppargs['colour']='aqua'
            ppargs['linecolour']='aqua'
            E_winter.plot_regression(dw0,dw1,region=region,deseasonalise=ds,**ppargs)
            plt.title('%s Summer (JF) vs Winter (JJA), 2005'%rname)
            plt.legend(loc='best')
            plt.xlabel('MEGAN')
            plt.ylabel('Satellite based')
            pname='Figs/Regression_%s_2005%s.png'%(rname,['','_DS'][ds])
            plt.savefig(pname)
            plt.close()
            print("Saved: ",pname)

def Compare_to_daily_cycle(month=datetime(2005,1,1),lat=-33,lon=151):
    '''
        Compare E_new at a lat/lon to the MEGAN daily cycle at that lat lon
    '''

    #Read E_new:
    Enew=E_new(day0=month,dayn=util.last_day(month))
    lati,loni=util.lat_lon_index(lat,lon,Enew.lats,Enew.lons)
    E_isop=Enew.E_isop[:,lati,loni]
    Enewdates=[d.replace(hour=13) for d in Enew.dates]


    # Read GEOS-Chem:
    GC=GC_class.Hemco_diag(day0=month,dayn=None,month=True)
    lati,loni=GC.lat_lon_index(lat,lon)
    GC_E_isop=GC.E_isop_bio[:,lati,loni]
    gcoffset=GC.local_time_offset[lati,loni]
    gcdates=[]
    gcmiddays=[]
    GC_E_isop_mids=[]
    for i,date in enumerate(GC.dates):
        gcdates.append(date+timedelta(seconds=int(3600*gcoffset)))
        if date.hour+gcoffset==13:
            gcmiddays.append(date)
            GC_E_isop_mids.append(GC_E_isop[i])

    # figure, first do whole timeline:
    f, axes = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 4]})
    a0, a1= axes[0],axes[1]
    plt.sca(a0)
    pp.plot_time_series(gcdates,GC_E_isop, dfmt='%d', color='r')
    pp.plot_time_series([gcdates[0],gcdates[-1]],np.repeat(np.nanmean(E_isop),2),color='k',linewidth=1)
    #pp.plot_time_series(gcmiddays,GC_E_isop_mids, color='r',linestyle='none',marker='x')
    #print(gcmiddays)
    #print(Enew.dates)
    #print(Enewdates)
    #pp.plot_time_series(Enewdates,E_isop, color='k', marker='x',linewidth=2)

    #plt.plot(np.arange(gcoffset,len(GC_E_isop)+gcoffset), GC_E_isop, color='r')

    for ax in [a0,]:
        plt.sca(ax)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off

    # then show daily cycle
    plt.sca(a1)

    pp.plot_daily_cycle(GC.dates,GC_E_isop,houroffset=gcoffset, color='r')
    #for i in range(len(E_isop)):
    #    plt.plot([10,14],[E_isop[i],E_isop[i]], color='k')
    plt.plot([9,16],np.repeat(np.nanmean(E_isop),2),'k',linewidth=2)

    pname='Figs/E_new_vs_Daily_MEGAN.png'
    plt.savefig(pname)
    print('Saved ',pname)
    plt.close()

    pp.plot_time_series(Enewdates,E_isop, color='k', marker='x',linewidth=2)
    pp.plot_time_series(Enewdates,Enew.GC_E_isop[:,lati,loni],color='r')
    plt.savefig('test.png')

def show_subregions():
    '''
        Show subregions over which several analyses are perform
    '''
    colors=pp.__subregions_colors__
    colors[0]='grey'
    pp.displaymap(region=np.array(pp.__AUSREGION__) + np.array([-5,-10,5,10]),
                  subregions=pp.__subregions__,labels=pp.__subregions_labels__,
                  colors=colors, linewidths=[3,2,2,2,2,2],
                  fontsize='large'  )
    plt.savefig('Figs/subregions.png')

def tga_summary(day0=datetime(2005,1,1), daye=datetime(2007,12,31)):
    '''
        calculate, print and plot emissions in Tg/annum from Enew and Egc
    '''
    years = util.list_years(day0,daye)
    omi=[]
    gc =[]
    gca=[]
    for year in years:
        d0=year
        de=datetime(d0.year,12,31)

        ## read MEGAN, get tg/a emissions
        #
        hemco=GC_class.Hemco_diag(d0,de)
        hemco._set_E_isop_bio_kgs()
        E_GC = hemco.E_isop_bio_kgs*3600 # kg/s -> kg/hr
        # sum over time to get Tg/year  (Tg/kg = 1e-9)
        E_GC_total = np.sum(E_GC, axis=0) * 1e-9
        subset = util.lat_lon_subset(hemco.lats,hemco.lons, region= pp.__AUSREGION__, data=[E_GC_total])

        E_GC_aus = subset['data'][0]

        ## Read E_OMI and get tg/a emissions
        #
        Enew=E_new(d0,de)
        lats=Enew.lats_lr
        lons=Enew.lons_lr
        dates=Enew.dates
        assert np.all(lats == subset['lats']), 'lats are wrong!?'

        # daylengths are from 14hrs (mid-summer) to 10hrs (mid-winter)
        # daylengths extend change by 20, 40, 60, 60, 40, 20 minutes between peaks
        monthly_daylength_changes = np.array([20, -20, -40, -60, -60, -40, -20, 20, 40, 60, 60, 40])
        monthly_daylengths = 14*60-20 + np.cumsum(monthly_daylength_changes) # in minutes

        daily_daylengths = np.array([monthly_daylengths[d.month-1] for d in dates])/60.0 # in hours
        # repeat over lat,lon axes
        daily_daylengths = np.repeat(daily_daylengths[:,np.newaxis], len(lats), axis=1)
        daily_daylengths = np.repeat(daily_daylengths[:,:,np.newaxis], len(lons), axis=2)
        midday_kg = Enew.E_PP_lr * Enew.conversion_to_kg_lr * 3600 # in kg/hours
        midday_kg[midday_kg<0] = 0
        daily_kg = midday_kg * 0.637 *daily_daylengths
        E_OMI_total = np.nansum(daily_kg,axis=0) * 1e-9 # kg/day -> total Tg

        print ('year: ',year.year)
        print ('    global MEGAN : ' , np.sum(E_GC_total))
        print ('    aus MEGAN    : ' , np.sum(E_GC_aus))
        print ('    aus OMI      : ' , np.nansum(E_OMI_total))
        # store yearly total (keep spatial dims)
        omi.append(E_OMI_total)
        gca.append(E_GC_aus)
        gc.append(E_GC_total)

    omi = np.array(omi) # should be [year, lats, lons]
    gc  = np.array(gc)
    gca = np.array(gca)

    pname='Figs/Emiss/tga_map.png'
    gca_flat = np.nanmean(gca,axis=0)
    omi_flat = np.nanmean(omi,axis=0)
    gca_flat[gca_flat<1e-6] = np.NaN
    omi_flat[omi_flat<1e-6] = np.NaN # set zeros to NaN for image
    a,b = pp.compare_maps([gca_flat,omi_flat],[lats,lats],[lons,lons], rmin=-400,rmax=400,
                          titles=['E$_{GC}$','E$_{OMI}$'], linear=True, pname=pname)


    # total over spatial dims
    omi= np.nansum(omi,axis=(1,2))
    gc = np.nansum(gc,axis=(1,2))
    gca= np.nansum(gca,axis=(1,2))
    print('TOTALS: ',day0.year, ' - ', daye.year)
    print(' mean(std) global MEGAN:  %.2f (%.2f)'%(np.mean(gc), np.std(gc)))
    print(' mean(std) aus MEGAN   :  %.2f (%.2f)'%(np.mean(gca), np.std(gca)))
    print(' mean(std) aus OMI     :  %.2f (%.2f)'%(np.mean(omi), np.std(omi)))

if __name__=='__main__':


    d0=datetime(2005,1,1)
    d1=datetime(2005,1,31)
    dn=datetime(2005,12,31)
    de=datetime(2007,12,31)

    ## Tga summary
    #
    #tga_summary()


    ## Plot megan daily cycle vs top-down emissions in several zones
    #yearly_megan_cycle(d0,de)
    # yearly cycle vs campaigns
    yearly_cycle_vs_campaigns()

    ## Plot MEGAN vs E_new (key)
    ## compare megan to a top down estimate, both spatially and temporally
    ## Ran 17/7/18 for Jenny jan06 check
    #MEGAN_vs_E_new(d0,dn)

    ## Plot showing comparison of different top-down estimates
    ## In Isop chapter results
    ## Ran 17/7/18 for jan06 check for Jenny
    #map_E_new(datetime(2006,1,1))

    ## Plot showing time series within Australia, and Sydney area
    ## In isop chapter results
    ## Ran 7/11/18
    with np.warnings.catch_warnings():
        E_regional_time_series()
        #np.warnings.filterwarnings('ignore')
        #for etype in ['gc','omi','pp']:
            #E_regional_multiyear(d0=d0,dn=de, etype=etype)
            #for monthly in [True, False]:
            #    E_regional_time_series(d0,de,etype=etype, force_monthly=monthly,
            #                           force_monthly_func='median')

        ## Time series at a particular location
        ## Takes a few minuts (use qsub), In isop chapter results
        ## Ran 25/7/18 - updating for pixel counts and errorbars
        #E_time_series(d0,de,#lon=pp.__cities__['Wol'][1]-2.5, locname='Inland',count=False)
        #              lat=pp.__cities__['Wol'][0],lon=pp.__cities__['Wol'][1],
        #              locname='Wollongong', ylims=[-0.2e13,2.5e13],
        #              monthly=True, monthly_func='median')


    #All_maps()
    #megan_SEA_regression()
    #Compare_to_daily_cycle()
    #dn=datetime(2005,12,31)
    #map_E_new()
    #check_E_new()

    #map_E_gc(month=d0,GC=GC_tavg(d0))


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

