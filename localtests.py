#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()


# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities.JesseRegression import RMA, OLS
from utilities import GMAO,GC_fio,fio, masks
import Inversion
import tests
from tests import utilities_tests, test_new_emissions, test_filters
import reprocess
import new_emissions
import Analyse_E_isop

from classes.E_new import E_new # E_new class
from classes import GC_class, campaign
from classes.omhchorp import omhchorp

from utilities import masks

import xbpch
import xarray
import pandas as pd
import seaborn as sns

import warnings
import timeit


# parallelism
import concurrent.futures as confut


###############
### Globals ###
###############
__VERBOSE__=True
region=pp.__AUSREGION__

#####
## SETUP STUFFS
#####


# start timer
start1=timeit.default_timer()

##########
### DO STUFFS
##########

import chapter_3_isop
df=chapter_3_isop.read_overpass_timeseries()

#test_filters.summary_pixels_filtered()
d0,dN = datetime(2005,1,1),datetime(2005,6,30)
dkeys=['E_PP_lr', 'pixels_PP_lr', # emissiosn and pixel counts
       'E_PP_err_lr','E_PPm_err_lr','SC_err_lr',  # Error in emissions estimate
       'VCC_err_lr','VCC_rerr_lr', # daily OmegaPP error in
       'slope_rerr_lr'] # monthly portional error in slope

# READ EMISSIONS AND ERROR
enew=E_new(d0,dN,dkeys=dkeys)
dates=enew.dates
months=util.list_months(d0,dN)
lats,lons=enew.lats_lr, enew.lons_lr
pix=enew.pixels_PP_lr
# GET MONTHLY TOTAL PIXELS
pixm=util.monthly_averaged(dates,pix,keep_spatial=True)['sum']

for key in dkeys:
    print(key, getattr(enew,key).shape)

# MASK OCEANS, 
E = enew.E_PP_lr
E[enew.oceanmask3d_lr] = np.NaN
Em      = util.monthly_averaged(dates,E,keep_spatial=True)['mean']
Eerr   = enew.E_PP_err_lr
Eerrm  = enew.E_PPm_err_lr
Ererr  = Eerr/E
Ererr[~np.isfinite(Ererr)] = np.NaN
Ererr[np.isclose(E,0.0)] = np.NaN
Ererrm = Eerrm/Em
Ererrm[~np.isfinite(Ererrm)] = np.NaN
Ererrm[np.isclose(Em,0.0)] = np.NaN

Srerr  = enew.slope_rerr_lr
Orerr  = enew.VCC_rerr_lr

# first lets do A seasonal plot of relative monthly error
plt.close()
chapter_3_isop.PlotMultiyear(Ererrm,months,lats,lons,weekly=False,
                             median=True,ylims=[-1,1])
plt.savefig('mya_rerr.png')
print("SAVED ",'mya_rerr.png')



mya = util.multi_year_average_spatial(E, dates)
myam = util.multi_year_average_spatial(Ererrm, months)


summer=np.array([0,1,11])
winter=np.array([5,6,7])
summersm = np.nanmean(myam['mean'][summer,:,:],axis=0)
wintersm = np.nanmean(myam['mean'][winter,:,:],axis=0)
Esummer = np.nanmean(mya['mean'][summer,:,:],axis=0)
Ewinter = np.nanmean(mya['mean'][winter,:,:],axis=0)


plt.close()

# Plot summer,winter maps of monthly rerr
plt.figure(figsize=[14,14])
vmin,vmax=1e11,1e13
linear=False

plt.subplot(2,2,1)
pp.createmap(Esummer,lats,lons,aus=True,
             linear=linear, vmin=vmin,vmax=vmax,
             clabel='C/cm2/s',
             title='mean a posteriori')
plt.ylabel("Summer")
plt.subplot(2,2,2)
pp.createmap(summersm,lats,lons,aus=True,
             linear=True, vmin=0,vmax=1,
             clabel='Portional',
             title='relative a posteriori error')
plt.ylabel("Summer")

plt.subplot(2,2,3)
pp.createmap(Ewinter, lats,lons, aus=True,
             linear=linear, vmin=vmin,vmax=vmax,
             clabel='C/cm2/s',
             title='mean a posteriori')
plt.ylabel("Winter")
plt.subplot(2,2,4)
pp.createmap(wintersm,lats,lons,aus=True,
             linear=True, vmin=0,vmax=1, 
             clabel='Portional',
             title='relative a posteriori error',
             pname='test_uncert.png')
plt.ylabel("Winter")




###########
### Record and time STUJFFS
###########

end=timeit.default_timer()
print("TIME: %6.2f minutes for stuff"%((end-start1)/60.0))


############
### To send into other scripts
##########


'''
# Read campaign data
mumba = campaign.mumba()
sps1  = campaign.sps(1)
sps2  = campaign.sps(2)
lat,lon = sps1.lat,sps1.lon

colors=['m','pink','orange']

# Read in emissions from E_new
enew_keys=['E_PP_lr']
#allkeys=['E_OMI', 'E_GC', 'E_PP','E_OMI_lr','E_GC_lr','E_PP_lr',
#         'E_MEGAN', 'pixels','pixels_PP','pixels_lr','pixels_PP_lr']
Enew  = E_new(d0,d1,enew_keys)
lats, lons = Enew.lats_lr, Enew.lons_lr
yi,xi = util.lat_lon_index(lat,lon,lats,lons)
Eomi = Enew.E_PP_lr[:,yi,xi]
Eomi[Eomi<0]=0
dates=Enew.dates

E_meg = GC_class.Hemco_diag(d0,d1)
meglats,meglons= E_meg.lats, E_meg.lons
megyi,megxi = util.lat_lon_index(lat,lon,meglats,meglons)
Emeg_o = E_meg.local_time_offset[megyi,megxi]
Emeg = E_meg.E_isop_bio[:,megyi,megxi]
megdates=E_meg.dates

plt.close()
fig = plt.figure(figsize=[16,8])
pp.plot_time_series(dates, Eomi, ylabel='molec cm$^{-2}$ s$^{-1}$',
                    linestyle='--', marker='*', color='b', label='E$_{OMI}$')
pp.plot_time_series(megdates, Emeg,
                    linestyle='-', color='k',label='E$_{MEGAN}$')
ax=plt.gca()
# new x and y axes (top and right)
newax = plt.twinx().twiny()
plt.sca(newax)
pp.plot_time_series(mumba.dates, mumba.isop, ylabel='ppb',
                    color=colors[0], label='mumba')

# combine legends from axes and put somewhere
handles, labels = ax.get_legend_handles_labels()
newhandles, newlabels= newax.get_legend_handles_labels()

#fig.legend(handles+newhandles, labels+newlabels, loc="best", frameon=True)
plt.legend(handles+newhandles, labels+newlabels,loc='best', fontsize=10,frameon=False)

#pp.InitMatplotlib() # set up plotting defaults
# Time series plots, how displaying each line
#linewidths  = [2,1,1,1]
#colours     = ['k','m','orange','red']

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
'''
#plt.figure(figsize=(14,16))
## create map, with gridsquare of comparison, and dots for campaigns
#plt.subplot(2,1,1)
#m=pp.displaymap(region=[-45,130,-14,155])
#pp.add_grid_to_map(m,)
## Add dot to map
#for i,[y,x] in enumerate([[mumba.lat,mumba.lon],[sps1.lat,sps1.lon],[sps2.lat,sps2.lon]]):
#    mx,my = m(x, y)
#    m.plot(mx, my, 'o', markersize=3, color=colors[i])
#
#plt.subplot(2,1,2)
#dates,isop=mumba.get_daily_hour(key='isop')
#d1,i1=sps1.get_daily_hour(key='isop')
#d2,i2=sps2.get_daily_hour(key='isop')
#Enew.get_series('isop',)
#
#pp.plot_yearly_cycle(isop,dates,color='m',label='MUMBA',linewidth=2)
#pp.plot_yearly_cycle(i1,d1,color='pink',label='SPS1',linewidth=2)
#pp.plot_yearly_cycle(i2,d2,color='orange',label='SPS2',linewidth=2)
#
#plt.legend()
#plt.title('isoprene yearly cycle')
#plt.tight_layout()


### Things to be finished
###

# Profile plots from ozone paper:
def monthly_profiles(hour=0, degradesondes=False):
    '''
    Profile mean and std deviation for each month for each site
    If you only want to consider a particular hour then set the hour parameter
        hour can be one of [0, 6, 12, 18, None]
    set degradesondes=True to degrade the sondes to matching GEOS-Chem resolution before finding the average..
    '''

    # read site data
    sites = [ fio.read_GC_station(p) for p in range(3) ]
    o3sondes = [ fio.read_sonde(p) for p in range(3) ]

    # some plot setups stuff
    titles=np.array([['Dec','Jan','Feb'],['Mar','Apr','May'],['Jun','Jul','Aug'],['Sep','Oct','Nov']])
    months=np.array([[11,0,1],[2,3,4],[5,6,7],[8,9,10]])
    seasonalcolours=seasonal_cmap.colors


    #for each station do this
    # site,sonde=sites[1],o3sondes[1]
    for site, sonde in zip(sites, o3sondes):

        # degrade sonde if we are doing that
        if degradesondes:
            sonde.degrade_vertically(site)

        # Set up plot axes
        f, axes = plt.subplots(4,3, sharex=True, sharey=True, figsize=(16,16))
        axes[3,1].set_xlabel('Ozone (ppb)')
        xlim=[0,125]
        axes[3,1].set_xlim(xlim)
        axes[1,0].set_ylabel('Altitude (km)')
        ylim=[0,14]
        axes[1,0].set_ylim(ylim)

        # Grab Ozone
        O3 = site['O3']
        s_O3 = np.array(sonde.o3ppbv)

        # metres to kilometres
        s_TP = np.array(sonde.tp) # sonde TP is already in km
        TP = site['TropopauseAltitude'] / 1000.0
        Z  = site['Altitudes']/1000.0
        s_Z  = np.array(sonde.gph) / 1000.0
        Znew= np.linspace(0,14,100)
        # need to vertically bin the O3 profiles,
        # interpolated at 100 points up to 14km
        means=np.zeros([12,100])
        medians=np.zeros([12,100])
        pct5 = np.zeros([12,100]) # 5th percentile
        pct95 = np.zeros([12,100]) # 95th percentile
        stds =np.zeros([12,100])
        TPm = np.zeros(12)
        s_means=np.zeros([12,100])
        s_medians=np.zeros([12,100])
        s_pct5 = np.zeros([12,100]) # 5th percentile
        s_pct95 = np.zeros([12,100]) # 95th percentile
        s_stds =np.zeros([12,100])
        s_TPm = np.zeros(12)
        s_counts=np.zeros(12)

        # bin data into 12 months
        allmonths=np.array([ d.month for d in site['Date'] ])
        s_allmonths=np.array( [ d.month for d in sonde.dates ])
        for month in range(12):
            # find indices matching the current month
            inds=np.where(allmonths == month+1)[0]
            if hour is not None:
                allhours =np.array([ d.hour for d in site['Date'] ])
                inds = np.where( (allmonths == month+1) * (allhours==hour) )[0]
            s_inds=np.where(s_allmonths == month+1)[0]
            n, s_n = len(inds), len(s_inds)
            s_counts[month]=s_n

            # each profile needs to be interpolated up to 14km
            profs=np.zeros([n,100])
            s_profs=np.zeros([s_n,100])
            for i in range(n):
                profs[i,:] = np.interp(Znew, Z[inds[i],:], O3[inds[i],:],left=np.NaN,right=np.NaN)
            for i in range(s_n):
                s_profs[i,:] = np.interp(Znew, s_Z[s_inds[i],:], s_O3[s_inds[i],:],left=np.NaN,right=np.NaN)
            means[month,:]=np.nanmean(profs,axis=0)
            medians[month,:]=np.nanmedian(profs,axis=0)
            stds[month,:] =np.nanstd(profs,axis=0)
            pct5[month,:] = np.nanpercentile(profs,5,axis=0)
            pct95[month,:] = np.nanpercentile(profs,95,axis=0)
            TPm[month] = np.nanmean(TP[inds])
            s_means[month,:]=np.nanmean(s_profs,axis=0)
            s_medians[month,:]=np.nanmedian(s_profs,axis=0)
            s_stds[month,:] =np.nanstd(s_profs,axis=0)
            s_pct5[month,:] = np.nanpercentile(s_profs,5,axis=0)
            s_pct95[month,:] = np.nanpercentile(s_profs,95,axis=0)
            s_TPm[month] = np.nanmean(s_TP[s_inds])

        # plot the median profiles and shade the area of 5th-95th percentiles
        for i in range(4):
            for j in range(3):
                plt.sca(axes[i,j]) # set current axis
                mind=months[i,j]
                X=medians[mind,:]
                Xl=X-pct5[mind,:]
                Xr=X+pct95[mind,:]
                s_X=s_medians[mind,:]
                s_Xl=s_X-s_pct5[mind,:]
                s_Xr=s_X+s_pct95[mind,:]

                # plot averaged profiles + std
                plt.plot(X, Znew , linewidth=3, color=col['GEOS'])
                plt.fill_betweenx(Znew, Xl, Xr, alpha=0.5, color=seasonalcolours[i])
                plt.plot(s_X, Znew , linewidth=3, color=col['Sonde'])
                plt.fill_betweenx(Znew, s_Xl, s_Xr, alpha=0.5, color=seasonalcolours[i])
                # plot tropopause
                Y=TPm[mind]
                s_Y=s_TPm[mind]
                plt.plot(xlim, [Y, Y], '--', linewidth=2, color=col['GEOS'])
                plt.plot(xlim, [s_Y, s_Y], '--', linewidth=2, color=col['Sonde'])
                # plot title and text
                plt.title(titles[i,j])
                # add count text to upper corner
                plt.text(.72*xlim[1], .5, "%d sondes"%s_counts[mind])

        # set title, and layout, then save figure
        stn_name=site['Station'].split(' ')[0]
        if hour is not None: stn_name+='_H%02d'%hour
        f.suptitle("Monthly Median Profiles over "+stn_name)
        outfile='images/eventprofiles/%s_monthprofiles.png'%stn_name
        if degradesondes:
            outfile='images/eventprofiles/%s_monthprofilesdegraded.png'%stn_name
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(outfile)
        print("Image saved to %s"%outfile)
        plt.close(f)

def show_ftir_data():
    '''
    '''
    datadir='Data/campaigns/Wgong/'
    data, attrs= fio.read_hdf5(datadir+'ftir_2007.h5')

    datetime = data['DATETIME'] # MJD2K: days since jan 1?
    dates=util.date_from_mjd2k(datetime)

    height = data['ALTITUDE']
    # VAR_NOTES = The altitudes are the centre (geometric mean of layer boundaries) of the altitude layers
    # VAR_UNITS = km


    # column hcho [molec cm-2]
    hcho_col = data['H2CO.COLUMN_ABSORPTION.SOLAR']

    # profile in ppmv, and apriori and AVG KERNAL
    hcho_prof     = data['H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR']
    # apriori has 1 entry per measurement but all are equal
    hcho_prof_apr = data['H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_APRIORI']
    hcho_prof_ak  = data['H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_AVK']

    def show_distr(x,y, vertical=False, label=None, color='blue', linecolor='k', quantile=False):
        var_2d=np.copy(y)
        if vertical:
            var_2d=np.copy(x)
        mid=np.nanmean(var_2d, axis=0)
        lower=mid - np.nanstd(var_2d, axis=0)
        upper=mid + np.nanstd(var_2d, axis=0)
        if quantile:
            lower=np.nanpercentile(var_2d,25,axis=0)
            upper=np.nanpercentile(var_2d,75,axis=0)

        if vertical:
            plt.fill_betweenx(y,lower,upper, color=color)
            plt.plot(mid,y,color=linecolor, linewidth=2)
        else:
            plt.fill_between(x,lower,upper, color=color)
            plt.plot(x,mid,color=linecolor, linewidth=2)

    show_distr(hcho_prof,height,vertical=True, color='blue')
    show_distr(hcho_prof_apr,height, vertical=True,color='red')
    plt.xlabel('hcho [ppmv]')
    plt.ylabel('height [km]')



def check_entries(d0=datetime(2005,1,1),d1=datetime(2005,1,31)):
    day=omhchorp(d0)
    month=omhchorp(d0,d1)

    inds_aus = day.inds_aus(maskocean=False)

    pp = day.data['AMF_PP'][inds_aus]
    gc = day.data['AMF_GC'][inds_aus]
    ppm = month.data['AMF_PP'][:,inds_aus]
    gcm = month.data['AMF_GC'][:,inds_aus]


    print(np.sum(~np.isnan(pp)),' good pp entries (1 day)')
    print(np.sum(~np.isnan(gc)),' good gc entries (1 day)')
    print(np.sum(~np.isnan(ppm)),' good pp entries (1 month)')
    print(np.sum(~np.isnan(gcm)),' good gc entries (1 month)')

def emisssions_vs_firefilter(d0=datetime(2005,1,1)):
    '''
    '''
    Enew=E_new(d0,)
    # mean Emissions estimates vs eachother

    f,axes = plt.subplots(3,3,figsize=(15,15))
    region=[Enew.lats[0],Enew.lons[0],Enew.lats[-1],Enew.lons[-1]]

    lats=Enew.lats
    lons=Enew.lons
    arrs=[[Enew.E_VCC_OMI, Enew.E_VCC, Enew.E_VCC_PP],
          [Enew.E_VCC_OMI_f, Enew.E_VCC_f, Enew.E_VCC_PP_f],
          []]

    labels=[['OMI','GC','PP'],
            ['OMI_f','GC_f','PP_f'],
            ['fire-nofire','fire-nofire','fire-nofire']]

    linear=False
    vmin=1e10
    vmax=1e13
    for i in range(3):
        for j in range(3):
            plt.sca(axes[i,j])
            if i < 2:
                arr=np.nanmean(arrs[i][j], axis=0) # average over time
            elif i==2:
                arr=np.nanmean(arrs[1][j], axis=0) - np.nanmean(arrs[0][j], axis=0)
                vmin=-5e12; vmax=5e12; linear=True

            pp.createmap(arr,lats,lons,title=labels[i][j],
                         region=region,
                         vmin=vmin,vmax=vmax,linear=linear)

    pname='Figs/Emiss/FireFilter%s.png'%mstr
    plt.suptitle('Emissions with and without fire filter %s'%mstr,fontsize=25)
    plt.savefig(pname)
    plt.close()



# to be moved somewhere:
def firetest():
    fires_per_area,lats,lons=fio.read_MOD14A1(d0,True)
    fires,lats,lons=fio.read_MOD14A1(d0,False)
    earth_sa=510e6 # 510.1 million km2
    count_a=np.sum(fires)
    count_b=np.mean(fires_per_area)*earth_sa*1e3
    print(count_a,count_b)
    print((count_a-count_b)/count_b)

    region=[-20,-30,40,50]
    f,axes=plt.subplots(3,1)
    plt.sca(axes[0])
    pp.createmap(fires,lats,lons,title='MODIS Fires 20050102',
                 colorbar=None, region=region,
                 linear=False, vmin=1,vmax=3e6)

    # lats lons are .1x.1 degrees
    # Try lower and higher resolution function:
    hlats,hlons,hlat_e,hlon_e = util.lat_lon_grid(latres=0.25,lonres=0.3125)
    llats,llons,llat_e,llon_e = util.lat_lon_grid(latres=2.0,lonres=2.5)

    hfires=util.regrid(fires,lats,lons,hlats,hlons,groupfunc=np.nansum)
    lfires=util.regrid(fires,lats,lons,llats,llons,groupfunc=np.nansum)
    print(np.nansum(hfires))
    print(np.nansum(lfires))
    plt.sca(axes[1])
    pp.createmap(hfires,hlats,hlons,title='High res',
                 colorbar=False,region=region,
                 linear=False, vmin=1,vmax=3e6)

    plt.sca(axes[2])
    pp.createmap(lfires,llats,llons,title='Low res', region=region,
                 clabel='fire pixels', pname='test_fires.png',
                 linear=False,cmapname='Reds',vmin=1,vmax=3e6)


#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')

def test_store_emissions_month(month=datetime(2005,1,1), GCB=None, OMHCHORP=None,
                              region=pp.__AUSREGION__):
    '''
        Store a month of new emissions estimates into an he5 file
        TODO: Add monthly option to just store month averages and month emissions
    '''
    month=datetime(2005,1,1)
    GCB=None
    OMHCHORP=None
    #                          region=pp.__AUSREGION__
    # Dates required: day0, dayN, and list of days between
    day0=util.first_day(month)
    dayn=util.last_day(day0)
    days=util.list_days(day0,dayn)

    # Handy date strings
    mstr=dayn.strftime('%Y%m')
    d0str=day0.strftime("%Y%m%d")
    dnstr=dayn.strftime("%Y%m%d")

    # File location to write to
    ddir="Data/Isop/E_new/"
    fname=ddir+"emissions_%s.h5"%(mstr)

    if __VERBOSE__:
        print("Calculating %s-%s estimated emissions over %s"%(d0str,dnstr,str(region)))
        print("will save to file %s"%(fname))

    # Dicts which will be saved:
    outdata={}
    outattrs={}

    # Read omhchorp VCs, AMFs, Fires, Smoke, etc...
    if OMHCHORP is None:
        OMHCHORP=omhchorp(day0=day0,dayn=dayn, ignorePP=False)
    if GCB is None:
        GCB=GC_class.GC_biogenic(day0,) # data like [time,lat,lon,lev]

    # subset our lats/lons
    # Arrays to be subset
    arrs_names=['VCC_OMI','VCC_GC','VCC_PP',
                'firemask','smokemask','anthromask',
                'gridentries','ppentries','col_uncertainty_OMI',
                ]
    # list indices
    arrs_i={s:i for i,s in enumerate(arrs_names)}
    # data from OMHCHORP
    arrs=[getattr(OMHCHORP,s) for s in arrs_names]

    OMHsubsets=util.lat_lon_subset(OMHCHORP.lats,OMHCHORP.lons,region,data=arrs, has_time_dim=True)
    omilats=OMHsubsets['lats']
    omilons=OMHsubsets['lons']
    omilati=OMHsubsets['lati']
    omiloni=OMHsubsets['loni']
    # map subsetted arrays into another dictionary
    OMHsub = {s:OMHsubsets['data'][arrs_i[s]] for s in arrs_names}

    # Need Vertical colums, slope, and backgrounds all at same resolution to get emissions
    VCC_GC                = OMHsub['VCC_GC']
    VCC_PP                = OMHsub['VCC_PP']
    VCC_OMI               = OMHsub['VCC_OMI']
    pixels                = OMHsub['gridentries']
    pixels_PP             = OMHsub['ppentries']
    uncert                = OMHsub['col_uncertainty_OMI']
    firefilter            = OMHsub['firemask']+OMHsub['smokemask']
    anthrofilter          = OMHsub['anthromask']

    # lets have a look at these things
    if True:
        vmin=1e14
        vmax=4e16
        f=plt.figure(figsize=(14,14))
        plt.subplot(221)
        pp.createmap(VCC_GC[0],omilats,omilons, title='GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(222)
        pp.createmap(VCC_OMI[0],omilats,omilons, title='OMI', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(223)
        pp.createmap(np.nanmean(VCC_GC,axis=0),omilats,omilons, title='mean_GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(224)
        pp.createmap(np.nanmean(VCC_OMI,axis=0),omilats,omilons, title='mean_OMI', region=region,vmin=vmin,vmax=vmax)
        plt.savefig('temp_VCC_check.png')
        plt.close()

    # GC.model_slope gets slope and subsets the region
    # Then Map slope onto higher omhchorp resolution:
    slope_dict=GCB.model_slope(region=region)
    GC_slope=slope_dict['slope']
    gclats,gclons = slope_dict['lats'],slope_dict['lons']
    GC_slope = util.regrid_to_higher(GC_slope,gclats,gclons,omilats,omilons,interp='nearest')
    if True:
        plt.figure()
        vmin=1e3;vmax=2e4
        pp.createmap(GC_slope,omilats,omilons,region=region, cbarfmt='%.0f', cbarxtickrot=20,
                     linear=True,vmin=vmin,vmax=vmax,title='GC_slope')
        plt.savefig('temp_slope_check.png')
        plt.close()

    # Also save smearing
    smear, slats,slons = Inversion.smearing(month,region=region)
    pp.createmap(smear,slats,slons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
                 linear=True, vmin=1000, vmax=10000,
                 clabel='S', pname='Figs/GC/smearing_%s.png'%mstr, title='Smearing %s'%mstr)
    smear = util.regrid_to_higher(smear,slats,slons,omilats,omilons,interp='nearest')
    pp.createmap(smear,omilats,omilons, latlon=True, GC_shift=True, region=pp.__AUSREGION__,
                 linear=True, vmin=1000, vmax=10000,
                 clabel='S', pname='Figs/GC/smearing_%s_interp.png'%mstr, title='Smearing %s'%mstr)
    print("Smearing plots saved in Figs/GC/smearing...")
    outdata['smearing'] = smear
    outattrs['smearing']= {'desc':'smearing = Delta(HCHO)/Delta(E_isop), '+
            'where Delta is the difference between full and half isoprene '+
            'emission runs from GEOS-Chem for %s,'%mstr +
            ' interpolated linearly from 2x2.5 to 0.25x0.3125 resolution'}

    # TODO: Smearing Filter
    smearfilter = smear > Inversion.__Thresh_Smearing__#5000 # something like this


    # emissions using different columns as basis
    # Fully filtered
    out_shape=VCC_GC.shape
    E_gc        = np.zeros(out_shape) + np.NaN
    E_pp        = np.zeros(out_shape) + np.NaN
    E_omi       = np.zeros(out_shape) + np.NaN

    # unfiltered:
    E_gc_u      = np.zeros(out_shape) + np.NaN
    E_pp_u      = np.zeros(out_shape) + np.NaN
    E_omi_u     = np.zeros(out_shape) + np.NaN

    BG_VCC      = np.zeros(out_shape) + np.NaN
    BG_PP       = np.zeros(out_shape) + np.NaN
    BG_OMI      = np.zeros(out_shape) + np.NaN

    # Need background values from remote pacific
    BG_VCCa, bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_GC,
                                                            OMHCHORP.lats, OMHCHORP.lons,
                                                            average_lons=True,has_time_dim=True)
    BG_PPa , bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_PP,
                                                            OMHCHORP.lats, OMHCHORP.lons,
                                                            average_lons=True,has_time_dim=True)
    BG_OMIa, bglats, bglons = util.remote_pacific_background(OMHCHORP.VCC_OMI,
                                                            OMHCHORP.lats, OMHCHORP.lons,
                                                            average_lons=True,has_time_dim=True)

    if True:
        vmin=1e13
        vmax=1e15
        lats=OMHCHORP.lats
        f=plt.figure(figsize=(14,14))
        plt.plot(lats,BG_VCCa[0],label='BG_GC[0]')
        plt.plot(lats,BG_OMIa[0],label='BG_OMI[0]')
        plt.plot(lats,np.nanmean(BG_VCCa,axis=0),label='mean BG_GC')
        plt.plot(lats,np.nanmean(BG_OMIa,axis=0),label='mean BG_OMI')
        plt.legend(loc='best')
        plt.xlabel('latitude')
        plt.ylabel('VCC [molec/cm2]')
        plt.savefig('temp_BG_check.png')
        plt.close()


    # cut the omhchorp backgrounds down to our latitudes
    BG_VCC = BG_VCCa[:,omilati]
    BG_PP = BG_PPa[:,omilati]
    BG_OMI = BG_OMIa[:,omilati]
    # Repeat them along our longitudes so we don't need to loop
    BG_VCC = np.repeat(BG_VCC[:,:,np.newaxis],len(omiloni),axis=2)
    BG_PP = np.repeat(BG_PP[:,:,np.newaxis],len(omiloni),axis=2)
    BG_OMI = np.repeat(BG_OMI[:,:,np.newaxis],len(omiloni),axis=2)
    # Also repeat Slope array along time axis to avoid looping
    GC_slope= np.repeat(GC_slope[np.newaxis,:,:], len(days),axis=0)



    if True:
        print("Enew Calc Shapes")
        print(VCC_GC.shape,BG_VCC.shape,GC_slope.shape)
        plt.plot(omilats,BG_VCC[0,:,0],'r',label='BG VCC')
        plt.plot(omilats,BG_OMI[0,:,0],'m', label='BG OMI')
        plt.legend()
        plt.title('Background over latitude')

    # Run calculation with no filters applied:
    E_gc_u       = (VCC_GC - BG_VCC) / GC_slope
    E_pp_u       = (VCC_PP - BG_PP) / GC_slope
    E_omi_u      = (VCC_OMI - BG_OMI) / GC_slope

    # run with filters
    # apply filters
    allmasks            = (firefilter + anthrofilter)>0 # + smearfilter
    print('FILTER DEETS')
    print(allmasks.shape)
    print(E_gc.shape)
    print(np.nansum(allmasks))
    print(np.nanmean(E_gc_u),np.nanmean(E_gc_u[allmasks]))


    assert not np.isnan(np.nansum(E_gc_u[allmasks])), 'Filtering nothing!?'

    # Mask gridsquares using fire and anthro filters
    E_gc                = np.copy(E_gc_u)
    E_pp                = np.copy(E_pp_u)
    E_omi               = np.copy(E_pp_u)
    E_gc[allmasks]      = np.NaN
    E_pp[allmasks]      = np.NaN
    E_omi[allmasks]     = np.NaN
    if True:
        # lets have a close look at these things
        vmin=1e10
        vmax=1e13
        f=plt.figure(figsize=(14,14))
        plt.subplot(211)
        # Plot E_new before and after filtering
        m,cs,cb=pp.createmap(E_gc_u[0],omilats,omilons, title='E_GC_u', region=[-40,130,-20,155],
                             vmin=vmin,vmax=vmax)

        # plot dots where filter should be
        for yi,y in enumerate(omilats):
            for xi,x in enumerate(omilons):
                if firefilter[0,yi,xi]:
                   mx,my = m(x,y)
                   m.plot(mx,my,'x',markersize=6,color='k')
                if anthrofilter[0,yi,xi]:
                   mx,my = m(x,y)
                   m.plot(mx,my,'d',markersize=3,color='k')
        plt.subplot(212)
        m,cs,cb=pp.createmap(E_gc[0],omilats,omilons, title='E_GC', region=[-40,130,-20,155],
                             vmin=vmin,vmax=vmax)

        pname='temp_E_filtering_check.png'
        plt.savefig(pname)
        print('saved ',pname)
        plt.close()

    if True:
        vmin=0
        vmax=1
        f=plt.figure(figsize=(14,14))
        plt.subplot(221)
        pp.createmap(firefilter[0],omilats,omilons,vmin=vmin,vmax=vmax,
                     region=region,linear=True,title='firefilter[0]')
        plt.subplot(222)
        pp.createmap(anthrofilter[0],omilats,omilons,vmin=vmin,vmax=vmax,
                     region=region,linear=True,title='anthrofilter[0]')
        plt.subplot(223)
        pp.createmap(np.nanmean(firefilter,axis=0),omilats,omilons,vmin=vmin,vmax=vmax,
                     region=region,linear=True,title='mean firefilter')
        plt.subplot(224)
        pp.createmap(np.nanmean(anthrofilter,axis=0),omilats,omilons,vmin=vmin,vmax=vmax,
                     region=region,linear=True,title='mean anthrofilter')
        plt.savefig('temp_Filters.png')
        plt.close()

    if True:
        vmin=1e10
        vmax=1e13
        f=plt.figure(figsize=(14,14))
        plt.subplot(221)
        pp.createmap(E_gc_u[0], omilats,omilons, title='E_GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(222)
        pp.createmap(E_omi_u[0], omilats,omilons, title='E_omi', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(223)
        pp.createmap(np.nanmean(E_gc_u,axis=0),omilats,omilons, title='mean_E_GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(224)
        pp.createmap(np.nanmean(E_omi_u,axis=0),omilats,omilons, title='mean_E_OMI', region=region,vmin=vmin,vmax=vmax)
        plt.suptitle('unfiltered')
        plt.savefig('temp_E_unfiltered_check.png')
        plt.close()

        f=plt.figure(figsize=(14,14))
        plt.subplot(221)
        pp.createmap(E_gc[0], omilats,omilons, title='E_GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(222)
        pp.createmap(E_omi[0], omilats,omilons, title='E_omi', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(223)
        pp.createmap(np.nanmean(E_gc,axis=0),omilats,omilons, title='mean_E_GC', region=region,vmin=vmin,vmax=vmax)
        plt.subplot(224)
        pp.createmap(np.nanmean(E_omi,axis=0),omilats,omilons, title='mean_E_OMI', region=region,vmin=vmin,vmax=vmax)
        plt.suptitle('filtered')
        plt.savefig('temp_E_filtered_check.png')
        plt.close()

    # Lets save both monthly averages and the daily amounts
    #

    # Save the backgrounds, as well as units/descriptions
    outdata['BG_VCC']    = BG_VCC
    outdata['BG_PP']     = BG_PP
    outdata['BG_OMI']    = BG_OMI
    outattrs['BG_VCC']   = {'units':'molec/cm2','desc':'Background: VCC zonally averaged from remote pacific'}
    outattrs['BG_PP']    = {'units':'molec/cm2','desc':'Background: VCC_PP zonally averaged from remote pacific'}
    outattrs['BG_OMI']   = {'units':'molec/cm2','desc':'Background: VCC_OMI zonally averaged from remote pacific'}

    # Save the Vertical columns, as well as units/descriptions
    outdata['VCC_GC']     = VCC_GC
    outdata['VCC_PP']     = VCC_PP
    outdata['VCC_OMI']    = VCC_OMI
    outattrs['VCC_GC']    = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column using recalculated shape factor, fire and anthro masked'}
    outattrs['VCC_PP']    = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column using PP code, fire and anthro masked'}
    outattrs['VCC_OMI']   = {'units':'molec/cm2','desc':'OMI (corrected) Vertical column, fire and anthro masked'}

    # Save the Emissions estimates, as well as units/descriptions
    outdata['E_VCC_GC']     = E_gc
    outdata['E_VCC_PP']     = E_pp
    outdata['E_VCC_OMI']    = E_omi
    outdata['E_VCC_GC_u']   = E_gc_u
    outdata['E_VCC_PP_u']   = E_pp_u
    outdata['E_VCC_OMI_u']  = E_omi_u
    if True:
        print("______EMISSIONS_____")
        print("Filtered  ,    Unfiltered")
        _blah = [print("%.5e  ,  %.5e "%(np.nanmean(E),np.nanmean(E_u))) for E,E_u in zip([E_omi,E_gc,E_pp],[E_omi_u,E_gc_u,E_pp_u])]

    outattrs['E_VCC_GC']    = {'units':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope'}
    outattrs['E_VCC_PP']    = {'units':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope'}
    outattrs['E_VCC_OMI']   = {'units':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope'}
    outattrs['E_VCC_GC_u']  = {'units':'molec OR atom C???/cm2/s',
                               'desc':'Isoprene Emissions based on VCC and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_PP_u']  = {'units':'molec OR atom C??/cm2/s',
                               'desc':'Isoprene Emissions based on VCC_PP and GC_slope, unmasked by fire or anthro'}
    outattrs['E_VCC_OMI_u'] = {'units':'molec OR/cm2/s',
                               'desc':'Isoprene emissions based on VCC_OMI and GC_slope, unmasked by fire or anthro'}

    # Extras like pixel counts etc..
    outdata['firefilter']   = firefilter.astype(np.int)
    outdata['anthrofilter'] = anthrofilter.astype(np.int)
    outdata['smearfilter']  = smearfilter.astype(np.int)
    outdata['pixels']       = pixels
    outdata['pixels_PP']    = pixels_PP
    outdata['uncert_OMI']   = uncert
    outattrs['firefilter']  = {'units':'N/A',
                               'desc':'Squares with more than one fire (over today or last two days, in any adjacent square) or AAOD greater than %.1f'%(fio.__Thresh_AAOD__)}
    outattrs['anthrofilter']= {'units':'N/A',
                               'desc':'Squares with tropNO2 from OMI greater than %.1e or yearly averaged tropNO2 greater than %.1e'%(fio.__Thresh_NO2_d__,fio.__Thresh_NO2_y__)}
    outattrs['smearfilter'] = {'units':'N/A',
                               'desc':'Squares where smearing greater than %.1f'%(Inversion.__Thresh_Smearing__)}
    outattrs['uncert_OMI']  = {'units':'?? molec/cm2 ??',
                               'desc':'OMI pixel uncertainty averaged for each gridsquare'}
    outattrs['pixels']      = {'units':'n',
                               'desc':'OMI pixels used for gridsquare VC'}
    outattrs['pixels_PP']   = {'units':'n',
                               'desc':'OMI pixels after PP code used for gridsquare VC'}

    # Adding time dimension (needs to be utf8 for h5 files)
    #dates = np.array([d.strftime("%Y%m%d").encode('utf8') for d in days])
    dates = np.array([int(d.strftime("%Y%m%d")) for d in days])
    outdata["time"]=dates
    outattrs["time"]={"format":"%Y%m%d", "desc":"year month day as integer (YYYYMMDD)"}
    fattrs={'region':"SWNE: %s"%str(region)}
    fattrs['date range']="%s to %s"%(d0str,dnstr)

    # Save lat,lon
    outdata['lats']=omilats
    outdata['lons']=omilons
    outdata['lats_e']=util.edges_from_mids(outdata['lats'])
    outdata['lons_e']=util.edges_from_mids(outdata['lons'])


    # Save file, with attributes
    print("NORMALLY WOULD SAVE NOW BUT THS IS TEST ENV")
    #fio.save_to_hdf5(fname,outdata,attrdicts=outattrs,fattrs=fattrs)
    if __VERBOSE__:
        print("%s should now be saved"%fname)
