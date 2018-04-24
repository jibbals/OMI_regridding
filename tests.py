'''
    File to test various calculations
    Tests for coding are moved to utilities/tests.py
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
from utilities import fio
import reprocess
from utilities import plotting as pp
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes.omhchorp import omhchorp as omrp
from classes.gchcho import match_bottom_levels
from classes.GC_class import GC_tavg

import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy

from datetime import datetime#, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
from matplotlib.ticker import FormatStrFormatter # tick formats

#import matplotlib.patches as mpatches
import seaborn # kdeplots

import random # random number generation
import timeit
# EG using timer:
#start_time=timeit.default_timer()
#runprocess()
#elapsed = timeit.default_timer() - start_time
#print ("TIMEIT: Took %6.2f seconds to runprocess()"%elapsed)

##############################
######### GLOBALS ############
##############################

cities = {'Syd':[-33.87,151.21], # Sydney
          'Can':[-35.28,149.13], # Canberra
          'Mel':[-37.81,144.96], # Melbourne
          'Ade':[-34.93,138.60], # Adelaide
         }

Ohcho='$\Omega_{HCHO}$'
Ovc='$\Omega_{VC}$'
Ovcc='$\Omega_{VCC}$'
Ovccpp='$\Omega_{VCC-PP}$' # Paul Palmer VCC
Oomi='$\Omega_{OMI}$'
Oomic="$\Omega_{OMI_{RSC}}$" #corrected original product
Ogc='$\Omega_{GC}$'


##############################
########## FUNCTIONS #########
##############################

def mmm(arr):
    print("%1.5e, %1.5e, %1.5e"%(np.nanmin(arr),np.nanmean(arr),np.nanmax(arr)))

def check_array(array, nonzero=False):
    '''
    print basic stuff about an array
    '''
    arrayt=array
    if nonzero:
        arrayt=array[np.nonzero(array)]
    print ('mean :%f'%np.nanmean(arrayt))
    print ('min :%f'%np.nanmin(arrayt))
    print ('max :%f'%np.nanmax(arrayt))
    print ('count :%f'%np.sum(np.isfinite(arrayt)))
    print ('shape :'+str(np.shape(arrayt)))

def no2_map(data, lats, lons, vmin, vmax, subzones, colors):
    '''
        Plot australia, with subzones and stuff added
    '''

    cmapname='plasma'

    bmap,cs,cb = pp.createmap(data, lats, lons, region=subzones[0],
                              vmin=vmin, vmax=vmax, clabel='molec/cm2',
                              cmapname=cmapname)
    # Add cities to map
    for city,latlon in cities.items():
        pp.add_point(bmap,latlon[0],latlon[1],markersize=12,
                     color='floralwhite', label=city, fontsize=12,
                     xlabeloffset=0,ylabeloffset=3000)

    # Add squares to map:
    for i,subzone in enumerate(subzones[1:]):
        pp.add_rectangle(bmap,subzone,color=colors[i+1],linewidth=2)

    return bmap,cs,cb

def no2_timeseries(no2_orig,dates,lats,lons,subzones,colors,ylims=[2e14,4e15],
                   print_values=False):
    '''
        time series for each subzone in no2_tests
    '''
    lw=[3,1,1,1,1]
    doys=[d.timetuple().tm_yday for d in dates]

    # loop over subzones
    for i,subzone in enumerate(subzones):
        # Subset our data to subzone
        no2=np.copy(no2_orig)
        lati,loni=util.lat_lon_range(lats,lons,subzone)
        no2 = no2[:,lati,:]
        no2 = no2[:,:,loni]

        # Mask ocean
        oceanmask=util.get_mask(no2[0],lats[lati],lons[loni],masknan=False,maskocean=True)
        print("Removing %d ocean pixels"%(365*np.sum(oceanmask)))
        no2[:,oceanmask] = np.NaN

        # Also remove negatives
        negmask=no2 < 0
        print("Removing %d negative pixels"%(np.sum(negmask)))
        no2[negmask]=np.NaN

        # get mean and percentiles of interest for plot
        #std = np.nanstd(no2,axis=(1,2))
        upper = np.nanpercentile(no2,75,axis=(1,2))
        lower = np.nanpercentile(no2,25,axis=(1,2))
        mean = np.nanmean(no2,axis=(1,2))
        totmean = np.nanmean(no2)

        plt.plot(doys, mean, color=colors[i],linewidth=lw[i])
        # Add IQR shading for first plot
        if i==0:
            plt.fill_between(doys, lower, upper, color=colors[i],alpha=0.2)

        # show yearly mean
        plt.plot([370,395],[totmean,totmean], color=colors[i],linewidth=lw[i]+1)

        # change to log y scale?
        plt.ylim(ylims)
        plt.yscale('log')
        plt.ylabel('molec/cm2')
        yticks=[2e14,5e14,1e15,4e15]
        ytickstr=['%.0e'%tick for tick in yticks]
        plt.yticks(yticks,ytickstr)
        plt.xlabel('Day of year')

        # Print some stats if desired
        if print_values:
            with open("no2_output.txt",'a') as outf: # append to file
                outf.write("Stats for %d, %s, %s"%(i, str(subzone), colors[i]))
                outf.write("  yearly mean:  %.3e"%totmean)
                outf.write("          std:  %.3e"%np.nanstd(no2))
                outf.write("      entries:  %d"%np.sum(~np.isnan(no2)))
                outf.write("  gridsquares:  %d"%np.prod(np.shape(no2)))
            print("Wrote stats to no2_output.txt")


#############################################################################
######################       TESTS                  #########################
#############################################################################

def omno2d_filter_determination(year=datetime(2005,1,1),
                                thresh=1e15,
                                region=pp.__AUSREGION__):
    '''
        examine year of omno2d and look at densities and threshhold
    '''

    # First read and whole year of NO2
    dat, attrs = fio.read_omno2d(datetime(year.year,1,1), util.last_day(datetime(year.year,12,1)))
    # Tropospheric cloud screened (<30%) no2 columns (molec/cm2)
    no2_orig=dat['tropno2']
    lats=dat['lats']
    lons=dat['lons']
    dates=dat['dates']

    # Want to look at timeseires and densities in these subregions:
    subzones=[region, # All of Australia
              [-36,148,-32,153], # Canberra, Newcastle, and Sydney
              [-36,134,-33,140], # Adelaide and port lincoln
              [-30,125,-25,135], # Emptly land
              [-39,142,-36,148], # Melbourne
              ]
    colors=['k','red','green','cyan','darkred']

    # Make two plots using following two subfunctions
    def no2_densities():
        '''
            Look at densities of no2 pixels from omno2d
        '''
        plotname=year.strftime('Figs/OMNO2_densities_%Y.png')
        no2=np.copy(no2_orig)

        # Get mean for whole year
        no2_mean = np.nanmean(no2,axis=0)

        # plot map with regions:
        plt.figure(figsize=[16,14])

        title = 'Mean OMNO2d %d'%year.year
        vmin = 1e14
        vmax = 1e15
        plt.subplot(2,2,1)
        bmap,cs,cb = no2_map(no2_mean,lats,lons,vmin,vmax,subzones,colors)
        plt.title(title)

        # One density plot for each region in subzones
        for i,subzone in enumerate(subzones):
            # Subset our data to subzone
            lati,loni=util.lat_lon_range(lats,lons,subzone)
            no2 = np.copy(no2_orig)
            no2 = no2[:,lati,:]
            no2 = no2[:,:,loni]

            # Mask ocean
            oceanmask=util.get_mask(no2[0],lats[lati],lons[loni],masknan=False,maskocean=True)
            print("Removing %d ocean pixels"%(365*np.sum(oceanmask)))
            no2[:,oceanmask] = np.NaN

            # Also remove negatives?
            negmask=no2 < 0
            print("Removing %d negative pixels"%(np.sum(negmask)))
            no2[negmask]=np.NaN

            # all Australia density map
            bw=5e13 # bin widths
            if i == 0:
                plt.subplot(2, 2, 2)
                pp.density(no2,bw=bw,color=colors[i], linewidth=2)
            else:
                plt.subplot(2,4,i+4)
                pp.density(no2,bw=bw, color=colors[i], linewidth=2) # should work with 3d
            plt.xlim([0,5e15])
            plt.plot([thresh,thresh],[0,1], '--k')

        plt.suptitle("OMNO2d NO2 columns %d"%year.year)
        plt.savefig(plotname)
        print("saved ",plotname)
        plt.close()

    def typical_no2():
        '''
            Plot of NO2 from OMNO2d product over Australia, including time series
        '''

        plotname=year.strftime('Figs/OMNO2_timeseries_%Y.png')


        # Tropospheric cloud screened (<30%) no2 columns (molec/cm2)
        no2=np.copy(no2_orig)
        print(no2.shape)
        no2_mean, no2_std = np.nanmean(no2,axis=0), np.nanstd(no2,axis=0)

        # plot stuff:
        plt.figure(figsize=[16,16])
        # MEAN | STDev
        titles = ['Mean %d'%year.year, 'Standard deviation %d'%year.year]
        vmins  = [1e14, None]
        vmaxes = [5e15, None]

        axes=[]
        bmaps=[]
        for i,arr in enumerate([no2_mean,no2_std]):
            axes.append(plt.subplot(2,2,i+1))
            vmin,vmax=vmins[i],vmaxes[i]
            bmap,cs,cb = no2_map(arr, lats, lons,vmin,vmax,subzones,colors)
            plt.title(titles[i])

            bmaps.append(bmap) # save the bmap for later

        # Bottom row
        axes.append(plt.subplot(2,1,2))

        # For each subset here, plot the timeseries
        no2_timeseries(no2_orig,dates,lats,lons,subzones,colors)

        plt.title('Mean time series (ocean masked) over %d'%year.year)

        plt.suptitle("OMNO2d NO2 columns")
        plt.savefig(plotname)
        print("saved ",plotname)
        plt.close()

    def CheckThresh():
        ''' Look at affect of applying threshhold '''
        pname='Figs/OMNO2_threshaffect_%d.png'%year.year
        fig, axes = plt.subplots(2,2,figsize=[16,16])
        no2=np.copy(no2_orig)
        no2_f=np.copy(no2)
        no2_f[no2_f>thresh] = np.NaN # filtered

        # plot stuff:
        titles = ['Mean %d'%year.year, 'With %.1e threshhold'%thresh]
        vmin = 1e14
        vmax = 2e15
        means=[]
        for i,arr in enumerate([no2,no2_f]):
            # Plot map and regions with and without values above threshhold filtered out
            # Also time series with and without filter

            means.append(np.nanmean(arr,axis=0))
            plt.sca(axes[0,i])
            no2_map(means[i],lats,lons,vmin,vmax,subzones,colors)
            plt.title(titles[i])

            plt.sca(axes[1,i])
            no2_timeseries(arr,dates,lats,lons,subzones,colors,print_values=True)

        plt.savefig(pname)
        print('saved ',pname)
        plt.close()


    # Now run both those functions
    #no2_densities()
    #typical_no2()
    CheckThresh()

def typical_aaod_month(month=datetime(2005,11,1)):
    ''' '''
    ymstr=month.strftime("%Y%m")
    pname2='Figs/AAOD_month_%s.png'%ymstr
    region=pp.__AUSREGION__
    #vmin=1e-3
    #vmax=1e-1
    vmin,vmax=1e-7,5e-2
    cmapname='pink_r'

    # also show a month of aaod during nov 2005 ( high transport month )
    plt.figure()
    plt.subplot(211)

    # read the aaod and average over the month
    aaod,dates,lats,lons=fio.read_smoke(month,util.last_day(month))
    aaod=np.nanmean(aaod,axis=0)

    # create map
    pp.createmap(aaod,lats,lons,region=region,cmapname=cmapname,
                 vmin=vmin,vmax=vmax,set_bad='blue')

    # also show density map
    plt.subplot(212)
    pp.density(aaod,lats,lons,region=region)

    plt.savefig(pname2)
    print("Saved ",pname2)
    plt.close()

def typical_aaods():
    '''
    Check typical aaod over Australia during specific events
    row a) normal
    row b) Black saturday: 20090207-20090314
    row c) transported smoke: 20051103,08,17
    row d) dust storm : 20090922-24
    '''

    # read particular days of aaod
    dates = [ datetime(2007,8,30), datetime(2009,2,19),
              datetime(2005,11,8), datetime(2009,9,23) ]

    # plot stuff
    plt.figure(figsize=(16,16))
    pname='Figs/typical_AAODs.png'
    region=pp.__AUSREGION__
    vmin=1e-4
    vmax=1e-1
    cmapname='pink_r'
    titles=['normal','black saturday','transported plume','dust storm']
    zooms=[None,[-40,140,-25,153],[-42,130,-20,155],[-40,135,-20,162]]
    TerraModis=['Figs/TerraModis_Clear_20070830.png',
                'Figs/TerraModis_BlackSaturday_20090219.png',
                'Figs/TerraModis_TransportedSmoke_20050811.png',
                'Figs/TerraModis_DustStorm_20090923.png']
    linear=False
    thresh=0.03

    for i,day in enumerate(dates):
        zoom=region
        plt.subplot(4,4,1+i*4)
        ymd=day.strftime('%Y %b %d')
        title = titles[i] +' '+ ymd
        aaod, lats, lons = fio.read_AAOD(day)
        m, cs, cb = pp.createmap(aaod, lats, lons, title=title, region=zoom,
                                 vmin=vmin, vmax=vmax, linear=linear,
                                 cmapname=cmapname, set_bad='blue')

        # Add hatch over threshhold values (if they exists)
        #(m, data, lats, lons, thresh, region=None):
        #pp.hatchmap(m,aaod,lats,lons,thresh,region=zoom, hatch='x',color='blue')

        if zooms[i] is not None:
            zoom=zooms[i]
            plt.subplot(4,4,2+i*4)
            m,cs,cb= pp.createmap(aaod, lats, lons ,region=zoom,
                                  vmin=vmin, vmax=vmax, linear=linear,
                                  cmapname=cmapname)
            # Add hatch to minimap also
            #pp.hatchmap(m,aaod,lats,lons,thresh,region=zoom, hatch='x',color='blue')

        plt.subplot(4,4,3+i*4)
        aaod, lats, lons = pp.density(aaod,lats,lons,region=zoom, vertical=True)
        plt.plot([0,50],[thresh,thresh]) # add line for thresh
        plt.title('density')
        plt.ylabel('AAOD')
        plt.gca().yaxis.set_label_position("right")
        #plt.xlim([-0.02,0.1])
        print('Mean AAOD=%.3f'%np.nanmean(aaod))
        print("using %d gridsquares"%np.sum(~np.isnan(aaod)))

        if TerraModis[i] is not None:
            plt.subplot(4,4,4+i*4)
            pp.plot_img(TerraModis[i])
            plt.title(ymd)

    plt.tight_layout()

    plt.savefig(pname)
    plt.close()
    print("Saved ",pname)



def smoke_vs_fire(d0=datetime(2005,1,1),dN=datetime(2005,1,31),region=pp.__AUSREGION__):
    '''
        Compare fire counts to smoke aaod in the omhchorp files
    '''
    d0str=d0.strftime('%Y%m%d')
    if dN is None:
        dN = d0

    dNstr=dN.strftime('%Y%m%d')
    n_times=(dN-d0).days + 1

    # Read the products from modis and omi, interpolated to 1x1 degrees
    fires, _dates, _modlats, _modlons = fio.read_fires(d0, dN, latres=1,lonres=1)
    aaod, _dates, _omilats, _omilons = fio.read_smoke(d0,dN,latres=1,lonres=1)

    assert all(_modlats==_omilats), 'interpolation is not working'
    lats=_modlats
    lons=_modlons
    # data fires and aaod = [times, lats, lons]
    # data.fires.shape; data.AAOD.shape

    f,axes=plt.subplots(2,2,figsize=(16,16)) # 2 rows 2 columns

    titles=['Fires','AAOD$_{500nm}$']
    linear=[True,False]
    fires=fires.astype(np.float)
    fires[fires<0] = np.NaN
    aaod[aaod<0]   = np.NaN # should do nothing

    # Average over time
    fires=np.nanmean(fires,axis=0)
    aaod=np.nanmean(aaod,axis=0)

    for i,arr in enumerate([fires,aaod]):

        # plot into right axis
        plt.sca(axes[0,i])
        pp.createmap(arr,lats,lons,title=titles[i],
                     linear=linear[i],region=region,
                     colorbar=True,cmapname='Reds',)
                     #vmin=vmins[i],vmax=vmaxes[i])
    plt.suptitle('Fires vs AAOD %s-%s'%(d0str,dNstr))

    # third subplot: regression
    plt.sca(axes[1,0])
    X=fires
    Y=aaod

    subset=util.lat_lon_subset(lats,lons,region,[X,Y])
    X,Y=subset['data'][0],subset['data'][1]
    lats,lons=subset['lats'],subset['lons']
    pp.plot_regression(X,Y,logscale=False,legend=False)
    plt.xlabel('Fires')
    plt.ylabel("AAOD")
    plt.title('Correlation')

    # Fourth plot: density of AAOD,Fires:
    plt.subplot(426)
    #plt.sca(axes[1,1])

    seaborn.set_style('whitegrid')
    seaborn.kdeplot(Y.flatten())# linestyle='-')
    plt.title('aaod density')
    plt.subplot(428)
    seaborn.set_style('whitegrid')
    seaborn.kdeplot(X.flatten())# linestyle='-')
    plt.title('fires density')

    pname='Figs/Smoke_vs_Fire_%s-%s.png'%(d0str,dNstr)
    plt.savefig(pname)
    print("Saved figure ",pname)


def smearing_calculation(date=datetime(2005,1,1)):
    '''
        S=change in HCHO column / change in E_isop
    '''
    region=pp.__AUSREGION__
    # READ normal and halfisop run outputs:
    full=GC_tavg(date)
    half=None

def check_HEMCO_restarts():
    '''
        Check how different the hemco restarts are between UCX and tropchem
    '''
    fpat='Data/GC_Output/%s/restarts/HEMCO_restart.200501010000.nc'
    fucx=fpat%'UCX_geos5_2x25'
    ftrp=fpat%'geos5_2x25_tropchem'
    ucx=fio.read_netcdf(fucx)
    trp=fio.read_netcdf(ftrp)
    print("Name, run, shape, mean")
    for name in ucx.keys():#['PARDF_DAVG','PARDR_DAVG']:
        ucx_mean=np.nanmean(ucx[name])
        print (name, "ucx", ucx[name].shape, ucx_mean)
        if name in trp.keys():
            trp_mean=np.nanmean(trp[name])
            print(name, 'trop', trp[name].shape ,trp_mean)
            print(['NOT EQUAL','EQUAL'][np.isclose(trp_mean,ucx_mean)])
        #createmap(ucx[
    return None



def compare_GC_OMI_new(date=datetime(2005,1,1),aus=True):
    '''
    '''
    vmin=1e14; vmax=1e17
    lllat=-75; urlat=75; lllon=-170; urlon=170
    austr=''
    dstr=date.strftime("%Y%m%d")
    if aus:
        lllat=-45; urlat=-8; lllon=108; urlon=156
        austr='AUS_'

    f=plt.figure(figsize=[13,13])
    # Figure of OMIcc, VCC, GC, diffs
    plt.subplot(223)
    gc=fio.read_gchcho(date)
    m,cs,cb=gc.PlotVC(lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=vmin, vmax=vmax, cm2=True)
    #pname="Figs/map_%sGC_%s.png"%(austr,dstr)
    plt.title(Ogc)

    om=omrp(date)
    lats=om.latitude
    lons=om.longitude
    VCC_OMI=om.VC_OMI_RSC
    VCC=om.VCC
    # plot the regridded OMI and reprocessed maps:
    for arr,arrstr,i in zip([VCC_OMI,VCC],[Oomi,Ogc],range(2)):
        plt.subplot(221+i)
        pp.createmap(arr,lats,lons, vmin=vmin, vmax=vmax, latlon=True,
              lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, colorbar=False)
        plt.title(arrstr)

    # relative differences
    #plt.subplot(224)
    #diffs=100.0*(VCC - VCC_OMI) / VCC_OMI
    #pp.linearmap(diffs,lats,lons,vmin=vmin,vmax=vmax,latlon=True,
    #          lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    #plt.title("%s vs %s relative difference"%(Ogc,Oomi))

    pname="Figs/map_GOM_%s%s.png"%(austr,dstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close()


def compare_products(date=datetime(2005,1,1), oneday=True, positiveonly=False,
                     lllat=-60, lllon=-179, urlat=50, urlon=179,pltname=""):
    '''
    Test a day or 8-day reprocessed HCHO map
    Plot VCs, both OMI and Reprocessed, and the RMA corellations
    TODO: Update to look only at corrected, including RandalMartin calculated
    '''

    # Grab reprocessed OMI data
    om=omrp(date,oneday=oneday)

    lons,lats =np.meshgrid(om.longitude,om.latitude)
    oceanmask=om.oceanmask  # true for ocean squares
    omi=om.VC_OMI
    vcc=om.VCC
    sgc='$\Omega_{OMIGCC}$'
    somi='$\Omega_{OMI}$'

    # Plot
    # VC_omi, VC_rp
    # diffs, Corellations

    f = plt.figure(num=0,figsize=(16,18))

    # Plot OMI, OMRP
    # set currently active axis from [2,2] axes array
    plt.subplot(221)
    m,cs,cb = pp.createmap(omi,lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title(somi)
    plt.subplot(222)
    m,cs,cb = pp.createmap(vcc,lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title(sgc)
    #ax=plt.subplot(223)
    #m,cs,cb = pp.linearmap(100.0*(omi-vcc)/omi, lats, lons,
    #                       vmin=-120, vmax=120,
    #                       lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #plt.title('100(%s-%s)/%s'%(somi,sgc,somi))
    plt.subplot(212)
    if positiveonly:
        omi[omi<0]=np.NaN
        vcc[vcc<0]=np.NaN
    pp.plot_corellation(omi, vcc, oceanmask=oceanmask,verbose=True)
    plt.title('RMA corellation')
    plt.xlabel(somi)
    plt.ylabel(sgc)

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=30)
    plt.tight_layout()
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%sproducts%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")


def check_products(date=datetime(2005,1,1), oneday=True, pltname='',
                   lllat=-60, lllon=-179, urlat=50, urlon=179):
    '''
    Test a day or 8-day reprocessed HCHO map
    Plot VCs, both OMI and Reprocessed, and the RMA corellations
    '''

    # Grab reprocessed OMI data
    om=omrp(date,oneday=oneday)

    counts = om.gridentries
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lons,lats =np.meshgrid(om.longitude,om.latitude)
    oceanmask=om.oceanmask  # true for ocean squares
    omi=om.VC_OMI
    vcc=om.VCC
    sgc='$\Omega_{OMIGCC}$'
    somi='$\Omega_{OMI}$'

    # Plot
    # VC_omi, VC_rp
    # diffs, Corellations

    plt.figure(num=0,figsize=(14,12))

    pp.plot_corellation(omi, vcc, oceanmask=oceanmask, verbose=True, logscale=False, lims=[-1e16,1.01e17])
    plt.xlabel(somi)
    plt.ylabel(sgc)
    #plt.tick_params(axis='both', which='major', labelsize=24)

    # save plot
    yyyymmdd = date.strftime("%Y%m%d")
    plt.title('RMA corellation %s'%yyyymmdd)#, fontsize=30)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%sregress%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def Test_Uncertainty(date=datetime(2005,1,1)):
    '''
        Effect on provided uncertainty with 8 day averaging
        Also calculate uncertainty as in DeSmedt2012
        First plot:
        Row1: 1 and 8 day uncertainty for 2005,1,1
        Row2: Timeseries of average uncertainty over australia (one and 8 day (and desmedt?))
        Second plot:
        Row1: 8 day unc vs DeSmedt calculated
        Row2: relative difference of top two,
    '''
    daystr=date.strftime("%Y %m %d")
    lllat=-50; lllon=110; urlat=-10; urlon=155
    # omi uncertainty: TODO: fix this in it's own function
    #plt.subplot(234)

    # Grab one day of reprocessed OMI data
    omrp1 = omrp(date,oneday=True)
    #sub1  = omrp1.inds_subset(lat0=lllat,lat1=urlat,lon0=lllon,lon1=urlon, maskocean=False, maskland=False)
    unc1  = omrp1.col_uncertainty_OMI
    omrp8 = omrp(date,oneday=False)
    #sub8  = omrp8.inds_subset(lat0=lllat,lat1=urlat,lon0=lllon,lon1=urlon, maskocean=False, maskland=False)
    count = omrp8.gridentries

    # divide avg uncertainty by 1 on sqrt count
    with np.errstate(divide='ignore', invalid='ignore'):
        unc8 = omrp8.col_uncertainty_OMI / np.sqrt(count)
        unc8[ ~ np.isfinite( unc8 )] = np.NaN

    #lats,lons = omrp1.latlon_bounds()
    lons, lats=np.meshgrid(omrp1.longitude,omrp1.latitude)

    # Plot1:
    #    Row1: 1 and 8 day uncertainty for 2005,1,1, and reduction map
    #    Row2: Timeseries of average uncertainty over australia (one and 8 day (and desmedt?))
    f = plt.figure(num=0,figsize=(16,12))
    ax=plt.subplot(231)
    m,cs,cb = pp.createmap(unc1,lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('One day')
    ax=plt.subplot(232)
    m,cs,cb = pp.createmap(unc8,lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('Eight days')
    ax=plt.subplot(233)
    m,cs,cb = pp.linearmap(count,lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon, vmin=1, vmax=24)
    cb.set_ticks([1,5,10,15,20,24])
    #cb.set_xticklabels()
    plt.title('Pixel count')
    ax=plt.subplot(212)
    plt.plot(np.arange(5),np.square(np.arange(5)-2.5))
    plt.title('Time series(TODO)')

    # save plot
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle('Uncertainty from %s'%daystr, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    outfig="Figs/Uncertainty.png"
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

    #Second plot:
    #    Row1: 8 day unc vs DeSmedt calculated
    #    Row2: relative difference of top two,
    # First calculate the DeSmedt method

def plot_swaths(region=pp.__AUSREGION__):
    '''
    plot swaths
    '''
    d0=datetime(2005,1,1)
    for dn in [datetime(2005,1,1),datetime(2005,1,8),datetime(2005,1,31)]:
        om=omrp(d0,dn)
        pname='Figs/swath_%s.png'%dn.strftime("%Y%m%d")
        cmargs={'pname':pname}
        om.plot_map(day0=d0,dayn=dn,**cmargs)

def look_at_swaths(date=datetime(2005,1,1), lllat=-80, lllon=-179, urlat=80, urlon=179, pltname=""):
    '''
        Plot one swath -> one day -> 8 days .
    '''

    # Grab OMI data
    omi_8=fio.read_omhcho_8days(date)
    omi_1=fio.read_omhcho_day(date)
    omi_s=fio.read_omhcho(fio.determine_filepath(date))

    counts= omi_8['counts']
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lonmids=omrp['longitude']
    latmids=omrp['latitude']
    lons,lats = np.meshgrid(lonmids,latmids)

    # Plot
    # SC, VC_omi, AMF_omi
    # VCC, VC_gc, AMF_GC
    # VC_OMI-GC, VC_GC-GC, GC_map
    # cuncs, AMF-correlation, AMF_GCz

    f, axes = plt.subplots(4,3,num=0,figsize=(16,20))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0,0])
    m,cs,cb = pp.createmap(omrp['SC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = pp.createmap(omrp['VC_OMI'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[0,2])
    m,cs,cb = pp.linearmap(omrp['AMF_OMI'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = pp.createmap(omrp['VCC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = pp.createmap(omrp['VC_GC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC GC')
    plt.sca(axes[1,2])
    m,cs,cb = pp.linearmap(omrp['AMF_GC'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GC}$')

    # Plot third row the GEOS-Chem map divergences

    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    OMIdiff=100*(omrp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omrp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[2,0])
    vmin,vmax = -150, 150
    m,cs,cb = pp.linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #m,cs,cb = pp.createmap(gc.VC_HCHO*1e-4, glats,glons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[2,1])
    m,cs,cb = pp.linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[2,2])
    m,cs,cb = pp.createmap(fineHCHO, lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')

    # plot fourth row: uncertainties and AMF_GCz
    #
    plt.sca(axes[3,0])
    m,cs,cb = pp.createmap(omrp['col_uncertainty_OMI'], lats, lons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('col uncertainty (VC$_{OMI} \pm 1 \sigma$)')
    plt.sca(axes[3,1])
    vc_gc,vc_omi=omrp['VC_GC'],omrp['VC_OMI']
    plt.scatter(vc_gc,vc_omi)
    plt.xlabel('VC$_{GC}$')
    plt.ylabel('VC$_{OMI}$')
    scatlims=[1e12,2e17]
    plt.yscale('log'); plt.ylim(scatlims)
    plt.xscale('log'); plt.xlim(scatlims)
    plt.plot(scatlims,scatlims,'k--',label='1-1') # plot the 1-1 line for comparison
    slp,intrcpt,r,CI1,CI2=RMA(vc_gc,vc_omi) # get regression
    plt.plot(scatlims, slp*np.array(scatlims)+intrcpt,color='red',
            label='slope=%4.2f, r=%4.2f'%(slp,r))
    plt.legend(title='lines',loc=0)
    plt.title("VC$_{GC}$ vs VC$_{OMI}$")
    plt.sca(axes[3,2])
    m,cs,cb = pp.linearmap(omrp['AMF_GCz'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GCz}$')

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%scorrected%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def Check_AMF_relevelling():
    '''
    Create a few example arrays and relevel the pressures and see what happens
    '''
    p0=np.array([1000,800,600,400,100,10])
    a0=np.array([10,9,8,7,6,5])
    p1=np.array([1100, 1050, 950, 500, 200, 80, 5])
    a1=np.array([9,8.5,8,7,6,5,4])
    p2=np.array([810, 650, 450, 250, 20, 1])
    a2=np.array([11,8,7,5,4,3])
    p3=np.array([1250, 1100, 900, 400, 100])
    a3=np.array([11,8,9,4,2.3])
    xlims=[2,12]
    ylims=[1300,.5]

    p00,p10,a00,a10=match_bottom_levels(p0.copy(),p1.copy(),a0.copy(),a1.copy())
    #print((p1,p10,a1,a10))
    assert p1[0]!=p10[0] and a1[0]!=a10[0], 'Err: No change in lowest level!'
    assert (a0 == a00).all() and (p0 == p00).all(), 'Err: Higher levels changed!'
    p1c=p1.copy() # make sure arguments aren't edited?
    p00,p10,a00,a10=match_bottom_levels(p0,p1c,a0,a1)
    assert (p1c==p1).all(), 'Err: Argument changed!'
    p11,p21,a11,a21=match_bottom_levels(p1,p2,a1,a2)
    p12,p32,a12,a32=match_bottom_levels(p1,p3,a1,a3)

    # look at 0vs1, 1vs2, 1vs3
    f,axes=plt.subplots(1,3,sharey=True,sharex=True,squeeze=True)

    # loop over comparisons
    for i, arrs in enumerate(((a0,p0,a1,p1,a10,p10),(a1,p1,a2,p2,a11,p11),(a1,p1,a3,p3,a32,p32))):
        aa,pa,ab,pb,an,pn = arrs
        plt.sca(axes[i])
        plt.plot(aa,pa, 'mo--',label='aa')
        plt.plot(ab,pb, 'b^--',label='ab')
        plt.plot(an,pn, 'r*',label='new ab')
        plt.plot(xlims,[pn[0],pn[0]],'k--', alpha=0.4) # horizontal line for examination

    plt.yscale('log')
    plt.ylim(ylims); plt.xlim([2,12])
    plt.legend(loc='best')

    plt.suptitle('Surface relevelling check')
    pname='Figs/SurfaceRelevelCheck.png'
    plt.savefig(pname)
    print(pname+' Saved')

def Check_OMI_AMF():
    '''
    See if the AMF_G is used to form the AMF in the OMI dataset
    '''
    # get one day of pixels
    om=fio.read_omhcho('omhcho/OMI-Aura_L2-OMHCHO_2005m0101t0020-o02472_v003-2014m0618t141821.he5',maxlat=60)
    #{'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
    #'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
    #'rad_ref_col':ref_c, 'RSC_OMI':rsc_omi,
    #'qualityflag':qf, 'xtrackflag':xqf, 'sza':sza,
    #'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}

    amf=om['AMF']
    goods=~np.isnan(amf)
    amf=np.array(amf[goods])
    s=np.array(om['apriori'][:,goods])
    w=np.array(om['omega'][:,goods])
    amfg=np.array(om['AMFG'][goods])
    p=np.array(om['plevels'][:,goods])

    dp=np.zeros([47,len(amf)])
    dp[-1,:]=p[-1,:]
    dp[0:46,:]=p[1:47,:]-p[0:46,:]
    dp[-1,:]=0.01 # top level pressure edge defined as 0.01hPa
    chris=np.sum(s*dp*w,axis=0) / np.sum(s*dp,axis=0)
    plt.scatter(chris,amf)
    plt.title('AMF calculation check on one OMI swath')
    plt.xlabel('Chris AMF')
    plt.ylabel('OMI AMF')
    plt.plot([0,3.5],[0,3.5],'--',label='1-1 line')
    plt.text(-0.5,3.4,"Chris AMF = $\Sigma_i (Shape(P_i) * \omega(P_i) * \Delta P_i) /  \Sigma_i (Shape(P_i) * \omega(P_i) )$")
    plt.text(-0.5,1.5,"mean AMF_G = %4.2f"%np.mean(amfg))
    plt.legend(loc=0)
    pname='Figs/Chris_AMF_Check.png'
    plt.savefig(pname)
    print('saved %s'%pname)

def Check_AMF():
    '''
    Check the various AMF's against eachother
    '''
    # get one day of pixels
    gps=reprocess.get_good_pixel_list(datetime(2005,1,1))
    amf=gps['AMF_OMI']
    amf_rm=gps['AMF_RM']
    amf_gc=gps['AMF_GC']
    lims=[0,4]
    plt.figure(0,figsize=[26,10])
    plt.subplot(131)
    pp.plot_corellation(amf,amf_gc,lims=lims,logscale=False)
    plt.xlabel('OMI AMF'); plt.ylabel('GC AMF')
    plt.subplot(132)
    pp.plot_corellation(amf,amf_rm,lims=lims,logscale=False)
    plt.xlabel('OMI AMF'); plt.ylabel('RM AMF')
    plt.subplot(133)
    pp.plot_corellation(amf_gc,amf_rm,lims=lims,logscale=False)
    plt.xlabel('GC AMF'); plt.ylabel('RM AMF')
    pname='Figs/Compare_AMFs.png'
    plt.savefig(pname)
    print('saved %s'%pname)

def Summary_RSC(date=datetime(2005,1,1), oneday=True):
    '''
    Print and plot a summary of the effect of our remote sector correction
    Plot 1: Reference Correction
        showing VCs before and after correction, with rectangle around region

    Plot 2: OMI Sensors difference from apriori over RSC
        Contourf of RSC correction function [sensor(X) vs latitude(Y)]
    '''

    ymdstr=date.strftime('%Y%m%d')
    # read reprocessed data
    dat=omrp(date,oneday=oneday)
    lats,lons=dat.latitude,dat.longitude
    # read geos chem data
    gcdat=fio.read_gchcho(date)
    gchcho=gcdat.VC_HCHO*1e-4 # molec/m2 -> molec/cm2
    gclats,gclons=gcdat.lats,gcdat.lons
    # plot 1) showing VCs before and after correction
    vmin,vmax=1e14,1e17
    f=plt.figure(0,figsize=(17,16))
    lims=(-60,30,45,160)
    lim2=(-65,-185,65,-115)
    for i,arr in enumerate([dat.VC_GC,dat.VCC]):
        #plt.subplot(221+i)
        plt.subplot2grid((2, 6), (0, 3*i), colspan=3)
        m,cs=pp.createmap(arr,lats,lons,colorbar=False,vmin=vmin,vmax=vmax,
                       lllat=lims[0],lllon=lims[1],urlat=lims[2],urlon=lims[3])
        plt.title([Ovc,Ovcc][i],fontsize=25)
        m.drawparallels([-40,0,40],labels=[1-i,0,0,0],linewidth=1.0)

    # print some stats of changes
    diffs=dat.VCC-dat.VC_GC
    print ("Mean difference VC - VCC:%7.5e "%np.nanmean(diffs))
    print ("%7.2f%%"%(np.nanmean(diffs)*100/np.nanmean(dat.VC_GC)))
    print ("std VC - VCC:%7.5e "%np.nanstd(diffs))

    # plot c) RSC by sensor and latitude
    plt.subplot2grid((2, 6), (1, 0), colspan=2)
    cp=plt.contourf(np.arange(1,60.1,1),dat.RSC_latitude,dat.RSC)
    plt.colorbar(cp)
    plt.xlabel('sensor'); plt.ylabel('latitude')
    plt.title('OMI corrections')
    plt.xlim([-1,61]);plt.ylim([-70,70])
    plt.yticks(np.arange(-60,61,15))
    # plt.imshow(dat.RSC, extent=(0,60,-65,65), interpolation='nearest', cmap=cm.jet, aspect="auto")

    # plot d,e) RSC effect
    for i,arr in enumerate([gchcho, dat.VCC]):
        plt.subplot2grid((2, 6), (1, 2*i+2), colspan=2)
        m,cs=pp.createmap(arr,[gclats,lats][i],[gclons,lons][i],colorbar=False,vmin=vmin,vmax=vmax,
                       lllat=lim2[0],lllon=lim2[1],urlat=lim2[2],urlon=lim2[3])
        plt.title([Ogc,Ovcc][i],fontsize=25)
        # rectangle around RSC
        #plot_rec(m,dat.RSC_region,color='k',linewidth=4)
        meridians=m.drawmeridians([-160,-140],labels=[0,0,0,1], linewidth=4.0, dashes=(None,None))
        m.drawparallels([-60,0,60],labels=[1,0,0,0],linewidth=0.0)
        for m in meridians:
            try:
                meridians[m][1][0].set_rotation(45)
            except:
                pass

    f.suptitle('Reference Sector Correction '+ymdstr,fontsize=30)
    # Add colourbar to the right
    f.tight_layout()
    f.subplots_adjust(top=0.95)
    f.subplots_adjust(right=0.84)
    cbar_ax = f.add_axes([0.87, 0.20, 0.04, 0.6])
    cb=f.colorbar(cs,cax=cbar_ax)
    cb.set_ticks(np.logspace(13,17,5))
    cb.set_label('molec/cm$^2$',fontsize=24)

    pltname='Figs/Summary_RSC_Effect%s_%s.png'%(['8d',''][oneday],ymdstr)
    f.savefig(pltname)
    print ('%s saved'%pltname)
    plt.close(f)


def Summary_Single_Profile():
    '''
    Get a single good pixel and show all the crap it needs to go through
    '''
    # plot left to right:
    # OMI Apriori, GC Apriori, Omega, AMF, AMF_new, VC_OMI, VCC
    day=datetime(2005,1,1)

    # Read OMHCHO data ( using reprocess get_good_pixels function )
    pixels=reprocess.get_good_pixel_list(day, getExtras=True)

    #N = len(pixels['lat']) # how many pixels do we have
    #i=random.sample(range(N),1)[0]
    #print(i)
    i=153375
    lat,lon=pixels['lat'][i],pixels['lon'][i]
    omega=pixels['omega'][:,i]
    AMF_G=pixels['AMF_G'][i]
    AMF_OMI=pixels['AMF_OMI'][i]
    w_pmids=pixels['omega_pmids'][:,i]
    apri=pixels['apriori'][:,i]
    SC=pixels['SC'][i]
    AMF_GC=pixels['AMF_GC'][i]
    VC_GC=SC/AMF_GC
    VC_OMI=SC/AMF_OMI

    # read gchcho
    #lat,lon=0,0
    gchcho = fio.read_gchcho(day)
    gcapri, gc_smids = gchcho.get_single_apriori(lat,lon)
    gc_pmids    = gchcho.get_single_pmid(lat,lon)

    # rerun the AMF calculation and plot the shampoo
    #innerplot='Figs/AMF_test/AMF_test_innerplot%d.png'%i
    AMFS,AMFZ = gchcho.calculate_AMF(omega, w_pmids, AMF_G, lat, lon)

    print( "AMF_s=", AMFS )
    print( "AMF_z=", AMFZ )
    print( "AMF_o=", AMF_OMI )

    # Also make a plot of the regression new vs old AMFs
    f=plt.figure(figsize=(9,12))

    plt.plot(apri,w_pmids,'-k_',linewidth=2,markersize=20)
    plt.xlabel('Molecules cm$^{-2}$')
    plt.ylabel('hPa')
    ax=plt.twiny()
    plt.plot(gcapri, gc_pmids,'-r_',linewidth=2,markersize=20)
    plt.plot(omega, w_pmids, '-',linewidth=2,color='fuchsia')
    plt.title('Old vs new Apriori (%4.1fN, %4.1fE)'%(lat,lon),y=1.06)
    ax.tick_params(axis='x', colors='red')
    plt.ylim([1015, .04])
    plt.yscale('log')

    # omhchorp final vert column
    om=omrp(day,oneday=True)
    omlat,omlon=om.latitude, om.longitude
    omlati,omloni= (np.abs(omlat-lat)).argmin(),(np.abs(omlon-lon)).argmin()
    vcc=om.VCC[omlati,omloni] # corrected averaged grid square value
    ta=plt.gca().transAxes
    fs=30

    plt.text(.3,.95, 'AMF$_{OMI}$=%5.2f'%AMF_OMI,transform=ta, fontsize=fs)
    plt.text(.3,.86, 'AMF$_{GC}$=%5.2f'%AMFS,transform=ta, fontsize=fs)
    plt.text(.3,.77, '$\Omega_{OMI}$=%4.2e'%VC_OMI,transform=ta, fontsize=fs)
    plt.text(.3,.68, '$\Omega_{GC}$=%4.2e'%VC_GC, transform=ta, fontsize=fs)
    plt.text(.3,.60, '$\Omega_{GCC}$=%4.2e'%vcc, transform=ta, fontsize=fs)
    plt.text(.3,.47, 'OMI a priori',color='k',transform=ta, fontsize=27)
    plt.text(.3,.41, 'GC a priori',color='r',transform=ta, fontsize=27)
    plt.text(.3,.35, '$Scattering Weights$',color='fuchsia',transform=ta, fontsize=27)

    fname='Figs/SummarySinglePixel.png'
    f.savefig(fname)
    print("Saved "+fname)
    plt.close(f)

def check_timeline():
    '''
    '''
    print("Check Timeline test to be implemented")

def test_reprocess_corrected(date=datetime(2005,1,1), oneday=True, lllat=-80, lllon=-179, urlat=80, urlon=179,pltname=""):
    '''
    Test a day or 8-day reprocessed HCHO map
    Plot VCs, both OMI and Reprocessed,
        as well as AMFs and comparison against GEOS-Chem model.
    '''

    # Grab one day of reprocessed OMI data
    omhchorp=fio.read_omhchorp(date,oneday=oneday)

    counts = omhchorp['gridentries']
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lonmids=omhchorp['longitude']
    latmids=omhchorp['latitude']
    lons,lats = np.meshgrid(lonmids,latmids)

    # Plot
    # SC, VC_omi, AMF_omi
    # VCC, VC_gc, AMF_GC
    # VC_OMI-GC, VC_GC-GC, GC_map
    # cuncs, AMF-correlation, AMF_GCz

    f, axes = plt.subplots(4,3,num=0,figsize=(16,20))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0,0])
    m,cs,cb = pp.createmap(omhchorp['SC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = pp.createmap(omhchorp['VC_OMI'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[0,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_OMI'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = pp.createmap(omhchorp['VCC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = pp.createmap(omhchorp['VC_GC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC GC')
    plt.sca(axes[1,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_GC'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GC}$')

    # Plot third row the GEOS-Chem map divergences

    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    OMIdiff=100*(omhchorp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omhchorp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[2,0])
    vmin,vmax = -150, 150
    m,cs,cb = pp.linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #m,cs,cb = pp.createmap(gc.VC_HCHO*1e-4, glats,glons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[2,1])
    m,cs,cb = pp.linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[2,2])
    m,cs,cb = pp.createmap(fineHCHO, lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')

    # plot fourth row: uncertainties and AMF_GCz
    #
    plt.sca(axes[3,0])
    m,cs,cb = pp.createmap(omhchorp['col_uncertainty_OMI'], lats, lons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('col uncertainty (VC$_{OMI} \pm 1 \sigma$)')
    plt.sca(axes[3,1])
    vc_gc,vc_omi=omhchorp['VC_GC'],omhchorp['VC_OMI']
    plt.scatter(vc_gc,vc_omi)
    plt.xlabel('VC$_{GC}$')
    plt.ylabel('VC$_{OMI}$')
    scatlims=[1e12,2e17]
    plt.yscale('log'); plt.ylim(scatlims)
    plt.xscale('log'); plt.xlim(scatlims)
    plt.plot(scatlims,scatlims,'k--',label='1-1') # plot the 1-1 line for comparison
    slp,intrcpt,r,CI1,CI2=RMA(vc_gc,vc_omi) # get regression
    plt.plot(scatlims, slp*np.array(scatlims)+intrcpt,color='red',
            label='slope=%4.2f, r=%4.2f'%(slp,r))
    plt.legend(title='lines',loc=0)
    plt.title("VC$_{GC}$ vs VC$_{OMI}$")
    plt.sca(axes[3,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_GCz'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GCz}$')

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%scorrected%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def analyse_VCC_pp(day=datetime(2005,3,1), oneday=False, ausonly=True):
    '''
    Look closely at AMFs over Australia, specifically over land
    and see how our values compare against the model and OMI swaths.and Paul Palmers code
    '''
    # useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,dayn=util.last_day(day))

    print("AMF mean   : %7.4f, std: %7.4f"%(np.nanmean(om.AMF_OMI),np.nanstd(om.AMF_OMI)))
    print("AMF_GC mean: %7.4f, std: %7.4f"%(np.nanmean(om.AMF_GC),np.nanstd(om.AMF_GC)))
    print("AMF_PP mean: %7.4f, std: %7.4f"%(np.nanmean(om.AMF_PP),np.nanstd(om.AMF_PP)))

    lats=om.latitude
    lons=om.longitude
    unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)

    if ausonly:
        # filter to just Australia rectangle [.,.,.,.]
        landinds=om.inds_aus(maskocean=True)
    else:
        landinds=om.inds_subset(maskocean=True)
    oceaninds=om.inds_subset(maskocean=False,maskland=True)

    # the datasets with nans and land or ocean masked
    OMP_l = [] # OMI, My, Paul palmer
    OMP_o = [] # OMI, My, Paul palmer
    OMP_str = ['OMI_RSC','VCC', 'VCC_PP']
    OMP_col = ['k','r','m']
    land_data=[]
    ocean_data=[]
    for arr in [om.VC_OMI_RSC, om.VCC, om.VCC_PP]:
        print(arr.shape)
        alsonans=np.isnan(arr)
        OMP_l.append(ma(arr, mask=~landinds))
        OMP_o.append(ma(arr,  mask=~oceaninds))
        land_data.append(arr[landinds* ~alsonans ])
        ocean_data.append(arr[oceaninds* ~alsonans])
    #ocean_data=np.transpose(ocean_data)
    #land_data=np.transpose(land_data)

    # Print the land and ocean averages for each product
    print("%s land averages (oceans are global):"%(['Global','Australian'][ausonly]))
    print("%25s   land,   ocean"%'')
    for arrl, arro, arrstr in zip(OMP_l,OMP_o,OMP_str):
        print("%21s: %5.3e,  %5.3e "%(arrstr, np.nanmean(arrl),np.nanmean(arro)))

    # Plot the histogram of VC entries land and sea
    f=plt.figure(figsize=(14,10))
    #land_data=np.transpose([OMP_l[ii][landinds] for ii in range(3)])
    #ocean_data=np.transpose([OMP_o[ii][oceaninds] for ii in range(3)])

    olabel=['ocean '+thing for thing in OMP_str]
    llabel=['land ' +thing for thing in OMP_str]
    print(np.shape(land_data),np.shape(land_data[0]))
    plt.hist(ocean_data, bins=np.logspace(13, 17, 20), color=['k','darkblue','lightblue'], label=olabel)
    plt.hist(land_data, bins=np.logspace(13, 17, 20), color=['grey','orange','yellow'], label=llabel)
    plt.xscale("log")
    plt.yscale('log',nonposy='clip') # logarithmic y scale handling zero
    plt.title('Vertical column distributions ($\Omega_{HCHO}$)',fontsize=26)
    plt.ylabel('frequency'); plt.xlabel('molec cm$^{-2}$')
    plt.legend(loc='center left')
    ta=plt.gca().transAxes
    plt.text(0.05,.95, 'land count=%d'%np.sum(landinds),transform=ta)
    plt.text(.05,.90, 'ocean count=%d'%np.sum(oceaninds),transform=ta)
    for ii in range (3):
        plt.text(.05,.85-0.05*ii, '%s mean(land)=%5.3e'%(OMP_str[ii],np.nanmean(OMP_l[ii])), transform=ta)
    ausstr=['','_AUS'][ausonly]
    timestr=['_month','_day'][oneday]
    pname='Figs/hist%s%s_%s.png'%(timestr,ausstr,ymdstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

    # show corellations for OMI_RSC- VCC
    f=plt.figure(figsize=(18,7))

    nans_l=np.isnan(OMP_l[0])
    nans_o=np.isnan(OMP_o[0])
    for ii in range(3):
        nans_l=nans_l+np.isnan(OMP_l[ii]) + OMP_l[ii].mask # all the nans we won't fit
        nans_o=nans_o+np.isnan(OMP_o[ii]) + OMP_o[ii].mask
    print( "nans_l")
    print(np.shape(nans_l))
    print("%d nans in land data "%np.sum(nans_l))
    for ii in range(3):
        plt.subplot(131+ii)
        # pp.plot_corellation()
        plt.scatter(OMP_o[ii], OMP_o[ii-1], color='blue', label="Ocean", alpha=0.4)
        plt.scatter(OMP_l[ii], OMP_l[ii-1], color='gold', label="Land")
        m,x0,r,ci1,ci2 = RMA(OMP_l[ii][~nans_l], OMP_l[ii-1][~nans_l])
        X=np.array([1e10,5e16])
        plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%.5f, r=%.5f'%(m,r))
        plt.plot( X, X, 'k--', label='1-1' )
        plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
        plt.ylabel(OMP_str[ii-1]); plt.xlabel(OMP_str[ii])

    # save plot
    pname='Figs/correlations%s%s_%s.png'%(timestr,ausstr,ymdstr)
    f.suptitle("Product comparison for %s"%ymdstr,fontsize=28)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def plot_VCC_rsc_gc_pp(d0=datetime(2005,3,1),dn=None,region=[-45, 99.5, -11, 160.0]):
    '''
        Plot columns with different amf bases
        also different fire filtering strengths
              |  VCC   |  VCC_gc   |  VCC_pp
        fire0 |
        fire1 |
        fire2 |
        fire4 |
        fire8 |
    '''

    # start by reading all the VCC stuff
    # useful strings
    ymdstr=d0.strftime('%Y%m%d')
    if dn is not None:
        ymdstr=ymdstr+'-%s'%dn.strftime('%Y%m%d')
    pname='Figs/VCC_fires_%s.png'%ymdstr
    pname2='Figs/VCC_entries_%s.png'%ymdstr
    vmin,vmax=4e15,9e15 # min,max for colourbar
    linear=True # linear colour scale?
    vmin2,vmax2=0,40

    # time stuff:
    start_time=timeit.default_timer()

    # read in omhchorp
    om=omrp(d0,dayn=dn, keylist=['VC_OMI_RSC','VCC','VCC_PP','gridentries','ppentries'])
    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to read omhchorp"%elapsed)

    #print(vars(om).keys()) # [3, 720, 1152] data arrays returned, along with lats/lons etc.
    oceanmask=om.oceanmask # lets just look at land squares

    # Regionmask is the area which isnt within the region subset
    regionmask=~om.inds_subset(lat0=region[0],lat1=region[2],lon0=region[1],lon1=region[3])

    # Plot rows,cols,size:
    f,axes=plt.subplots(5,3,figsize=[18,18])
    # second plot just for entries
    f2, axes2=plt.subplots(5,3,figsize=[18,18])

    # first line is maps of VCC, VC_GC, VCC_PP
    titles=["OMI VCC","S(z) updated","S(z)+$\omega$(z) updated"]

    for i,arr in enumerate([om.VC_OMI_RSC, om.VCC, om.VCC_PP]):
        # entries for normal or ppamf
        entries=np.copy([om.gridentries,om.ppentries][i==2])
        entries=entries.astype(np.float) # so I can Nan the ocean/non-aus areas

        if len(np.shape(arr))==3:
            arr=np.nanmean(arr,axis=0) # average over time
            entries=np.nansum(entries,axis=0) # how many entries
        arr[oceanmask]=np.NaN # nanify the ocean
        arr[regionmask]=np.NaN # nanify outside the region
        entries[oceanmask]=np.NaN
        entries[regionmask]=np.NaN

        plt.sca(axes[0,i]) # first row ith column
        m,cs,cb= pp.createmap(arr,om.lats,om.lons,
                              region=region,
                              linear=linear,vmin=vmin,vmax=vmax,
                              cmapname='rainbow',colorbar=False)
        plt.title(titles[i])

        # add a little thing showing entries and mean and max
        txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries),np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr), 'max = %.2e'%np.nanmax(arr)]
        for txt, yloc in zip(txt,[0.01,0.07,0.13]):
            plt.text(0.01, yloc, txt,
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='k', fontsize=10)

        # also plot entries
        plt.sca(axes2[0,i])
        pp.createmap(entries,om.lats,om.lons,region=region,
                     cmapname='jet', colorbar=False,
                     linear=True,vmin=vmin2,vmax=vmax2)
        plt.title(titles[i])

    # Now loop over the same plots after NaNing our different fire masks
    for j, N in enumerate([1,2,4,8]):

        # read in fire mask for 1,2,4,8 days prior masking
        # firemask is 3dimensional: [days,lats,lons]
        fstart=timeit.default_timer()
        firemask=om.make_fire_mask(d0,dN=dn,days_masked=N)
        felapsed = timeit.default_timer() - fstart
        print ("TIMEIT: Took %6.2f seconds to make_fire_mask(%d days)"%(felapsed,N))

        for i, arr in enumerate([om.VC_OMI_RSC, om.VCC, om.VCC_PP]):
            plt.sca(axes[j+1,i]) # jth row ith column

            # pixelcounts for normal or ppamf
            entries=np.copy([om.gridentries,om.ppentries][i==2]).astype(np.float)

            # Nanify fire squares
            arr[firemask]=np.NaN

            # masking to our specific region:
            if len(np.shape(arr))==3:
                for k in range(np.shape(arr)[0]):
                    entries[k][regionmask]=np.NaN

            firepix=int(np.nansum(entries[firemask])) # how many fire pixels
            entries[firemask]=0
            # flatten arrays for plotting and entries
            if len(np.shape(arr))==3:
                arr=np.nanmean(arr,axis=0) # average over time
                entries=np.nansum(entries,axis=0) # how many entries

            # Nanify ocean squares and area outside region
            arr[regionmask]=np.NaN
            arr[oceanmask]=np.NaN
            entries[regionmask]=np.NaN
            entries[oceanmask]=np.NaN

            plt.sca(axes[j+1,i]) # first row ith column
            pp.createmap(arr,om.lats,om.lons,region=region,
                         linear=linear,vmin=vmin,vmax=vmax,
                         cmapname='rainbow', colorbar=False)

            # add a little thing showing entries and mean and max
            txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries), np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr),
                 'max = %.2e'%np.nanmax(arr), 'firepix=%d'%firepix]
            for txt, yloc in zip(txt,[0.01,0.07,0.13,0.19]):
                plt.text(0.01, yloc, txt,
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=plt.gca().transAxes,
                     color='k', fontsize=10)

            # also plot entries
            plt.sca(axes2[j+1,i])
            m2,cs2,cb2= pp.createmap(entries,om.lats,om.lons,region=region,
                                     cmapname='jet',linear=True, colorbar=False,
                                     vmin=vmin2,vmax=vmax2)

    # Add row labels
    rows = ['%d days'%fdays for fdays in [0,1,2,4,8]]
    rows[0]='fire filter\n'+rows[0]
    for ax, ax2, row in zip(axes[:,0], axes2[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='small')

    ticks=[np.logspace(np.log10(vmin),np.log10(vmax),5),np.linspace(vmin,vmax,5)][linear]
    pp.add_colourbar(f,cs,ticks=ticks,label='molec/cm$^2$')
    pp.add_colourbar(f2,cs2,ticks=np.linspace(vmin2,vmax2,5),label='pixels')

    f.savefig(pname)
    plt.close(f)
    print("Saved ",pname)
    f2.savefig(pname2)
    plt.close(f2)
    print("Saved ",pname2)

    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to run plot_VCC_rsc_gc_pp()"%elapsed)

    #createmap(data, lats, lons, make_edges=False, GC_shift=True,
    #          vmin=None, vmax=None, latlon=True,
    #          region=__GLOBALREGION__, aus=False, linear=False,
    #          clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
    #          cbarorient='bottom',
    #          pname=None,title=None,suptitle=None, smoothed=False,
    #          cmapname=None, fillcontinents=None):

def CompareMaps(day=datetime(2005,1,1), oneday=False, ausonly=True, PP=False):
    '''
    '''
    #useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,oneday=oneday)

    lats=om.latitude
    lons=om.longitude
    unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)

    # the datasets with nans and land or ocean masked
    OMP = [om.VC_OMI_RSC, om.VCC] # OMI, My, Paul palmer
    OMP_str = ['OMI_RSC','VCC']
    if PP:
        OMP.append(om.VCC_PP)
        OMP_str.append('VCC_PP')

    # Plot the maps:
    #
    f=plt.figure(figsize=(16,13))
    lllat=-65; urlat=65; lllon=-175; urlon=175
    if ausonly:
        lllat=-50; urlat=-5; lllon=100; urlon=170

    for ii in range([2,3][PP]):
        arr=OMP[ii]
        # Maps
        spnum=[221,231][PP]
        plt.subplot(spnum+ii)
        m,cs,cb = pp.createmap(OMP[ii],mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
        plt.title(OMP_str[ii],fontsize=20)

        # Difference maps
        diff=(OMP[ii]-OMP[ii-1]) * 100.0 / OMP[ii-1]
        spnum=[223,234][PP]
        plt.subplot(spnum+ii)
        m,cs,cb = pp.linearmap(diff,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=-200,vmax=200)
        plt.title('(%s - %s)*100/%s'%(OMP_str[ii],OMP_str[ii-1],OMP_str[ii-1]),fontsize=20)
    plt.suptitle("Maps of HCHO on %s"%ymdstr)

    ausstr=['','_AUS'][ausonly]
    eightstr=['_8day',''][oneday]
    pname='Figs/maps%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)




def test_calculation_corellation(day=datetime(2005,1,1), oneday=False, aus_only=False):
    '''
    Look closely at AMFs over Australia, specifically over land
    and see how our values compare against the model and OMI swaths.
    '''
    # useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,oneday=oneday)
    VCC=om.VCC
    VCC_PP=om.VCC_PP
    VC_OMI_RSC=om.VC_OMI_RSC
    VC_OMI=om.VC_OMI

    lats=om.latitude
    lons=om.longitude
    unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)

    if aus_only:
        # filter to just Australia rectangle [.,.,.,.]
        landinds=om.inds_aus(maskocean=True)
    else:
        landinds=om.inds_subset(maskocean=True)
    oceaninds=om.inds_subset(maskocean=False,maskland=True)

    # the datasets with nans and land or ocean masked
    vcomi_l     = ma(VC_OMI,mask=~landinds)
    vcomic_l      = ma(VC_OMI_RSC,mask=~landinds)
    vcc_l       = ma(VCC,mask=~landinds)
    vcc_pp_l    = ma(VCC_PP, mask=~landinds)
    vcomi_o     = ma(VC_OMI,mask=~oceaninds)
    vcomic_o      = ma(VC_OMI_RSC,mask=~oceaninds)
    vcc_o       = ma(VCC,mask=~oceaninds)
    vcc_pp_o    = ma(VCC_PP, mask=~oceaninds)
    landunc     = ma(unc,mask=~landinds)

    # Print the land and ocean averages for each product
    print("%s land averages (oceans are global):"%(['Global','Australian'][aus_only]))
    print("%25s   land,   ocean"%'')
    for arrl,arro,arrstr in zip([vcomi_l, vcomic_l, vcc_l, vcc_pp_l],[vcomi_o,vcomic_o,vcc_o, vcc_pp_o],['OMI','OMI_RSC','OMI_GCC', 'VCC_PP']):
        print("%21s: %5.3e,  %5.3e "%(arrstr, np.nanmean(arrl),np.nanmean(arro)))

    f=plt.figure(figsize=(14,10))
    # Plot the histogram of VC entries land and sea
    land_data=np.transpose([VC_OMI_RSC[landinds],VCC[landinds]])
    ocean_data=np.transpose([VC_OMI_RSC[oceaninds],VCC[oceaninds]])
    olabel=['ocean '+thing for thing in [Oomic,Ovcc]]
    llabel=['land ' +thing for thing in [Oomic,Ovcc]]
    plt.hist(ocean_data, bins=np.logspace(13, 17, 20), color=['darkblue','lightblue'], label=olabel)
    plt.hist(land_data, bins=np.logspace(13, 17, 20), color=['orange','yellow'], label=llabel)
    plt.xscale("log")
    plt.yscale('log',nonposy='clip') # logarithmic y scale handling zero
    plt.title('Vertical column distributions ($\Omega_{HCHO}$)',fontsize=26)
    plt.ylabel('frequency'); plt.xlabel('molec cm$^{-2}$')
    plt.legend(loc='center left')
    ta=plt.gca().transAxes
    plt.text(0.05,.95, 'land count=%d'%np.sum(landinds),transform=ta)
    plt.text(.05,.90, 'ocean count=%d'%np.sum(oceaninds),transform=ta)
    plt.text(.05,.86, '%s mean(land)=%5.3e'%(Ovcc,np.nanmean(vcc_l)),transform=ta)
    plt.text(.05,.82, '%s mean(land)=%5.3e'%(Oomic,np.nanmean(vcomic_l)),transform=ta)
    ausstr=['','_AUS'][aus_only]
    eightstr=['_8day',''][oneday]
    pname='Figs/land_VC_hist%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

    # Plot the maps:
    #

    f=plt.figure(figsize=(16,13))
    plt.subplot(231)
    lllat=-65; urlat=65; lllon=-175; urlon=175
    if aus_only:
        lllat=-50; urlat=-5; lllon=100; urlon=170

    # OMI_RSC map
    m,cs,cb = pp.createmap(vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title(Oomic,fontsize=20)

    # VCC map
    plt.subplot(232)
    m,cs,cb = pp.createmap(vcc_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title(Ovcc,fontsize=20)

    # (VCC- OMI_RSC)/OMI_RSC*100 map
    plt.subplot(233)
    m,cs,cb = pp.linearmap((vcc_l-vcomic_l)*100/vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=-200,vmax=200)
    plt.title('(%s - %s)*100/%s'%(Ovcc,Oomic,Oomic),fontsize=20)

    # show corellations for OMI_RSC- VCC
    nans=np.isnan(vcc_l) + np.isnan(vcomic_l) + vcomic_l.mask # all the nans we won't fit
    plt.subplot(223)
    plt.scatter(vcomic_o, vcc_o, color='blue', label="Ocean", alpha=0.4)
    plt.scatter(vcomic_l, vcc_l, color='gold', label="Land")
    m,x0,r,ci1,ci2 = RMA(VC_OMI_RSC[~nans], VCC[~nans])
    X=np.array([1e10,5e16])
    plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%.5f, r=%.5f'%(m,r))
    plt.plot( X, X, 'k--', label='1-1' )
    plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
    plt.ylabel(Ovcc); plt.xlabel(Oomic)

    # show corellations for OMI - VCC
    plt.subplot(224)
    nans=np.isnan(vcc_l) + np.isnan(vcomi_l) + vcc_l.mask # all the nans we won't fit
    plt.scatter(vcomi_o, vcc_o, color='blue', label="Ocean", alpha=0.4)
    plt.scatter(vcomi_l, vcc_l, color='gold', label="Land")
    m,x0,r,ci1,ci2 = RMA(vcomi_l[~nans],vcc_l[~nans])
    X=np.array([1e10,5e16])
    plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%4.2f, r=%.2f'%(m,r))
    plt.plot( X, X, 'k--', label='1-1' )
    plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
    plt.ylabel(Ovcc); plt.xlabel(Oomi)

    # save plot
    pname='Figs/correlations%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    f.suptitle("Product comparison for %s"%ymdstr,fontsize=28)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def test_amf_calculation(scount=50):
    '''
    Grab input argument(=50) columns and the apriori and omega values and check the AMF calculation on them
    Plot old, new AMF and integrand at each column
    Also check the AMF calculated using non sigma normalised shape factor
    '''
    day=datetime(2005,1,1)
    # Read OMHCHO data ( using reprocess get_good_pixels function )
    pixels=reprocess.get_good_pixel_list(day, getExtras=True)
    N = len(pixels['lat']) # how many pixels do we have

    # read gchcho
    gchcho = fio.read_gchcho(day)

    # check the random columns
    AMF_old=[]
    AMF_z, AMF_s=[],[]

    # find 50 GOOD columns(non nans)
    for i, jj in enumerate(random.sample(range(N), scount)):
        lat,lon=pixels['lat'][jj],pixels['lon'][jj]
        omega=pixels['omega'][:,jj]
        AMF_G=pixels['AMF_G'][jj]
        AMF_old.append(pixels['AMF_OMI'][jj])
        w_pmids=pixels['omega_pmids'][:,jj]
        # rerun the AMF calculation and plot the shampoo
        innerplot='Figs/AMF_test/AMF_test_innerplot%d.png'%i
        AMFS,AMFZ = gchcho.calculate_AMF(omega, w_pmids, AMF_G, lat, lon, plotname=innerplot,debug_levels=True)
        AMF_s.append(AMFS)
        AMF_z.append(AMFZ)
    #print( "AMF_s=", AMF_s[:] )
    #print( "AMF_z=", AMF_z[:] )
    #print( "AMF_o=", AMF_old[:] )

    # Also make a plot of the regression new vs old AMFs
    f=plt.figure(figsize=(12,12))
    amfs=np.array(pixels['AMF_GC'])
    amfo=np.array(pixels['AMF_OMI'])
    plt.scatter(amfs, amfo, color='k', label='pixel AMFs')
    # line of best fit
    isnans= np.isnan(amfs) + np.isnan(amfo)
    slp,intrcpt,r,p,sterr=stats.linregress(amfs[~isnans],amfo[~isnans])
    # straight line on log-log plot is 10^(slope*log(X) + Yintercept)
    Xarr=np.array([1,6])
    plt.plot(Xarr,slp*Xarr+intrcpt, color='red', label='slope=%.5f, r=%.5f'%(slp,r))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('Figs/AMF_test/AMF_test_corr.png')
    plt.close(f)

    # Post regridding, check the AMFs within 50 degrees of the equator, and color by land/sea
    # Achieve this using maskoceans
    #
    f=plt.figure(figsize=(10,10))
    omhchorp=fio.read_omhchorp(day,oneday=True, keylist=['AMF_GC','AMF_OMI','latitude','longitude'])
    lats=omhchorp['latitude']
    lons=omhchorp['longitude']
    mlons,mlats=np.meshgrid(lons,lats)  # [lats x lons] ?
    amf=omhchorp['AMF_GC']      # [lats x lons]
    amfo=omhchorp['AMF_OMI']    # [lats x lons]
    ocean=maskoceans(mlons,mlats,amf,inlands=False).mask
    landamfo=amfo.copy()
    landmlats=mlats.copy()
    landmlons=mlons.copy()
    landamf=amf.copy()
    oceanamfo=amfo.copy()
    oceanmlats=mlats.copy()
    oceanmlons=mlons.copy()
    oceanamf=amf.copy()
    for arrr in [landamf, landamfo, landmlats, landmlons]:
        arrr[ocean]=np.NaN
    for arrr in [oceanamf, oceanamfo, oceanmlats, oceanmlons]:
        arrr[~ocean]=np.NaN
    m,cs,cb=pp.linearmap(landmlats,landmlons,landamfo)
    f.savefig('Figs/oceancheck.png')
    plt.close(f)
    # Check slopes and regressions of ocean/non ocean AMFs
    f=plt.figure(figsize=(14,14))
    plt.scatter(amf[ocean], amfo[ocean], color='cyan', alpha=0.5)
    plt.scatter(amf[~ocean],amfo[~ocean], color='fuchsia', alpha=0.5)
    slopeo,intercepto,ro,p,sterr = stats.linregress(amf[ocean], amfo[ocean])
    slopel,interceptl,rl,p,sterr = stats.linregress(amf[~ocean],amfo[~ocean])
    plt.plot(Xarr, slopel*Xarr+interceptl,color='fuchsia',
            label='Land: slope=%.5f, r=%.5f'%(slopel,rl))
    plt.plot(Xarr, slopeo*Xarr+intercepto, color='cyan',
            label='Ocean: slope=%.5f, r=%.5f'%(slopeo,ro))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('Figs/AMF_test/AMF_test_corr_masked.png')

    #amfland=maskoceans(mlons,mlats,amf,inlands=False)
    #amfoland=maskoceans(mlons,mlats,amfo,inlands=False)


def compare_cloudy_map():
    '''
    Check the result of running the 8 day reprocess after filtering cloudy grid
        cells.
    '''
    date=datetime(2005,1,1,0)
    # read the dataset WITHOUT cloud filtering:
    data=fio.read_omhchorp(date)
    counts = data['gridentries']
    lonmids=data['longitude']
    latmids=data['latitude']
    # read the dataset WITH cloud filtering
    # ( at that point I stored regridded vc seperately )
    cloudyfile="omhchorp/cloudy_omhcho_8p0.25x0.31_20050101.he5"
    cloudydata=fio.read_omhchorp(date,filename=cloudyfile)
    cloudycounts=cloudydata['GridEntries']

    # look at differences before and after cloud filter at 0.4 threshhold
    print("%4e entries before cloud filter, %4e entries after cloud filter"%(np.nansum(cloudycounts),np.nansum(counts)))
    print("Cloud filtered stats:")
    print(" MIN,\tMEAN,\tMAX ")
    print("OLD AMF: ")
    mmm(data['AMF_OMI'])
    print("NEW AMF: ")
    mmm(data['AMF_GC'])
    print("OLD_VC:  ")
    mmm(data['VC_OMI'] )
    print("NEW_VC:  ")
    mmm(data['VC_GC'] )
    print("Unfiltered stats:")
    print("OLD AMF: ")
    mmm(cloudydata['AMF_OMI'])
    print("NEW AMF: ")
    mmm(cloudydata['AMF_GC'])
    print("OLD_VC:  ")
    mmm(cloudydata['VC_OMI'] )
    print("NEW_VC:  ")
    mmm(cloudydata['VC_GC'] )

    # Plot OMI, oldrp, newrp VC map
    f, axes = plt.subplots(2,3,num=0,figsize=(16,14))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    #plt.sca(axes[0,0])
    i=0
    for d in [cloudydata, data]:
        plt.sca(axes[i,0])
        m,cs,cb = pp.createmap(d['VC_OMI'],latmids,lonmids)
        plt.title('VC Regridded')
        plt.sca(axes[i,1])
        m,cs,cb = pp.createmap(d['VC_GC'],latmids,lonmids)
        plt.title('VC Reprocessed')
        plt.sca(axes[i,2])
        m,cs,cb = pp.linearmap(d['AMF_GC'],latmids,lonmids,vmin=1.0,vmax=6.0)
        plt.title('AMF reprocessed')
        i=1

    # save plots
    plt.tight_layout()
    plt.savefig("Figs/cloud_filter_effects.png")
    plt.close()

def test_fires_fio():
    '''
    Test 8 day average fire interpolation and fio
    '''
    day = datetime(2005,1,1)
    ## get normal and interpolated fire
    orig, lats, lons=fio.read_8dayfire(day)

    lonres,latres = 0.3125, 0.25
    regr, lats2, lons2=fio.read_8dayfire_interpolated(day, latres=latres, lonres=lonres)

    check_array(orig)
    check_array(regr)

    # Number checks..
    assert np.max(orig) == np.max(regr), "Maximum values don't match..."
    print ("Mean orig = %4.2f\nMean regridded = %4.2f"%(np.mean(orig[orig>-1]), np.mean(regr[regr>-1])))

    ## PLOTTING
    ##

    # EG plot of grids...
    plt.figure(0,figsize=(10,8))
    # tasmania lat lon = -42, 153
    m0=Basemap(llcrnrlat=-45, urcrnrlat=-37, llcrnrlon=142, urcrnrlon=150,
              resolution='i',projection='merc')
    m0.drawcoastlines()
    m0.fillcontinents(color='coral',lake_color='aqua')
    m0.drawmapboundary(fill_color='white')

    # original grid = thick black
    d=[1000,1]
    m0.drawparallels(lats-0.25, linewidth=2.5, dashes=d)
    m0.drawmeridians(lons-0.25, linewidth=2.5, dashes=d)
    # new grid = thin brown
    m0.drawparallels(lats2-latres/2.0, color='brown', dashes=d)
    m0.drawmeridians(lons2-lonres/2.0, color='brown', dashes=d)
    plt.title('Original grid(black) vs new grid(red)')
    plt.savefig('Figs/AQUAgrids.png')
    plt.close()
    ## Regridded Data plot comparison
    # plot on two subplots
    fig=plt.figure(1, figsize=(14,9))
    fig.patch.set_facecolor('white')

    # mesh lon/lats
    mlons,mlats = np.meshgrid(lons,lats)
    mlons2,mlats2 = np.meshgrid(lons2,lats2)
    axtitles=['original (0.5x0.5)',
              'regridded to %1.4fx%1.4f (latxlon)' % (latres,lonres)]

    ax1=plt.subplot(121)
    m1,cs1, cb1 = pp.linearmap(orig,mlats,mlons, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)
    ax2=plt.subplot(122)
    m2,cs2, cb2 = pp.linearmap(regr,mlats2,mlons2, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)

    for i in range(2):
        [ax1,ax2][i].set_title(axtitles[i])
        [cb1, cb2][i].set_label('Fire Count (8 day)')

    plt.suptitle('AQUA 2005001',fontsize=20)
    plt.tight_layout()
    plt.savefig('Figs/AQUA2005001.png')
    plt.close()

def test_fires_removed(day=datetime(2005,1,25),oneday=False):
    '''
    Check that fire affected pixels are actually removed
    '''
    # Read one or 8 day average:
    #
    omhchorp= omrp(day, oneday=oneday)
    pre     = omhchorp.VCC
    count   = omhchorp.gridentries
    lats,lons=omhchorp.latitude,omhchorp.longitude
    pre_n   = np.nansum(count)
    ymdstr=day.strftime(" %Y%m%d")
    # apply fire masks
    #
    fire8           = omhchorp.fire_mask_8 == 1
    fire16          = omhchorp.fire_mask_16 == 1
    post8           = dcopy(pre)
    post8[fire8]    = np.NaN
    post16          = dcopy(pre)
    post16[fire16]  = np.NaN
    post8_n         = np.nansum(count[~fire8])
    post16_n        = np.nansum(count[~fire16])

    # print the sums
    print("%1e entries, %1e after 8day fire removal, %1e after 16 day fire removal"%(pre_n,post8_n,post16_n))

    # compare and beware
    #
    f = plt.figure(num=0,figsize=(16,6))
    # Plot pre, post8, post16
    # Fires Counts?

    vmin,vmax=1e14,1e17
    Ovcc='$\Omega_{VCC}$'
    titles= [ Ovcc+s for s in ['',' - 8 days of fires', ' - 16 days of fires'] ]
    for i,arr in enumerate([pre,post8,post16]):
        plt.subplot(131+i)
        m,cs = pp.ausmap(arr,lats,lons,vmin=vmin,vmax=vmax,colorbar=False)
        plt.title(titles[i])
    pname='Figs/fire_exclusion_%s.png'%['8d','1d'][oneday]
    plt.suptitle("Effects of Fire masks"+ymdstr,fontsize=28)
    plt.tight_layout()
    #plt.subplots_adjust(top=0.92)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def test_gchcho():
    '''
    Function tests gchcho output file created by create_column.pro
    Saves a plot of HCHO density,
    apriori sigma shape before/after interpolation,
    and total column hcho map
    '''
    ## Files to be read based on date
    day0 = datetime(2005,1,1)
    ymstr= day0.strftime('%Y %b')
    ymdstr=day0.strftime('%Y%m%d')
    print("Testing GCHCHO stuff on "+ymstr)

    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)

    plt.figure(figsize=(14,12))
    m,cs,cb=gchcho.PlotVC()
    plt.savefig('Figs/GCHCHO_Vertical_Columns%s.png'%ymdstr)
    plt.clf()

    plt.figure(figsize=(12,12))
    gchcho.PlotProfile()
    plt.savefig('Figs/GCHCHO_EGProfile%s.png'%ymdstr)
    plt.clf()

def test_hchorp_apriori():
    '''
    Check the apriori saved into reprocessed file looks as it should.
    (Compares omhchorp against gchcho, and omhchorg)
    Currently looks ok(11/4/16 monday)
    UPDATE: 20160906
        omhchorp no longer contains the aprioris and omegas - testing that stuff is done in amf tests.
        now this test just looks at the gchcho vs omhcho aprioris
    '''
    ## Files to be read based on date
    day0 = datetime(2005,1,1)
    ymstr= day0.strftime('%Y %b')
    #yyyymm= day0.strftime('%Y%m')
    print("Testing GCHCHO apriori stuff on "+ymstr)

    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)

    # Read omhcho day of swaths:
    omhcho = fio.read_omhcho_day(day0) # [ plevs, ~24k rows, 60 cols ]
    omishape = omhcho['apriori']
    omilats, omilons = omhcho['lats'],omhcho['lons']
    omiplevs=omhcho['plevels']

    # Sample of normalized columns
    scount=15 # expect around 7 columns which aren't on nan pixels
    ii=random.sample(range(len(omilats[:,0])), scount)
    jj=random.sample(range(len(omilats[0,:])), scount)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,9))
    for i in range(scount):
        # indices of random lat/lon pair
        rowi,coli=ii[i],jj[i]
        lat=omilats[rowi,coli]
        if np.isnan(lat):
            continue
        lon=omilons[rowi,coli]
        omiS=omishape[:,rowi,coli]
        omiP=omiplevs[:,rowi,coli]
        ax1.plot(omiS,omiP, label="%-7.2f,%-7.2f"%(lat,lon))
        gcShape,gcSig=gchcho.get_single_apriori(lat,lon,z=False)
        ax2.plot(gcShape,gcSig)
    ax1.set_xlabel('Molecules/cm2')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title('OMI apriori')
    ax1.set_ylim([1050, 0.05])
    ax1.set_yscale('log')
    ax2.set_ylim([1.04,-0.04])
    ax2.set_ylabel('Sigma')
    ax2.set_title('GEOS-Chem S$_\sigma$')
    ax2.set_xlabel('unitless')
    ax1.legend(title='    lat,  lon ',loc=0)
    pltname='Figs/Shape_Factor_Examples.png'
    plt.savefig(pltname); print("%s saved!"%pltname)

def check_high_amfs(day=datetime(2005,1,1)):
    '''
    Get good pixels, look at AMFs over 10, examine other properties of pixel
    '''
    print("Running check_high_amfs")

    start_time=timeit.default_timer()
    pix=reprocess.get_good_pixel_list(day, getExtras=True)
    elapsed = timeit.default_timer() - start_time
    print ("Took %6.2f seconds to read %d entries"%(elapsed, len(pix['AMF_OMI'])))

    cloud=np.array(pix['cloudfrac'])    # [~1e6]
    lat=np.array(pix['lat'])            #
    # inds2 = high amf, cloud < 0.4
    inds2= (np.array(pix['AMF_GC']) > 10.0) * (cloud < 0.4)
    # inds1 = non high amf, cloud < 0.4
    inds1= (~inds2) * (cloud < 0.4)
    omegas=pix['omega']         # [47, ~1e6]
    aprioris=pix['apriori']     #
    sigmas=pix['sigma']         #

    assert np.sum(np.abs(pix['qualityflag']))==0, 'qualityflags non zero!'
    assert np.sum(np.abs(pix['xtrackflag']))==0, 'xtrackflags non zero!'

    # normal AMFS
    omegas1=omegas[:,inds1]
    n1=len(omegas1[0,:])
    sigmas1=pix['sigma'][:,inds1]
    # cloud seems fine, check relation to latitude
    cloud1=cloud[inds1]
    lat1=lat[inds1]
    # high AMFS
    omegas2=omegas[:,inds2]
    n2=len(omegas2[0,:])
    sigmas2=pix['sigma'][:,inds2]
    cloud2=cloud[inds2]
    lat2=lat[inds2]
    f = plt.figure(figsize=(15,9))
    a1 = f.add_subplot(1, 3, 1)
    a2 = f.add_subplot(1, 3, 2, sharex = a1, sharey=a1)
    a3 = f.add_subplot(2, 3, 3)
    a4 = f.add_subplot(2, 3, 6, sharex = a3)
    plt.sca(a1)
    # plot random set of 100 normal omegas
    for i in random.sample(range(n1),100):
        plt.plot(omegas1[:,i],sigmas1[:,i],alpha=0.4,color='orange')
    # plot mean omega
    X,Y = np.mean(omegas1, axis=1), np.mean(sigmas1,axis=1)
    Xl,Xr = X-2*np.std(omegas1,axis=1),X+2*np.std(omegas1,axis=1)
    plt.plot(X,Y, linewidth=3, color='k')
    plt.fill_betweenx(Y, Xl, Xr, alpha=0.5, color='cyan')
    plt.text(4.75,.9, 'mean cloud =%6.2f%%'%(np.mean(cloud1)*100.))
    plt.text(4.75,.86, 'count = %d'%n1)
    plt.text(6, 0.12, 'Mean profile',fontweight='bold')
    plt.text(6, 0.16, '$\pm$2 std devs', color='cyan')
    plt.text(6, 0.20, '100 random $\omega$ profiles', color='orange')
    plt.title('AMF <= 10.0')
    plt.ylabel('Altitude ($\sigma$)')
    plt.sca(a2)
    #plot 100 highamf omegas
    for i in random.sample(range(n2),100):
        plt.plot(omegas2[:,i],sigmas2[:,i],alpha=0.4,color='orange')
    # plot mean omega
    X,Y = np.mean(omegas2, axis=1), np.mean(sigmas2,axis=1)
    Xl,Xr = X-2*np.std(omegas2,axis=1),X+2*np.std(omegas2,axis=1)
    plt.plot(X,Y, linewidth=3, color='k')
    plt.fill_betweenx(Y, Xl, Xr, alpha=0.5, color='cyan')
    plt.text(5.75,.9, 'mean cloud =%6.2f%%'%(np.mean(cloud2)*100.))
    plt.text(5.75,.86, 'count = %d'%n2)
    plt.title('AMF > 10.0')
    plt.ylim([1.05,-0.05])
    plt.xlim([-0.1, np.max(omegas2)])
    # plot cloud frac histogram
    plt.sca(a3)
    plt.hist(lat1)
    plt.title('histogram: AMF < 10')
    plt.sca(a4)
    plt.hist(lat2, bins=np.arange(-90,91,5))
    plt.title('histogram: AMF > 10')
    plt.setp(a3, xticks=np.arange(-90,91,30)) # set xticks for histograms
    plt.xlabel('Latitude')
    plt.suptitle('Scattering weights ($\omega$)')
    outpic='Figs/high_AMF_omegas.png'
    plt.savefig(outpic)
    print('saved %s'%outpic)

def check_RSC(day=datetime(2005,1,1), track_corrections=False):
    '''
    Grab the RSC from both GEOS-Chem and OMI for a particular day
    Plot and compare the RSC region
    Plot the calculated corrections
    '''
    print("Running check_RSC")
    # Read in one day average
    omhchorp1=fio.read_omhchorp(day,oneday=True)
    yyyymmdd=day.strftime("%Y%m%d")
    rsc=omhchorp1['RSC']
    ref_lat_bins=omhchorp1['RSC_latitude']
    rsc_gc=omhchorp1['RSC_GC'] # RSC_GC is in molecs/cm2 as of 11/08/2016

    ## plot each track with a slightly different colour
    #
    if track_corrections:
        f1=plt.figure(0,figsize=(8,10))
        colors=[plt.cm.jet(i) for i in np.linspace(0, 1, 60)]
        for track in range(60):
            plt.plot(rsc[:,track], ref_lat_bins, '.', color=colors[track])
            #plt.plot(rsc_function(ref_lat_bins,track), ref_lat_bins, color=colors[track])
        plt.ylabel('latitude')
        plt.xlabel('molecules/cm2')
        plt.title('Reference sector correction for %s'%yyyymmdd)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=plt.Normalize(vmin=0,vmax=60))
        sm._A=[]
        cb=plt.colorbar(sm)
        cb.set_label('track')
        outfig1='Figs/track_corrections%s.png'%yyyymmdd
        plt.savefig(outfig1)
        print(outfig1+' saved')
        plt.close(f1)

    def RSC_map(lons,lats,data,labels=[0,0,0,0]):
        ''' Draw standard map around RSC region '''
        vmin, vmax= 5e14, 1e16
        lat0, lat1, lon0, lon1=-75, 75, -170, -130
        m=Basemap(lon0,lat0,lon1,lat1, resolution='i', projection='merc')
        cs=m.pcolormesh(lons, lats, data, latlon=True,
              vmin=vmin,vmax=vmax,norm = LogNorm(), clim=(vmin,vmax))
        cs.cmap.set_under('white')
        cs.cmap.set_over('pink')
        cs.set_clim(vmin,vmax)
        m.drawcoastlines()
        m.drawmeridians([ -160, -140],labels=labels)
        return m, cs

    ## plot the reference region,
    #
    # we need 501x3 bounding edges for our 500x2 2D data (which is just the 500x1 data twice)
    rsc_lat_edges= list(ref_lat_bins-0.18)
    rsc_lat_edges.append(90.)
    rsc_lat_edges = np.array(rsc_lat_edges)
    lons_gc, lats_gc = np.meshgrid( [-160., -150, -140.], rsc_lat_edges )

    # Make the GC map:
    f2=plt.figure(1, figsize=(14,10))
    plt.subplot(141)
    rsc_gc_new = np.transpose(np.array([ rsc_gc, rsc_gc] ))
    m,cs = RSC_map(lons_gc,lats_gc,rsc_gc_new, labels=[0,0,0,1])
    m.drawparallels([-45,0,45],labels=[1,0,0,0])
    cb=m.colorbar(cs,"right",size="8%", pad="3%")
    cb.set_label('Molecules/cm2')
    plt.title('RSC_GC')

    # add the OMI reference sector for comparison
    sc_omi=omhchorp1['SC'] # [ 720, 1152 ] molecs/cm2
    lats_rp=omhchorp1['latitude']
    lons_rp=omhchorp1['longitude']
    rsc_lons= (lons_rp > -160) * (lons_rp < -140)
    newlons=pp.regularbounds(lons_rp[rsc_lons])
    newlats=pp.regularbounds(lats_rp)
    newlons,newlats = np.meshgrid(newlons,newlats)
    # new map with OMI SC data
    plt.subplot(142)
    plt.title("SC_OMI")
    m,cs=RSC_map(newlons, newlats, sc_omi[:,rsc_lons])
    m.drawmeridians([ -160, -140],labels=[0,0,0,0])

    ## Another plot using OMI_VC (old reprocessed data)
    #
    vc_omi=omhchorp1['VC_OMI']
    plt.subplot(143)
    plt.title('VC_OMI')
    m,cs=RSC_map(newlons,newlats,vc_omi[:,rsc_lons])

    ## One more with VC_GC over the ref sector
    #
    vc_gc=omhchorp1['VC_GC'] # [ 720, 1152 ] molecs/cm2
    # new map with OMI SC data
    plt.subplot(144)
    plt.title("VC_GC")
    m,cs=RSC_map(newlons,newlats,vc_gc[:,rsc_lons])


    f2.suptitle('GEOS_Chem VC vs OMI SC over RSC on %s'%yyyymmdd)
    outfig2='Figs/RSC_GC_%s.png'%yyyymmdd
    plt.savefig(outfig2)
    print(outfig2+' saved')
    plt.close(f2)

def check_flags_and_entries(day=datetime(2005,1,1), oneday=True):
    '''
    Count entries and how many entries were filtered out
    Plot histograms of some fields for one or 8 day averaged omhchorp data
    '''

    print("Running check_flags_and_entries")
    # Read in one day average
    omhchorp=fio.read_omhchorp(day,oneday=oneday)
    yyyymmdd=day.strftime("%Y%m%d")
    rsc=omhchorp['RSC']
    rsc_gc=omhchorp['RSC_GC'] # RSC_GC is in molecs/cm2 as of 11/08/2016
    rsc_region=omhchorp['RSC_region']
    print(("RSC region: ", rsc_region))
    VC_GC=omhchorp['VC_GC']
    VC_OMI=omhchorp['VC_OMI']
    VCC=omhchorp['VCC']
    entries=omhchorp['gridentries']

    # show histograms for positive and negative part of data
    # pass in two axes
    def show_distribution(axes, bins, data, log=True):
        notnans=~np.isnan(data)
        fdata=data[notnans]
        plus=fdata[fdata>0]
        minus=-1*fdata[fdata<0]
        plt.sca(axes[0])
        plt.hist(minus,bins=bins)
        if log: plt.xscale("log")
        axes[0].invert_xaxis()
        plt.sca(axes[1])
        plt.hist(plus,bins=bins)
        if log: plt.xscale("log")

    f, axes = plt.subplots(2,2)
    # Logarithmic space from 1e12 - 1e19, 50 steps
    logbins=np.logspace(12,19,50)

    show_distribution(axes[0,:], logbins, VC_GC)
    axes[0,0].set_title("Negative GC_VC_HCHO")
    axes[0,1].set_title('Positive GC_VC_HCHO')

    show_distribution(axes[1,:], logbins, VC_OMI)
    axes[1,0].set_title("Negative OMI_VC_HCHO")
    axes[1,1].set_title('Positive OMI_VC_HCHO')

    plt.suptitle("VC distributions for %d day average on %s"%([8,1][oneday],yyyymmdd), fontsize=25)
    plt.savefig("Figs/distributions%d_%s"%([8,1][oneday], yyyymmdd))
    plt.close()

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running tests.py")
    pp.InitMatplotlib()
    #Summary_Single_Profile()

    # GEOS Chem trop vs ucx restarts
    #check_HEMCO_restarts()

    #analyse_VCC_pp(oneday=False, ausonly=True)
    #typical_aaods()
    #typical_aaod_month()
    omno2d_filter_determination()


    #for dates in [ [datetime(2005,1,1),datetime(2005,1,8)],
    #               [datetime(2005,2,1),datetime(2005,2,28)],
    #               [datetime(2006,4,1),datetime(2006,4,14)]]:
    #    smoke_vs_fire(dates[0],dates[1])

    #plot_VCC_rsc_gc_pp(d0=datetime(2005,3,1),dn=datetime(2005,3,31))
    #plot_swaths()
    # AMF tests and correlations
    #Check_OMI_AMF()
    #Check_AMF()
    #Check_AMF_relevelling()
    #test_amf_calculation(scount=5)
    #for aus_only in [True, False]:
    #    test_calculation_corellation(day=datetime(2005,1,1), oneday=False, aus_only=aus_only)
    #Test_Uncertainty()

    #compare_GC_OMI_new()

    # fires things
    #test_fires_fio()
    #test_fires_removed()

    #check_flags_and_entries() # check how many entries are filtered etc...
    # check some days (or one or no days)
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [0, 8, 16, 24, 32, 112] ]
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [112] ]
    #dates=[ datetime(2005,1,1) ]
    #check_products(date=dates[0],oneday=False)
    #Summary_RSC(date=dates[0], oneday=False)
    #dates=[ ]

    #CompareMaps(day=dates[0],oneday=False,ausonly=False)
    #Summary_RSC(oneday=False)
    #for day in dates:
        #test_reprocess_corrected(date=day, oneday=oneday)
        #test_reprocess_corrected(date=day, oneday=oneday, lllat=-50,lllon=100,urlat=-10,urlon=170, pltname="zoomed")
        #compare_products(date=day,oneday=True)
        #compare_products(date=day,oneday=True,positiveonly=True,pltname="positive")

        #compare_products(date=day,oneday=True, lllat=-50,lllon=100,urlat=-10,urlon=170, pltname="zoomed")
    # to be updated:
    #check_timeline()
    #reprocessed_amf_correlations()

    # other tests
    #check_high_amfs()
    #test_hchorp_apriori()
    #test_gchcho()

    # Check that cloud filter is doing as expected using old output without the cloud filter
    #compare_cloudy_map()

    # check the ref sector correction is not weird.
    #check_RSC(track_corrections=True)
