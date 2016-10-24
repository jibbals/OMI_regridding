'''
File to test various parts of fio.py and reprocess.py
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
import fio
import reprocess
from JesseRegression import RMA
from omhchorp import omhchorp as omrp

import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy

from datetime import datetime#, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
#import matplotlib.patches as mpatches

import timeit
import random

Ohcho='$\Omega_{HCHO}$'
Ovc='$\Omega_{VC}$'
Ovcc='$\Omega_{VCC}$'
Oomi='$\Omega_{OMI}$'
Ogc='$\Omega_{GC}$'

##############################
########## FUNCTIONS #########
##############################

def InitMatplotlib():
    """set some Matplotlib stuff."""
    matplotlib.rcParams["text.usetex"]      = False     # 
    matplotlib.rcParams["legend.numpoints"] = 1         # one point for marker legends
    matplotlib.rcParams["figure.figsize"]   = (12, 10)  #
    matplotlib.rcParams["font.size"]        = 18        # font sizes:
    matplotlib.rcParams["axes.titlesize"]   = 26        # title font size
    matplotlib.rcParams["axes.labelsize"]   = 20        #
    matplotlib.rcParams["xtick.labelsize"]  = 16        #
    matplotlib.rcParams["ytick.labelsize"]  = 16        #

def regularbounds(x,fix=False):
    # Take a lat or lon array input and return the grid edges
    
    newx=np.zeros(len(x)+1)
    xres=x[1]-x[0]
    newx[0:-1]=np.array(x) - xres/2.0
    newx[-1]=newx[-2]+xres
    # If the ends are outside 90N/S or 180E/W then bring them back
    if fix:
        if newx[-1] >= 90: newx[-1]=89.99
        if newx[0] <= -90: newx[0]=-89.99
        if newx[-1] >= 180: newx[-1]=179.99
        if newx[0] <= -180: newx[0]=-179.99
    return newx

def createmap(data,lats,lons, vmin=5e13, vmax=1e17, latlon=True, 
              lllat=-80, urlat=80, lllon=-179, urlon=179, colorbar=True):
    # Create a basemap map with 
    m=Basemap(llcrnrlat=lllat,  urcrnrlat=urlat,
          llcrnrlon=lllon, urcrnrlon=urlon,
          resolution='i',projection='merc')
    if len(lats.shape) == 1:
        #latnew=regularbounds(lats)
        #lonnew=regularbounds(lons)
        #lonsnew,latsnew=np.meshgrid(lonnew,latnew)
        lonsnew,latsnew=np.meshgrid(lons,lats)
    else:
        latsnew,lonsnew=(lats,lons)
    cs=m.pcolormesh(lonsnew, latsnew, data, latlon=latlon, 
                vmin=vmin,vmax=vmax,norm = LogNorm(), clim=(vmin,vmax))
    cs.cmap.set_under('white')
    cs.cmap.set_over('pink')
    cs.set_clim(vmin,vmax)
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    if not colorbar:
        return m,cs
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    cb.set_label('Molecules/cm2')
    
    return m, cs, cb
def globalmap(data,lats,lons):
    return createmap(data,lats,lons,lllat=-80,urlat=80,lllon=-179,urlon=179)
def ausmap(data,lats,lons,vmin=None,vmax=None,colorbar=True):
    return createmap(data,lats,lons,lllat=-50,urlat=-5,lllon=100,urlon=160, vmin=vmin, vmax=vmax, colorbar=colorbar)
def linearmap(data,lats,lons,vmin=None,vmax=None, latlon=True, 
              lllat=-80, urlat=80, lllon=-179, urlon=179):
    
    m=Basemap(llcrnrlat=lllat,  urcrnrlat=urlat,
          llcrnrlon=lllon, urcrnrlon=urlon,
          resolution='l',projection='merc')
    
    if len(lats.shape) == 1:
        latsnew=regularbounds(lats)
        lonsnew=regularbounds(lons)
        lonsnew,latsnew=np.meshgrid(lonsnew,latsnew)
    else:
        latsnew,lonsnew=(lats,lons)
    cs=m.pcolormesh(lonsnew, latsnew, data, latlon=latlon, vmin=vmin, vmax=vmax)
    if vmin is not None:
        cs.cmap.set_under('white')
        cs.cmap.set_over('pink')
        cs.set_clim(vmin,vmax)
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    return m, cs, cb

def plot_rec(bmap, inlimits, color=None, linewidth=None):
    '''
    Plot rectangle on basemap(arg 0) using [lat0,lon0,lat1,lon1](arg 1)
    '''
    # lat lon pairs for each corner
    limits=inlimits
    if limits[0]==-90:
        limits[0]=-89
    if limits[2]==90:
        limits[2]=89
    ll = [ limits[0], limits[1]]
    ul = [ limits[2], limits[1]]
    ur = [ limits[2], limits[3]]
    lr = [ limits[0], limits[3]]
    # shape to draw lats(y) and lons(x)
    ys = np.array([ll[0], ul[0],
                  ul[0], ur[0],
                  ur[0], lr[0],
                  lr[0], ll[0]])
    xs = np.array([ll[1], ul[1],
                  ul[1], ur[1],
                  ur[1], lr[1],
                  lr[1], ll[1]])
    print(xs,ys)
    x,y=bmap(xs,ys)
    bmap.plot(x, y, latlon=False, color=color, linewidth=linewidth)
    
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

#############################################################################
######################       TESTS                  #########################
#############################################################################

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
    gclats,gclons=gcdat.lats,gcdat.lons
    # plot 1) showing VCs before and after correction
    vmin,vmax=1e14,1e17
    f=plt.figure(0,figsize=(17,16))
    titles=[Ovc,Ovcc,Ogc]
    sbplots=[221,222,212]
    lims=[(-60,30,45,160),(-60,30,45,160),(-75,-170,75,160)]
    for i,arr in enumerate([dat.VC_GC,dat.VCC,gcdat.VC_HCHO*1e-4]):
        plt.subplot(sbplots[i])
        ilats=[lats,gclats][i==2]
        ilons=[lons,gclons][i==2]
        m,cs=createmap(arr,ilats,ilons,colorbar=False,vmin=vmin,vmax=vmax,
                       lllat=lims[i][0],lllon=lims[i][1],urlat=lims[i][2],urlon=lims[i][3])
        plt.title(titles[i],fontsize=25)
        if i==2:
            # rectangle around RSC
            plot_rec(m,dat.RSC_region,color='k',linewidth=4)
    
    plt.suptitle('Reference Sector Correction '+ymdstr,fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.20, 0.05, 0.6])
    cb=f.colorbar(cs,cax=cbar_ax)
    cb.set_ticks(np.logspace(13,17,5))
    cb.set_label('molec/cm$^2$',fontsize=24)
    
    pltname='pictures/Summary_RSC_Effect_%s.png'%ymdstr
    f.savefig(pltname)
    print ('%s saved'%pltname)
    plt.close(f)
    # plot 2:
    print("RSC_Summary() plot two to be implemented")

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
    m,cs,cb = createmap(omhchorp['SC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = createmap(omhchorp['VC_OMI'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[0,2])
    m,cs,cb = linearmap(omhchorp['AMF_OMI'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = createmap(omhchorp['VCC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = createmap(omhchorp['VC_GC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC GC')
    plt.sca(axes[1,2])
    m,cs,cb = linearmap(omhchorp['AMF_GC'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GC}$')
    
    # Plot third row the GEOS-Chem map divergences
    
    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    OMIdiff=100*(omhchorp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omhchorp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[2,0])
    vmin,vmax = -150, 150
    m,cs,cb = linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #m,cs,cb = createmap(gc.VC_HCHO*1e-4, glats,glons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[2,1])
    m,cs,cb = linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[2,2])
    m,cs,cb = createmap(fineHCHO, lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')
    
    # plot fourth row: uncertainties and AMF_GCz
    # 
    plt.sca(axes[3,0])
    m,cs,cb = createmap(omhchorp['col_uncertainty_OMI'], lats, lons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
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
    m,cs,cb = linearmap(omhchorp['AMF_GCz'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GCz}$')
    
    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="pictures/%scorrected%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def test_calculation_corellation(day=datetime(2005,1,1), oneday=False, aus_only=False):
    '''
    Look closely at AMFs over Australia, specifically over land
    and see how our values compare against the model and OMI swaths.
    '''
    # useful strings
    ymdstr=day.strftime('%Y%m%d')
    Ovcc='$\Omega_{OMI_{GCC}}$'
    Oomic="$\Omega_{OMI_{RSC}}$"
    Oomi="$\Omega_{OMI}$"
    
    # read in omhchorp
    om=omrp(day,oneday=oneday)
    VCC=om.VCC
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
    vcomi_o     = ma(VC_OMI,mask=~oceaninds)
    vcomic_o      = ma(VC_OMI_RSC,mask=~oceaninds)
    vcc_o       = ma(VCC,mask=~oceaninds)
    landunc     = ma(unc,mask=~landinds)
    
    # Print the land and ocean averages for each product
    print("%s land averages (oceans are global):"%(['Global','Australian'][aus_only]))
    print("%25s   land,   ocean"%'')
    for arrl,arro,arrstr in zip([vcomi_l, vcomic_l, vcc_l],[vcomi_o,vcomic_o,vcc_o],['OMI','OMI_RSC','OMI_GCC']):
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
    pname='pictures/land_VC_hist%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)
    
    # Plot the maps:
    #
    
    f=plt.figure(figsize=(16,13))
    plt.subplot(231)
    lllat=-80; urlat=80; lllon=-175; urlon=175
    if aus_only:
        lllat=-50; urlat=-5; lllon=100; urlon=170
    
    # OMI_RSC map
    m,cs,cb = createmap(vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title(Oomic,fontsize=20)
    
    # VCC map
    plt.subplot(232)
    m,cs,cb = createmap(vcc_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title(Ovcc,fontsize=20)
    
    # (VCC- OMI_RSC)/OMI_RSC*100 map
    plt.subplot(233)
    m,cs,cb = linearmap((vcc_l-vcomic_l)*100/vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=-200,vmax=200)
    plt.title('(%s - %s)*100/%s'%(Ovcc,Oomic,Oomic),fontsize=20)
    
    # omi uncertainty:
    plt.subplot(234)
    m,cs,cb = createmap(landunc, mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title("Column uncertainty (OMI)",fontsize=20)
    
    # show corellations for OMI_RSC- VCC
    nans=np.isnan(vcc_l) + np.isnan(vcomic_l) + vcomic_l.mask # all the nans we won't fit
    plt.subplot(235)
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
    plt.subplot(236)
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
    pname='pictures/correlations%s%s_%s.png'%(eightstr,ausstr,ymdstr)
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
        innerplot='pictures/AMF_test/AMF_test_innerplot%d.png'%i
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
    plt.plot([1,75], 10**(slp*np.log(np.array([1,75]))+intrcpt),color='red',
            label='slope=%.5f, r=%.5f'%(slp,r))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('pictures/AMF_test/AMF_test_corr.png')
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
    m,cs,cb=linearmap(landmlats,landmlons,landamfo)
    f.savefig('oceancheck.png')
    plt.close(f)
    # Check slopes and regressions of ocean/non ocean AMFs
    f=plt.figure(figsize=(14,14))
    plt.scatter(amf[ocean], amfo[ocean], color='cyan', alpha=0.5)
    plt.scatter(amf[~ocean],amfo[~ocean], color='fuchsia', alpha=0.5)
    slopeo,intercepto,ro,p,sterr = stats.linregress(amf[ocean], amfo[ocean])
    slopel,interceptl,rl,p,sterr = stats.linregress(amf[~ocean],amfo[~ocean])
    plt.plot([1,60], slopel*np.array([1,60])+interceptl,color='fuchsia',
            label='Land: slope=%.5f, r=%.5f'%(slopel,rl))
    plt.plot([1,60], slopeo*np.array([1,60])+intercepto, color='cyan',
            label='Ocean: slope=%.5f, r=%.5f'%(slopeo,ro))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('pictures/AMF_test/AMF_test_corr_masked.png')
    
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
        m,cs,cb = createmap(d['VC_OMI'],latmids,lonmids)
        plt.title('VC Regridded')
        plt.sca(axes[i,1])
        m,cs,cb = createmap(d['VC_GC'],latmids,lonmids)
        plt.title('VC Reprocessed')
        plt.sca(axes[i,2])
        m,cs,cb = linearmap(d['AMF_GC'],latmids,lonmids,vmin=1.0,vmax=6.0)
        plt.title('AMF reprocessed')
        i=1
        
    # save plots
    plt.tight_layout()
    plt.savefig("pictures/cloud_filter_effects.png")
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
    plt.savefig('pictures/AQUAgrids.png')
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
    m1,cs1, cb1 = linearmap(orig,mlats,mlons, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)
    ax2=plt.subplot(122)
    m2,cs2, cb2 = linearmap(regr,mlats2,mlons2, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)
    
    for i in range(2):
        [ax1,ax2][i].set_title(axtitles[i])
        [cb1, cb2][i].set_label('Fire Count (8 day)')
    
    plt.suptitle('AQUA 2005001',fontsize=20)
    plt.tight_layout()
    plt.savefig('pictures/AQUA2005001.png')
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
        m,cs = ausmap(arr,lats,lons,vmin=vmin,vmax=vmax,colorbar=False)
        plt.title(titles[i])
    pname='pictures/fire_exclusion_%s.png'%['8d','1d'][oneday]
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
    plt.savefig('pictures/GCHCHO_Vertical_Columns%s.png'%ymdstr)
    plt.clf()
    
    plt.figure(figsize=(12,12))
    gchcho.PlotProfile()
    plt.savefig('pictures/GCHCHO_EGProfile%s.png'%ymdstr)
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
    pltname='pictures/Shape_Factor_Examples.png'
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
    outpic='pictures/high_AMF_omegas.png'
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
        outfig1='pictures/track_corrections%s.png'%yyyymmdd
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
    newlons=regularbounds(lons_rp[rsc_lons])
    newlats=regularbounds(lats_rp)
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
    outfig2='pictures/RSC_GC_%s.png'%yyyymmdd
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
    plt.savefig("pictures/distributions%d_%s"%([8,1][oneday], yyyymmdd))
    plt.close()

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running tests.py")
    InitMatplotlib()
    Summary_RSC()
    #test_fires_fio()
    #test_fires_removed()
    #test_amf_calculation() # Check the AMF stuff
    #check_flags_and_entries() # check how many entries are filtered etc...
    # check some days (or one or no days)
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [0, 8, 16, 24, 32, 112] ]
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [112] ]
    #dates=[ datetime(2005,1,1) ]
    dates=[ ]
    for day in dates:
        for oneday in [True, False]:
            #test_reprocess_corrected(date=day, oneday=oneday)
            #test_reprocess_corrected(date=day, oneday=oneday, lllat=-50,lllon=100,urlat=-10,urlon=170, pltname="zoomed")
            for aus_only in [True, False]:
                test_calculation_corellation(day=day, oneday=oneday, aus_only=aus_only)
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
