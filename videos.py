'''
File to test various parts of fio.py and reprocess.py
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
import h5py
from omhchorp import omhchorp as omrp

import numpy as np
from numpy.ma import MaskedArray as ma

from datetime import datetime, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
from matplotlib.animation import FuncAnimation as funcAni
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
from glob import glob

import omhchorp as omrp

datafields = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Data Fields/'
geofields  = 'HDFEOS/SWATHS/OMI Total Column Amount HCHO/Geolocation Fields/'

##############################
########## FUNCTIONS #########
##############################

def read_omhcho(path, szamax=60, screen=[-5e15, 1e17], maxlat=None, verbose=False):
    '''
    Read info from a single swath file
    NANify entries with main quality flag not equal to zero
    NANify entries where xtrackqualityflags aren't zero
    Chopped down version from fio.py
    '''
    
    # Total column amounts are in molecules/cm2
    field_qf    = datafields+'MainDataQualityFlag'
    #field_clouds= datafields+'AMFCloudFraction'
    field_rsc   = datafields+'ReferenceSectorCorrectedVerticalColumn' # molec/cm2
    field_xqf   = geofields +'XtrackQualityFlags'
    field_lon   = geofields +'Longitude'
    field_lat   = geofields +'Latitude'
    field_sza   = geofields +'SolarZenithAngle'
    
    ## read in file:
    with h5py.File(path,'r') as in_f:
        ## get data arrays
        lats    = in_f[field_lat].value     #[ 1644, 60 ]
        lons    = in_f[field_lon].value     #
        
        rsc     = in_f[field_rsc].value     # ref sector corrected vc
        
        #clouds  = in_f[field_clouds].value  # cloud fraction
        qf      = in_f[field_qf].value      #
        xqf     = in_f[field_xqf].value     # cross track flag
        sza     = in_f[field_sza].value     # solar zenith angle
        #
        ## remove missing values and bad flags: 
        # QF: missing<0, suss=1, bad=2
        if verbose:
            print("%d pixels in %s prior to filtering"%(np.sum(~np.isnan(rsc)),path))
        suss       = qf != 0
        if verbose:
            print("%d pixels removed by main quality flag"%np.nansum(suss))
        rsc[suss] = np.NaN
        lats[suss] = np.NaN
        lons[suss] = np.NaN
        
        # remove xtrack flagged data
        xsuss       = xqf != 0
        if verbose:
            removedcount= np.nansum(xsuss+suss) - np.nansum(suss)
            print("%d further pixels removed by xtrack flag"%removedcount)
        rsc[xsuss] = np.NaN
        lats[xsuss] = np.NaN
        lons[xsuss] = np.NaN
        
        # remove pixels polewards of maxlat
        if maxlat is not None:
            with np.errstate(invalid='ignore'):
                rmlat   = np.abs(lats) > maxlat
            if verbose:
                removedcount=np.nansum(rmlat+xsuss+suss) - np.nansum(xsuss+suss)
                print("%d further pixels removed as |latitude| > 60"%removedcount)
            rsc[rmlat] = np.NaN
            lats[rmlat] = np.NaN
            lons[rmlat] = np.NaN
        
        # remove solarzenithangle over 60 degrees
        if szamax is not None:
            rmsza       = sza > szamax
            if verbose:
                removedcount= np.nansum(rmsza+rmlat+xsuss+suss) - np.nansum(rmlat+xsuss+suss)
                print("%d further pixels removed as sza > 60"%removedcount)
            rsc[rmsza] = np.NaN
            lats[rmsza] = np.NaN
            lons[rmsza] = np.NaN
        
        # remove VCs outside screen range
        if screen is not None:
            # ignore warnings from comparing NaNs to Values
            with np.errstate(invalid='ignore'):
                rmscr   = (rsc<screen[0]) + (rsc>screen[1]) # A or B
            if verbose:
                removedcount= np.nansum(rmscr+rmsza+rmlat+xsuss+suss)-np.nansum(rmsza+rmlat+xsuss+suss)
                print("%d further pixels removed as value is outside of screen"%removedcount)
            rsc[rmscr] = np.NaN
            lats[rmscr] = np.NaN
            lons[rmscr] = np.NaN
    
    #return everything in a structure
    return {'lats':lats,'lons':lons, 'RSC_OMI':rsc}

def read_omhcho_day(day=datetime(2005,1,1),verbose=False):
    '''
    Read an entire day of omhcho swaths
    '''
    omhchopath='/media/jesse/My Book/jwg366/OMI/omhcho/'
    fnames=glob(omhchopath+'OMI-Aura*%4dm%02d%02d*'%(day.year, day.month, day.day))
    data=read_omhcho(fnames[0],verbose=verbose) # read first swath
    swths=[]
    for fname in fnames[1:]: # read the rest of the swaths
        swths.append(read_omhcho(fname,verbose=verbose))
    for swth in swths: # combine into one struct
        for key in swth.keys():
            axis= [0,1][key in ['omega','apriori','plevels']]
            data[key] = np.concatenate( (data[key], swth[key]), axis=axis)
    return data

def read_omhcho_8days(day=datetime(2005,1,1)):
    '''
    Read in 8 days all at once
    '''
    data8=read_omhcho_day(day)
    for date in [ day + timedelta(days=d) for d in range(1,8) ]:
        data=read_omhcho_day(date) 
        for key in data.keys():
            axis= [0,1][key in ['omega','apriori','plevels']]
            data8[key] = np.concatenate( (data8[key], data[key]), axis=axis)
    return data8

def createmap(data,lats,lons, vmin=5e13, vmax=1e17, latlon=True, 
              lllat=-50,  urlat=20, lllon=0, urlon=165):
    # Create a basemap map with 
    m=Basemap(llcrnrlat=lllat,  urcrnrlat=urlat,
          llcrnrlon=lllon, urcrnrlon=urlon,
          resolution='i',projection='merc')
    if len(lats.shape) == 1:
        lonsnew,latsnew=np.meshgrid(lons,lats)
    else:
        latsnew,lonsnew=(lats,lons)
    cs=m.pcolormesh(lonsnew, latsnew, data, latlon=latlon, 
                vmin=vmin,vmax=vmax,norm = LogNorm(), clim=(vmin,vmax))
    cs.cmap.set_under('white')
    cs.cmap.set_over('pink')
    cs.set_clim(vmin,vmax)
    
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    cb.set_label('Molecules/cm2')
    
    return m, cs, cb

def linearmap(data,lats,lons,vmin=None,vmax=None, latlon=True, 
              lllat=-80, urlat=80, lllon=-179, urlon=179):
    
    m=Basemap(llcrnrlat=lllat,  urcrnrlat=urlat,
          llcrnrlon=lllon, urcrnrlon=urlon,
          resolution='l',projection='merc')
    
    cs=m.pcolormesh(lons, lats, data, latlon=latlon, vmin=vmin, vmax=vmax)
    
    if vmin is not None:
        cs.cmap.set_under('white')
        cs.cmap.set_over('pink')
        cs.set_clim(vmin,vmax)
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    return m, cs, cb
    
def ausmap(data,lats,lons,vmin=None,vmax=None, linear=False):
    lllat=-50;urlat=-5;lllon=100;urlon=160
    fn=[createmap,linearmap][linear]
    return fn(data,lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon)

#############################################################################
######################       videos                 #########################
#############################################################################

def swath_video(days=20, seconds=None):
    ''' Create one and 8day averaged swath videos - for X years worth of data at Y seconds per frame '''

    ## Files to be read:
    #all the omhcho files...
    dates= [datetime(2005,1,1) + timedelta(days=d) for d in range(days)]
    daystrs = [day.strftime("%Y %m %d") for day in dates]
    
    daylist=[]
    for day in dates:
        daylist.append(read_omhcho_day())
    
    ## Plotting
    fig = plt.figure(figsize=(14,11))
    
    lats=daylist[0]['lats']
    lons=daylist[0]['lons']
    data=daylist[0]['RSC_OMI']
    m,cs,cb=createmap(data,lats,lons)
    
    cb.set_label('Molec/cm2')
    Ohcho='$\Omega_{HCHO}$ '
    txt=plt.title(Ohcho+daystrs[0])
    
    plt.savefig('checkvideo.png')
    
    # millisecond frame interval
    interval= 500
    if seconds is not None: interval=seconds*1000
    
    # animate 
    def updatefig(i):
        #global cs
        #for c in cs.collections: c.remove()
        #cs.remove()
        m,cs,cb = createmap(daylist[i]['RSC_OMI'],daylist[i]['lats'],daylist[i]['lons'])
        txt.set_text(Ohcho+daystrs[i])
        
    ani = funcAni(fig, updatefig, frames=len(dates), interval=interval)
    
    ani.save('Swaths_1day_Movie.mp4')
    print("Did it do it?")

def product_video(days=20,millisecs=250):
    ''' create video of RSC product '''
    # should just be able to read in the days from fio.read_omhchorp
    dates= [datetime(2005,1,1) + timedelta(days=d) for d in range(days)]
    daystrs = [day.strftime("%Y %m %d") for day in dates]
    
    daylist=[]
    for day in dates:
        daylist.append(omrp(day,oneday=True))
    
    ## Plotting
    fig = plt.figure(figsize=(14,11))
    
    lats=daylist[0].latitude
    lons=daylist[0].longitude
    #mlons,mlats = np.meshgrid(lons,lats)
    data=daylist[0].VCC
    m,cs,cb=createmap(data,lats,lons)
    
    cb.set_label('Molec/cm2')
    Ohcho='$\Omega_{HCHO}$ '
    txt=plt.title(Ohcho+daystrs[0])
    
    plt.savefig('checkvideo.png')
    
    # animate 
    def updatefig(i):
        #global cs
        #for c in cs.collections: c.remove()
        #cs.remove()
        m,cs,cb = createmap(daylist[i]['RSC_OMI'],daylist[i]['lats'],daylist[i]['lons'])
        txt.set_text(Ohcho+daystrs[i])
        
    ani = funcAni(fig, updatefig, frames=len(dates), interval=millisecs)
    
    ani.save('VCC_1day_Movie.mp4')
    print("Did it do it?")


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
    plt.legend(loc=0)
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
    
    # save plot
    pname='pictures/correlations%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    f.suptitle("Product comparison for %s"%ymdstr,fontsize=28)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

    
##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running videos.py")
    swath_video()