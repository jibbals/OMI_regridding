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

from omhchorp import omhchorp as omrp


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

def createmap(data,lats,lons, vmin=5e13, vmax=1e17, 
              lllat=-50,  urlat=20, lllon=0, urlon=165):
    # Create a basemap map with 
    m=Basemap(llcrnrlat=lllat,  urcrnrlat=urlat,
          llcrnrlon=lllon, urcrnrlon=urlon,
          resolution='l',projection='merc')
    mlons,mlats=np.meshgrid(lons,lats)
    x,y=m(mlons,mlats)
    
    cs=m.pcolormesh(x, y, data, vmin=vmin,vmax=vmax,
                    norm = LogNorm(), clim=(vmin,vmax))
    cs.cmap.set_under('white')
    cs.cmap.set_over('pink')
    cs.set_clim(vmin,vmax)
    
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    cb.set_label('Molec/cm2')
    
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

def product_video(days=5,millisecs=600,save_pics=False,swaths=False, oneday=True):
    ''' create video of RSC product '''
    
    fs=36
    fs2=28
    dates = [datetime(2005,1,1) + timedelta(days=d) for d in range(days)]
    if not oneday:
        dates = [datetime(2005,1,1) + timedelta(days=d*8) for d in range(int(np.floor(days/8)))]
    
    daystrs = [day.strftime("%Y %m %d") for day in dates]
    daylist = []
    for day in dates:
        daylist.append(omrp(day,oneday=oneday))
    
    prefix    = ['vcc','swath'][swaths]
    postfix   = ['8d','1d'][oneday]
    datalist  = [ [d.VCC, d.VC_OMI][swaths] for d in daylist]
    
    # lat lon edges
    lats,lons = daylist[0].latlon_bounds()
    Ohcho     = '$\Omega_{HCHO}$ '
    
    # if we're saving all the pictures, loop through, save, return
    if save_pics:
        fig = plt.figure(figsize=(14,11))
        for i,day in enumerate(daylist):
            # Create the figure
            m,cs,cb=createmap(datalist[i],lats,lons)
            cb.set_label('Molec/cm2',fontsize=fs2)
            plt.title(Ohcho+daystrs[i],fontsize=fs)
            # save and clear the figure:
            pname='pictures/video/%s_%s_%03d.png'%(prefix,postfix,i)
            plt.tight_layout()
            plt.savefig(pname)
            plt.clf()
            print('saved '+pname)
        plt.close(fig)
        return
    
    ## initial plot for video
    fig = plt.figure(figsize=(14,11))
    
    m,cs,cb=createmap(datalist[0],lats,lons)
    cb.set_label('Molec/cm2',fontsize=fs2)
    plt.title(Ohcho+daystrs[0],fontsize=fs)
    plt.tight_layout()
    # animate update function
    def updatefig(i):
        plt.clf()
        m,cs,cb = createmap(datalist[i],lats,lons)
        plt.title(Ohcho+daystrs[i])
        cb.set_label('Molec/cm2')
        plt.tight_layout()
        print('Plotting '+daystrs[i])
        
    ani = funcAni(fig, updatefig, frames=len(dates), interval=millisecs)
    
    vname='%s_%say_Movie.mp4'%(prefix,postfix)
    ani.save(vname)
    print("saved "+vname)


##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running videos.py")
    for save_pics in [True,False]:
        product_video(days=80, oneday=True, millisecs=750, save_pics=save_pics)
        product_video(days=80, oneday=False, millisecs=800, save_pics=save_pics)
    #product_video(days=8, oneday=True, millisecs=750, save_pics=True)
        