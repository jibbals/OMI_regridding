#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:28:15 2016

Hold functions which will generally plot or print stuff

@author: jesse
"""
###############
### MODULES ###
###############
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap, interp
import matplotlib.pyplot as plt
plt.ioff() # seems to fix bug with plotting against datetimes
import matplotlib.dates as mdates
import warnings # To ignore warnings
#import matplotlib.colors as mcolors #, colormapping
from matplotlib.colors import LogNorm # for lognormal colour bar
from datetime import timedelta
import seaborn # for density plots
import matplotlib.image as mpimg # for direct image plotting

# Add parent folder to path
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#sys.path.insert(0,os.path.dirname(currentdir))

from utilities.JesseRegression import RMA
from utilities import utilities as util
from utilities.utilities import regrid
from utilities import GMAO

#sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=False

# S W N E
__AUSREGION__=GMAO.__AUSREGION__
__GLOBALREGION__=GMAO.__GLOBALREGION__

__cities__ = {'Syd':[-33.87,151.21], # Sydney
              'Can':[-35.28,149.13], # Canberra
              'Mel':[-37.81,144.96], # Melbourne
              'Ade':[-34.93,138.60], # Adelaide
              'Wol':[-34.43,150.89], # Wollongong
             }

# Want to look at timeseires and densities in these subregions:
__subzones_AUS__     = [__AUSREGION__,  # first zone is container for the rest
                        [-36,148,-32,153], # Canberra, Newcastle, and Sydney
                        [-36,134,-33,140], # Adelaide and port lincoln
                        [-30,125,-25,135], # Emptly land
                        [-39,142,-36,148], # Melbourne
                       ]
__subzones_colours__ = ['k', 'red', 'green', 'cyan', 'darkred']

__subzones_labels__  = ['Aus', 'Sydney','Adelaide','Mid','Melbourne']

# Want to look at timeseires and densities in these subregions:
__subregions__ = GMAO.__subregions__
__subregions_colors__ = GMAO.__subregions_colors__
__subregions_labels__ = GMAO.__subregions_labels__

###############
### METHODS ###
###############

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
    matplotlib.rcParams['image.cmap'] = 'plasma' #'PuRd' #'inferno_r'       # Colormap default

def regularbounds(x,fix=False):
    '''
        Take a lat or lon array input and return the grid edges
        Works on regular grids, and GEOSChem lat mids
    '''
    # assert x[1]-x[0] == x[2]-x[1], "Resolution at edge not representative"
    # replace assert with this if it works, HANDLES GEOS CHEM LATS PROBLEM ONLY
    if not np.isclose(x[1]-x[0], x[2]-x[1]):
        xres=x[2]-x[1]   # Get resolution away from edge
        if __VERBOSE__:
            print("Edge resolution %.3f, replaced by %.3f"%(x[1]-x[0], xres))
        x[0]=x[1]-xres   # push out the edges
        x[-1]=x[-2]+xres #
        if __VERBOSE__:
            print("Lats: %.2f, %.2f, %.2f, ..., %.2f, %.2f, %.2f"%tuple([x[i] for i in [0,1,2,-3,-2,-1]]))

    # new vector for array
    newx=np.zeros(len(x)+1)
    # resolution from old vector
    xres=x[1]-x[0]
    # edges will be mids - resolution / 2.0
    newx[0:-1]=np.array(x) - xres/2.0
    # final edge
    newx[-1]=newx[-2]+xres

    # Finally if the ends are outside 90N/S or 180E/W then bring them back
    if fix:
        if newx[-1] >= 90: newx[-1]=89.99
        if newx[0] <= -90: newx[0]=-89.99
        if newx[-1] >= 180: newx[-1]=179.99
        if newx[0] <= -180: newx[0]=-179.99

    return newx

def add_colourbar(f,cs,ticks=None,label=None,fontsize=15):
    '''
        Add a colour bar to a figure
    '''
    f.tight_layout()
    f.subplots_adjust(top=0.95)
    f.subplots_adjust(right=0.84)
    cbar_ax = f.add_axes([0.87, 0.20, 0.04, 0.6])
    cb=f.colorbar(cs,cax=cbar_ax)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label,fontsize=fontsize)
    return cb

def add_rectangle(bmap, limits, color='k', linewidth=1):
    '''
    Plot rectangle on basemap(arg 0) using [lat0,lon0,lat1,lon1](arg 1)
    '''
    # lat lon pairs for each corner
    ll = [ limits[0], limits[1]]
    ul = [ limits[2], limits[1]]
    ur = [ limits[2], limits[3]]
    lr = [ limits[0], limits[3]]
    # shape to draw lats(y) and lons(x)
    ys = [ll[0], ul[0],
          ul[0], ur[0],
          ur[0], lr[0],
          lr[0], ll[0]]
    xs = [ll[1], ul[1],
          ul[1], ur[1],
          ur[1], lr[1],
          lr[1], ll[1]]
    bmap.plot(xs, ys, latlon = True, color=color, linewidth=linewidth)

def add_point(bmap, lat,lon,
              color='k', marker='*',markersize=5,
              label=None, fontsize=12, fontcolor='k',
              xlabeloffset=1000,ylabeloffset=2000):
    '''
        Add point with optional label onto bmap
    '''
    x,y = bmap(lon,lat)

    bmap.plot(x,y, marker, color=color, markersize=markersize)
    if label is not None:
        plt.text(x+xlabeloffset,y+ylabeloffset, label,
                 color=fontcolor, fontsize=fontsize)

def add_regression(X,Y,label=None, addlabel=True, exponential=False, **pargs):
    '''
    plots RMA between X and Y
        Y = mX+b
    or X and ln Y if exponential flag is set
        ln Y = mX+b
        Y = exp(mX + b)
    '''
    Y2=np.copy(Y)
    if exponential:
        Y2[Y2<=0] = np.NaN
        Y2=np.log(Y2)

    # find regression
    m,b,r,ci1,ci2 = RMA(np.array(X), np.array(Y2))
    xx= np.array([np.nanmin(X), np.nanmax(X)])
    Xspace=np.linspace(xx[0], xx[1], 30)
    Yline=m*Xspace + b
    if exponential:
        Yline= np.exp( m * Xspace + b)

    # set up lable
    if addlabel and (label is None):
        n=len(X)
        label='Y = %.2fX + %.2f ; r=%.2f, N=%d'%(m,b,r,n)
        if exponential:
            n=np.sum(~np.isnan(Y2))
            label='Y = exp(%.2fX + %.2f) ; r=%.2f, N=%d'%(m,b,r,n)

    plt.plot(Xspace,Yline,label=label, **pargs)
    return m,b,r,ci1,ci2


def basicmap(data, lats, lons, latlon=True,
              aus=False, region=__GLOBALREGION__, linear=False,
              pname=None,title=None,suptitle=None,
              vmin=None,vmax=None,colorbar=True):
    '''
        Pass in data[lat,lon], lats[lat], lons[lon]
    '''
    if __VERBOSE__:
        print("basicmap called: %s"%str(title))
        #print("Data %s, %d lats and %d lons"%(str(data.shape),len(lats), len(lons)))

    # Create a basemap map with region as inputted
    if aus: region=__AUSREGION__
    lllat=region[0]; urlat=region[2]; lllon=region[1]; urlon=region[3]
    m=Basemap(llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon,
              resolution='i', projection='merc')

    ## basemap pcolormesh uses data edges
    ##

    # Make edges into 2D meshed grid
    mlons,mlats=np.meshgrid(lons,lats)

    norm=None
    if not linear:
        if __VERBOSE__:print("basicmap() is removing negatives")
        norm=LogNorm()
        data[data<=0.0]=np.NaN

    cs=m.pcolormesh(mlons, mlats, data, latlon=latlon, norm=norm)
    if vmin is None:
        vmin = np.nanmin(data) + np.abs(np.nanmin(data))*0.1
    if vmax is None:
        vmax = np.nanmax(data) - np.abs(np.nanmax(data))*0.1
    cs.set_clim(vmin,vmax)

    # draw coastline and equator(no latlon labels)
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0])

    # add titles and cbar label
    if title is not None:
        plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    cb=None
    if colorbar:
        cbargs={'size':'5%','pad':'1%','extend':'both'}
        cb=m.colorbar(cs,"bottom", **cbargs)

    # if a plot name is given, save and close figure
    if pname is not None:
        plt.savefig(pname)
        print("Saved "+pname)
        plt.close()
        return

    # if no colorbar is wanted then don't return one (can be set externally)
    return m, cs, cb

def createmap(data, lats, lons, make_edges=False, GC_shift=True,
              vmin=None, vmax=None, latlon=True,
              region=__GLOBALREGION__, aus=False, linear=False,
              clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
              ticks=None, cbarorient='bottom',
              xticklabels=None,
              set_bad=None, set_under=None, set_over=None,
              pname=None,title=None,suptitle=None, smoothed=False,
              cmapname=None):
    '''
        Pass in data[lat,lon], lats[lat], lons[lon]
        arguments:
            set_bad='blue' #should mask nans as blue
            GC_shift=True #will shift plot half a square left and down
        Returns map, cs, cb
    '''

    # Create a basemap map with region as inputted
    if aus: region=__AUSREGION__
    if __VERBOSE__:
        print("createmap called over %s (S,W,N,E)"%str(region))
        #print("Data %s, %d lats and %d lons"%(str(data.shape),len(lats), len(lons)))

    # First reduce data,lats,lons to the desired region (should save plotting time)
    regionplus=np.array(region) + np.array([-5,-10,5,10]) # add a little padding so edges aren't lost
    lati,loni=util.lat_lon_range(lats,lons,regionplus)
    data=data[lati,:]
    data=data[:,loni]
    lats=lats[lati]
    #print(lons)
    #print(loni)

    lons=lons[loni]


    lllat=region[0]; urlat=region[2]; lllon=region[1]; urlon=region[3]
    m=Basemap(llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon,
              resolution='i', projection='merc')

    # plt.colormesh arguments will be added to dictionary
    pcmeshargs={}

    if not linear:
        if __VERBOSE__:
            print('removing %d negative datapoints in createmap'%np.nansum(data<0))
        # ignore warnings of NaN comparison
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category =RuntimeWarning)
            data[data<=0] = np.NaN
        pcmeshargs['norm']=LogNorm()

    # Set vmin and vmax if necessary
    if vmin is None:
        vmin=1.05*np.nanmin(data)
    if vmax is None:
        vmax=0.95*np.nanmax(data)

    ## basemap pcolormesh uses data edges
    ##
    lats_e,lons_e=lats,lons
    lats_m,lons_m=lats,lons
    if make_edges:
        if __VERBOSE__: print("Making edges from lat/lon mids")
        nlat,nlon=len(lats), len(lons)
        lats_e=regularbounds(lats)
        lons_e=regularbounds(lons)
        assert nlat == len(lats_e)-1, "regularbounds failed: %d -> %d"%(nlat, len(lats_e))
        assert nlon == len(lons_e)-1, "regularbounds failed: %d -> %d"%(nlon, len(lons_e))
        ## midpoints, derive simply from edges
        lons_m=(lats_e[0:-1] + lats_e[1:])/2.0
        lats_m=(lons_e[0:-1] + lons_e[1:])/2.0
    elif GC_shift: # non edge-based grids need to be shifted left and down by half a box
        latres=lats[3]-lats[2]
        lonres=lons[3]-lons[2]
        lats=lats-latres/2.0
        lons=lons-lonres/2.0
        lats[lats < -89.9] = -89.9
        lats[lats > 89.9]  =  89.9
        lats_e,lons_e=lats,lons
        lats_m,lons_m=lats,lons

    ## interpolate for smoothed output if desired
    ##
    if smoothed:
        factor=5
        if __VERBOSE__: print("Smoothing data, by factor of %d"%factor)
        # 'increase' resolution
        nlats = factor*data.shape[0]
        nlons = factor*data.shape[1]
        lonsi = np.linspace(lons_m[0],lons[-1],nlons)
        latsi = np.linspace(lats_m[0],lats[-1],nlats)

        # also increase resolution of our edge lats/lons
        lats_e=regularbounds(latsi);
        lons_e=regularbounds(lonsi)
        lonsi, latsi = np.meshgrid(lonsi, latsi)
        # Smoothe data to increased resolution
        data = interp(data,lons,lats,lonsi,latsi)

    # Make edges into 2D meshed grid
    mlons_e,mlats_e=np.meshgrid(lons_e,lats_e)
    #x_e,y_e=m(lons_e,lats_e)

    errmsg="pcolormesh likes edges for lat/lon (array: %s, lats:%s)"%(str(np.shape(data)),str(np.shape(mlats_e)))
    if __VERBOSE__:
        print(errmsg)

    if cmapname is None:
        cmapname = matplotlib.rcParams['image.cmap']

    cmap=plt.cm.cmap_d[cmapname]
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))


    if set_bad is not None:
        cmap.set_bad(set_bad,alpha=0.0)

    pcmeshargs.update({'vmin':vmin, 'vmax':vmax, 'clim':(vmin, vmax),
                'latlon':latlon, 'cmap':cmap, })


    #force nan into any pixel with nan results, so color is not plotted there...
    mdata=np.ma.masked_invalid(data) # mask non-finite elements
    #mdata=data # masking occasionally throws up all over your face

    if __VERBOSE__:
        shapes=tuple([ str(np.shape(a)) for a in [mlats_e, mlons_e, mdata, mdata.mask] ])
        print("lats: %s, lons: %s, data: %s, mask: %s"%shapes)

    #for arr in mlons_e,mlats_e,mdata:
    #    print(np.shape(arr))
    cs=m.pcolormesh(mlons_e, mlats_e, mdata, **pcmeshargs)
    # colour limits for contour mesh
    if set_over is not None:
        cs.cmap.set_over(set_over)
    if set_under is not None:
        cs.cmap.set_under(set_under)
    cs.set_clim(vmin,vmax)


    # draw coastline and equator(no latlon labels)
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0])

    # add titles and cbar label
    if title is not None:
        plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    cb=None
    if colorbar:
        cbargs={'format':cbarfmt, 'ticks':ticks,
                'size':'5%', 'pad':'1%', 'extend':'both'}
        cb=m.colorbar(cs, cbarorient, **cbargs)
        if xticklabels is not None:
            cb.ax.set_xticklabels(xticklabels)

        if clabel is not None:
            cb.set_label(clabel)
        if cbarxtickrot is not None:
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=cbarxtickrot)

    # if a plot name is given, save and close figure
    if pname is not None:
        plt.savefig(pname)
        print("Saved "+pname)
        plt.close()
        return

    # if no colorbar is wanted then don't return one (can be set externally)
    return m, cs, cb

def density(data,lats=None,lons=None,region=None, **kdeargs):
    '''
        Plot seaborn density plot of data
        optionally subset to region
    '''
    Y=data
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region, data=[data])
        Y=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']
    #seaborn.set_style('whitegrid')
    seaborn.kdeplot(Y.flatten(),**kdeargs)
    return Y,lats,lons

def distplot(data, lats=None, lons=None, region=None, **distargs):
    '''
        plot seaborn distplot of data.flatten()
    '''
    Y=data
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region, data=[data])
        Y=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']
    #seaborn.set_style('whitegrid')
    mask=np.isnan(Y.flatten())
    seaborn.distplot(Y.flatten()[~mask],**distargs)
    return Y,lats,lons

def hatchmap(m, datamap, lats, lons, thresh, region=None,hatch='x',color='k'):
    '''
        add hatching to a basemap map
        optionally subset to a region

        region=[S,W,N,E]
        hatch = one or more of: \ / | - + x o O . *
    '''
    data=np.copy(datamap)
    if region is not None:
        subset=util.lat_lon_subset(lats,lons,region=region,data=[data])
        data=subset['data'][0]
        lats=subset['lats']
        lons=subset['lons']

    # Remove nans so they are not hatched
    data[np.isnan(data)] = thresh-1

    # Set up mask of everywhere under the threshhold
    aaod_masked=np.ma.masked_less_equal(data,thresh)
    mlons,mlats=np.meshgrid(lons,lats)
    mx,my = m(mlons,mlats)
    m.pcolor(mx, my, aaod_masked, hatch=hatch, color=color, alpha=0.)

def get_colors(cmapname, howmany, values=None,vmin=None,vmax=None):
    cmap=plt.cm.cmap_d[cmapname]
    if values is not None:
        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)
        norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax,clip=True)

        #mapper=plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return cmap(norm(values))

    return cmap(np.linspace(0, 1, howmany))

def plot_img(path):
    ''' '''
    img=mpimg.imread(path)
    plt.imshow(img)
    axes=plt.gca().axes
    axes.get_yaxis().set_visible(False)
    axes.get_xaxis().set_visible(False)

#def plot_swath(day, reprocessed=False,
#              oneday=True, region=__AUSREGION__,
#              edges=False , vmin=None, vmax=None,
#              aus=False, linear=True, clabel='molec/cm2', colorbar=True,
#              cbarfmt=None, cbarxtickrot=None,
#              pname=None,title=None,suptitle=None, smoothed=False,
#              cmapname=None, fillcontinents=None):
#    '''
#        Wrapper to plot gridded swath output for a day
#    '''
#
#
#    #swaths=fio.read_omhcho_day(day)
#    dkey=['VC_OMI_RSC','VCC'][reprocessed]
#    swaths=fio.read_omhchorp(day,oneday=oneday,keylist=[dkey,'latitude','longitude'])
#    data=swaths[dkey]
#    lats=swaths['latitude']
#    lons=swaths['longitude']
#
#    return createmap(data, lats, lons, edges=edges ,
#              vmin=vmin, vmax=vmax, latlon=True,
#              region=region, aus=aus, linear=linear,
#              clabel=clabel, colorbar=colorbar, cbarfmt=cbarfmt,
#              cbarxtickrot=cbarxtickrot, pname=pname,title=title,
#              suptitle=suptitle, smoothed=smoothed,
#              cmapname=cmapname, fillcontinents=fillcontinents)


def plot_rec(bmap, inlimits, color=None, linewidth=1):
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
    x,y=bmap(xs,ys)
    bmap.plot(x, y, latlon=False, color=color, linewidth=linewidth)

def plot_regression(X,Y, limsx=None, limsy=None, logscale=True,
                     legend=True, legendfont=22,
                     colour='k',linecolour='r', diag=True, oceanmask=None,
                     colours=None,size=None, cmap='rainbow', showcbar=True,
                     clabel=''):
    '''
        Regression between X and Y
        Optional to colour by some third list of equal length by setting colours
        Can alternatively split by oceanmask.
    '''
    X=np.array(X)
    Y=np.array(Y)
    nans=np.isnan(X) + np.isnan(Y)
    if limsx is None:
        spread = np.nanmax(X) - np.nanmin(X)
        limsx  = np.array([np.nanmin(X)-0.05*spread, np.nanmax(X)+0.05*spread])
    if limsy is None:
        spread = np.nanmax(Y) - np.nanmin(Y)
        limsy  = np.array([np.nanmin(Y)-0.05*spread, np.nanmax(Y)+0.05*spread])
    limsx0=np.copy(limsx);

    if oceanmask is None:
        if colours is None:
            plt.scatter(X[~nans], Y[~nans],color=colour)
        else:
            cm = plt.cm.get_cmap(cmap)
            sc=plt.scatter(X[~nans], Y[~nans], c=colours[~nans], s=size,
                           cmap=cm)
            if showcbar:
                cb=plt.colorbar(sc)
                cb.set_label(clabel)
        m,b,r,CI1,CI2=RMA(X[~nans], Y[~nans]) # get regression
        plt.plot(limsx, m*np.array(limsx)+b,color=linecolour,
                 label='Y = %.1eX + %.2e\n r=%.5f, n=%d'%(m,b,r,np.sum(~nans)))
    else:
        omask=~(nans+~oceanmask ) # true where not nan or land
        lmask=~(nans+oceanmask ) # true where not nan or ocean
        # first scatter plot everything not oceanic
        plt.scatter(X[omask], Y[omask], color='blue', alpha=0.4)#, label="Ocean" )
        plt.scatter(X[lmask], Y[lmask], color='gold')#, label="Land")

        # Line of best fit and RMA regression:
        lm,lx0,lr,lci1,lci2 = RMA(X[lmask], Y[lmask])
        m,x0,r,ci1,ci2 = RMA(X[omask], Y[omask])
        #move limit for lobf if log scale goes to negative
        if m*limsx[0] + x0 < 0 and logscale:
            limsx[0] = -x0/m + 100
        if (lm*limsx[0] + lx0 < 0) and logscale:
            limsx[0] = -lx0/lm + 100

        #plot lobf and label
        plt.plot( limsx, lm*limsx+lx0, color='k', linewidth=2,
                label='Land: Y = %.1eX + %.2e; r=%.5f'%(lm,lx0,lr))
        plt.plot( limsx, m*limsx+x0, color='blue',
                label='Ocean: Y = %.1eX + %.2e, r=%.5f'%(m,x0,r))

        print('Land: Y = %.5fX + %.2e; r=%.5f'%(lm,lx0,lr))
        print('with CI ranges of slope %2.5f, %2.5f'%(lci1[0][0],lci1[0][1]))
        print('with CI ranges of intercept %1.5e, %1.5e'%(lci1[1][0],lci1[1][1]))
        print('min, max land X: %.3e,%.3e'%(np.min(X[lmask]),np.max(X[lmask])) )
        print('min, max land Y: %.3e,%.3e'%(np.min(Y[lmask]),np.max(Y[lmask])) )

    if legend:
        plt.legend(loc=2,scatterpoints=1, fontsize=legendfont,frameon=False)
    if logscale:
        plt.yscale('log'); plt.xscale('log')
    plt.ylim(limsy); plt.xlim(limsx0)
    if diag:
        plt.plot(limsx0,limsx0,'--',color='k',label='1-1') # plot the 1-1 line for comparison

def plot_time_series(datetimes,values,
                     ylabel=None,xlabel=None, pname=None, legend=False, title=None,
                     monthly=False, monthly_func='mean',
                     xtickrot=30, dfmt='%Y%m', xticks=None, **pltargs):
    ''' plot values over datetimes '''

    if monthly:
        mdata=util.monthly_averaged(datetimes,values)
        datetimes=mdata['middates']
        values=mdata[monthly_func] # 'median', or 'mean'

    dates = mdates.date2num(datetimes)
    #plt.plot_date(dates, values, **pltargs)
    plt.plot(dates, values, **pltargs)

    #Handle ticks:
    #plt.gcf().autofmt_xdate()
    plt.xticks(rotation=xtickrot)
    myFmt = mdates.DateFormatter(dfmt)
    plt.gca().xaxis.set_major_formatter(myFmt)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if legend:
        plt.legend()
    if title is not None:
        plt.title(title)
    if xticks is not None:
        #xt=[]
        #xt.append(mdates.date2num(xticks))
        plt.xticks(xticks)
    if pname is not None:
        plt.savefig(pname)
        print('%s saved'%pname)
        plt.close()


def subzones_map(data, lats, lons, cmapname='plasma',
                 cities=__cities__,
                 subzones=__subzones_AUS__,
                 colors=__subzones_colours__,
                 labelcities=True,
                 **createmapargs):
    '''
        Plot australia, with subzones and stuff added
        Extra named arguments can be sent to createmap using **createmapargs
    '''


    region=subzones[0]

    bmap,cs,cb = createmap(data, lats, lons, region=region,
                           cmapname=cmapname, **createmapargs)
    # Add cities to map
    if labelcities:
        for city,latlon in cities.items():
            add_point(bmap,latlon[0],latlon[1],markersize=6, marker='o',
                      color='k', label=city, fontsize=12,
                      xlabeloffset=-50000,ylabeloffset=30000)

    # Add squares to map:
    for i,subzone in enumerate(subzones[1:]):
        add_rectangle(bmap,subzone,color=colors[i+1],linewidth=2)

    return bmap,cs,cb

def subzones_TS(data, dates, lats, lons,
                subzones=__subzones_AUS__,colors=__subzones_colours__, labels=None,
                logy=False, ylims=None, skip_first_region=False):
    '''
        time series for each subzone in data[time,lats,lons]
    '''
    series=[]
    #if use_doys:
    #    doys = [ d.timetuple().tm_yday for d in dates ]
    #else: # use days from start of input
    d0=dates[0]
    doys = [ (d-d0).days for d in dates ]

    if labels is None:
        labels=[None,]*len(subzones)

    # loop over subzones
    for i,subzone in enumerate(subzones):
        # Subset our data to subzone
        if i==0 and skip_first_region:
            continue
        datz=np.copy(data)
        lati,loni=util.lat_lon_range(lats,lons,subzone)
        datz = datz[:,lati,:]
        datz = datz[:,:,loni]

        # Also remove negatives
        #negmask=datz < 0
        #print("Removing %d negative squares"%(np.sum(negmask)))
        #datz[negmask]=np.NaN

        # get mean and percentiles of interest for plot
        #std = np.nanstd(no2,axis=(1,2))
        upper = np.nanpercentile(datz,75,axis=(1,2))
        lower = np.nanpercentile(datz,25,axis=(1,2))
        mean = np.nanmean(datz,axis=(1,2))
        series.append(mean)
        totmean = np.nanmean(datz)

        lw=[1,4][i==0] # linewidth

        # plot timeseries
        #plt.plot(doys, mean, color=colors[i],linewidth=lw)
        plt.plot_date(dates, mean, '-', color=colors[i],linewidth=lw)
        # Add IQR shading for first plot
        if i==0:
            #plt.fill_between(doys, lower, upper, color=colors[i],alpha=0.2, label=labels[i])
            plt.fill_between(dates, lower, upper, color=colors[i],alpha=0.2, label=labels[i])

        # show yearly mean
        #endbit=np.max(doys)
        #doyslen=np.max(doys)-np.min(doys)
        #plt.xlim([np.min(doys)-doyslen/100.0, endbit + doyslen*.15])
        #plt.plot([endbit+doyslen*.02, endbit+doyslen*.1], [totmean, totmean],
        #         color=colors[i], linewidth=lw, label=labels[i])
        endbit=np.max(dates)
        dateslen=np.max(dates)-np.min(dates)
        plt.xlim([np.min(dates)-dateslen/100.0, endbit + dateslen*.15])
        plt.plot([endbit+dateslen*.02, endbit+dateslen*.1], [totmean, totmean],
                 color=colors[i], linewidth=lw, label=labels[i])

        # change to log y scale?
        if logy:
            plt.yscale('log')

        if ylims is not None:
            plt.ylim(ylims)
            #yticks=list(ylims)
            #ytickstr=['%.0e'%tick for tick in yticks]
            #plt.yticks(yticks,ytickstr)

        #plt.ylabel(ylabel)
        plt.xlabel('Days since %s'%d0.strftime("%Y%m%d"))

    #plt.gcf().autofmt_xdate()
    if labels[0] is not None:
        plt.legend(loc='best', fontsize=10)
    return series

def subzones(data, dates, lats, lons, comparison=None, subzones=__subzones_AUS__,
             pname=None,title=None,suptitle=None, comparisontitle=None,
             clabel=None, vmin=None, vmax=None, linear=False,
             mapvmin=None, mapvmax=None,
             maskoceans=True,
             labelcities=True, labels=None,
             force_monthly=False, force_monthly_func='mean',
             colors=__subzones_colours__, noplot=False):
    '''
        Look at map of data[time,lats,lons], draw subzones, show time series over map and subzones
        can clear ocean with maskoceans=True (default).
        Region mapped is the first of the subzones
        If a mask is applied then also show map and time series after applying mask
    '''
    #region=subzones[0]
    j=int(comparison is not None)

    axs=[]
    series=[]
    series2=[]
    axs.append(plt.subplot(2,j+1,1))
    createmapargs={'vmin':mapvmin, 'vmax':mapvmax, 'clabel':clabel,
                 'title':title, 'suptitle':suptitle, 'linear':linear}

    # Mask ocean
    if maskoceans:
        oceanmask=util.oceanmask(lats,lons)
        print("Removing %d ocean squares"%(np.sum(oceanmask)))
        data[:,oceanmask] = np.NaN
        if comparison is not None:
            comparison[:,oceanmask] = np.NaN

    data_mean = np.nanmean(data,axis=0)
    if comparison is not None:
        comparison_mean=np.nanmean(comparison,axis=0)

    subzones_map(data_mean,lats,lons,subzones=subzones,colors=colors, labelcities=labelcities,
                 **createmapargs)

    # For each subset here, plot the timeseries
    if force_monthly:

        monthly_data=util.monthly_averaged(dates,data,keep_spatial=True)
        data=monthly_data[force_monthly_func]

        if comparison is not None:
            monthly_comparison=util.monthly_averaged(dates,comparison,keep_spatial=True)
            comparison=monthly_comparison[force_monthly_func]
        # use date list of monthly midpoints
        dates=monthly_data['dates']

    axs.append(plt.subplot(2,j+1,2+j))
    series=subzones_TS(data, dates, lats, lons, subzones=subzones,colors=colors,
                       labels=labels, ylims=[vmin,vmax])

    if comparison is not None:

        axs.append(plt.subplot(2,2,2))
        createmapargs['title']=comparisontitle
        createmapargs['suptitle']=None
        subzones_map(comparison_mean, lats, lons, labelcities=labelcities,
                     subzones=subzones,colors=colors,
                     **createmapargs)

        axs.append(plt.subplot(2,2,4, sharey=axs[1]))
        series2=subzones_TS(comparison, dates, lats, lons,
                            subzones=subzones, colors=colors,
                            ylims=[vmin,vmax])


    if pname is not None:
        plt.savefig(pname)
        print("saved ",pname)
        plt.close()

    return axs, series, series2, dates



def compare_maps(datas, lats, lons, pname=None, titles=['A','B'], suptitle=None,
                 clabel=None, region=__AUSREGION__, vmin=None, vmax=None,
                 rmin=-200.0, rmax=200., amin=None, amax=None,
                 axeslist = [None,None,None,None],
                 maskocean = False, ticks=None,
                 lower_resolution = False, normalise = False,
                 linear=False, alinear=True, rlinear=True):
    '''
        Plot two maps and their relative and absolute differences
        axeslist can be used to redirect panels... just don't set pname
        options:
            maskocean: mask ocean squares to np.NaN before plotting
            normalise: normalised by subtracting mean of whole map from each
            lower_resolution: if different resolutions for the maps, use lower one for both
                otherwise higher one is used for both
            linear: bmap scale
            alinear: absolute diff scale
            rlinear: relative diff scale
            axeslist = axes on which to send the plotting done in this function [11,12,21,22]
    '''
    A=datas[0]
    B=datas[1]
    Alats=lats[0]
    Alons=lons[0]
    Blats=lats[1]
    Blons=lons[1]


    # regrid the lower resolution data to upper unless flag set
    # Alats is higher resolution
    if len(Alats) > len(Blats):
        if lower_resolution:
            A = regrid(A,Alats,Alons,Blats,Blons)
            Alats,Alons=Blats,Blons
        else:
            B = regrid(B,Blats,Blons,Alats,Alons)
            Blats,Blons=Alats,Alons
    # Alats is lower resolution
    if len(Alats) < len(Blats):
        if lower_resolution:
            B = regrid(B,Blats,Blons,Alats,Alons)
            Blats,Blons=Alats,Alons
        else:
            A = regrid(A,Alats,Alons,Blats,Blons)
            Alats,Alons=Blats,Blons
    lats=Alats
    lons=Alons

    if maskocean:
        oceanmask=util.oceanmask(lats,lons)
        A[oceanmask]=np.NaN
        B[oceanmask]=np.NaN
    if normalise:
        A = A-np.nanmean(A)
        B = B-np.nanmean(B)
        linear=True # hard to use log scale centred around 0

    if vmax is None:
        vmax = np.nanmax(np.array(np.nanmax(A),np.nanmax(B)))
    if vmin is None:
        vmin = np.nanmin(np.array(np.nanmin(A),np.nanmin(B)))
    if amax is None:
        amax = vmax
    if amin is None:
        amin = -vmax

    # set up plot
    f,axes=plt.subplots(2,2,figsize=(16,14))

    # first plot plain maps
    plt.sca(axes[0,0])
    if axeslist[0] is not None:
        plt.sca(axeslist[0])
    args={'region':region, 'clabel':clabel, 'linear':linear, 'ticks':ticks,
          'lats':lats, 'lons':lons, 'title':titles[0] + "(A)", 'cmapname':'viridis',
         'vmin':vmin, 'vmax':vmax}
    createmap(A, **args)

    plt.sca(axes[0,1])
    if axeslist[1] is not None:
        plt.sca(axeslist[1])
    args['title']=titles[1] + "(B)"
    createmap(B, **args)

    # Next plot abs/rel differences
    plt.sca(axes[1,0])
    if axeslist[2] is not None:
        plt.sca(axeslist[2])
    args['title']="%s - %s"%("A","B")
    args['ticks']=None # don't make ticks for difference maps
    args['vmin']=amin; args['vmax']=amax
    args['linear']=alinear
    args['cmapname']='bwr'
    createmap(A-B, **args)

    plt.sca(axes[1,1])
    if axeslist[3] is not None:
        plt.sca(axeslist[3])
    args['title']="100*(%s-%s)/%s"%("A", "B", "B")
    args['vmin']=rmin; args['vmax']=rmax
    args['linear']=rlinear
    args['clabel']="%"
    createmap((A-B)*100.0/B, suptitle=suptitle, **args)

    if pname is not None:
        plt.tight_layout()
        plt.savefig(pname)
        print("SAVED FIGURE ",pname)


    if np.sum([a is not None for a in axeslist]) < 1 :
        plt.close(f)

    return (A,B)

def add_grid_to_map(m, xy0=(-181.25,-89.), xyres=(2.5,2.), color='k', linewidth=1.0, dashes=[1000,1], labels=[0,0,0,0],xy1=None):
    '''
    Overlay a grid onto the thingy
    Inputs:
        m: the basemap object to be gridded
        leftbot: [left, bottom]  #as lon,lat
        xyres: lon,lat resolution in degrees
        color: grid colour
        linewidth: of grid lines
        dashes: [on,off] for dash pattern
        label: [left,right,top,bottom] to be labelled
    '''
    if xy1 is None:
        xy1 = (180.0001,90.0001)
    else:
        xy1[0]=xy1[0]+0.0001
        xy1[1]=xy1[1]+0.0001
    # lats
    y=np.arange(xy0[0], xy1[0], xyres[0])
    # lons
    x=np.arange(xy0[1], xy1[1], xyres[1])
    # add grid to map
    m.drawparallels(x, color=color, linewidth=linewidth, dashes=dashes, labels=labels)
    m.drawmeridians(y, color=color, linewidth=linewidth, dashes=dashes, labels=labels)
    #drawmeridians(meridians, color=’k’, linewidth=1.0, zorder=None, dashes=[1, 1], labels=[0, 0, 0, 0], labelstyle=None, fmt=’%g’, xoffset=None, yoffset=None, ax=None, latmax=None, **kwargs)

def displaymap(region=__AUSREGION__,
               subregions=[], labels=[], colors=[], linewidths=[],
               fontsize='small', bluemarble=True,drawstates=True):
    '''
        regions are [lat,lon,lat,lon]
    '''
    m = Basemap(projection='mill', resolution='i',
        llcrnrlon=region[1], llcrnrlat=region[0],
        urcrnrlon=region[3], urcrnrlat=region[2])
    if bluemarble:
        m.bluemarble()
    else:
        m.drawcountries()
    if drawstates:
        m.drawstates()

    # Add lats/lons to map
    add_grid_to_map(m,xy0=(-10,-80),xyres=(10,10),dashes=[1,1e6],labels=[1,0,0,1])
    for r in subregions:
        if len(labels)<len(r):
            labels.append('')
        if len(colors)<len(r):
            colors.append('k')
        if len(linewidths)<len(r):
            linewidths.append(1)
    # add subregions and little lables:
    for r,l,c,lw in zip(subregions, labels, colors, linewidths):
        plot_rec(m,r,color=c, linewidth=lw)
        lon,lat=r[1],r[2]
        x,y = m(lon,lat)
        plt.text(x+100,y-130,l,fontsize=fontsize,color=c)

    return m

def plot_daily_cycle(dates, data, houroffset=0, color='k', overplot=False):
    '''
    Daily cycle from inputs:
        dates: list of datetimes
        data: corresponding data
        houroffset: roll the array for local time matching
    '''
    dates=[d+timedelta(seconds=int(3600*houroffset)) for d in dates]
    d0=dates[0]
    dE=dates[-1]

    n_days=len(util.list_days(d0,dE,month=False))
    hours=np.array([d.hour for d in dates]) # [0, ..., 23, 0, ...]
    days=np.array([d.day for d in dates])   # [0, ...,  0, 1, ...]
    # split data into 24xn_days array
    arr=np.zeros([24, n_days]) + np.NaN
    for i in range(n_days):

        #dinds=np.arange(i*24,(i+1)*24)
        # match hours in this day
        dinds = np.where(days==(i+1))[0]
        dhours = hours[dinds]
        # rotate for nicer view (LOCAL TIME)
        # EG: 11th hour ... 35th hour if houroffset is 11
        #print(dinds)
        arr[:,i]=data[dinds]

        # for now just plot
        print('hours',dhours)
        plt.plot(dhours,data[dinds], color=color)

    return arr
    #plt.ylabel('E_isop_biogenic [kgC/cm2/s]')
    #plt.xlabel('hour(LT)')
    #plt.suptitle(self.dates[0].strftime("%b %Y"))
    #plt.tight_layout()
    #plt.savefig(pname)
    #print("SAVED FIGURE:",pname)

def plot_yearly_cycle(data,dates, **plotargs):
    '''
        Plot a time series wrapped over a year
    '''
    # split data into years
    ydates=util.list_years(dates[0],dates[-1],dates=dates) # split by years
    for year in ydates:
        inds = np.array([ (d >= year[0]) * (d <= year[-1]) for d in dates ], dtype=np.bool)
        doys = [d.timetuple().tm_yday for d in year]
        plt.plot(doys, np.array(data)[inds], **plotargs)
        # only want to label this guy once
        if 'label' in plotargs:
            _label=plotargs.pop('label')
    # looking at year
    plt.xlim([-5,370])
    plt.xlabel('DOY')

# TODO: Rename to add_region_of_dots in other scripts
def add_dots_to_map(bmap, lats, lons, region, cmapname='rainbow',
                    add_rectangle=True, landonly=True,
                    marker='o',markersize=5):
    '''
        Add dots (optionally coloured) to the bmap
        lats, lons will be subset to subregion
    '''


    lati,loni=util.lat_lon_range(lats,lons,region)
    #lats,lons=lats[lati],lons[loni]
    oceanmask=util.oceanmask(lats,lons, inlands=False)

    if add_rectangle:
        # Add rectangle around where we are correlating
        add_rectangle(bmap,region,linewidth=2)

    colors=get_colors(cmapname,len(lati)*len(loni))

    # iterator ii, one for each land grid square we will regress
    ii=0
    for y in lati:
        for x in loni:
            # Don't correlate oceanic squares
            if landonly and oceanmask[y,x]:
                ii=ii+1
                continue

            # Add dot to map
            mx,my = bmap(lons[x], lats[y])
            bmap.plot(mx, my, marker, markersize=markersize, color=colors[ii])
    return colors

def add_dots_to_map(bmap, lats, lons,
                    marker='o', markersize=5,
                    **bmapargs):
    '''
        Add dots (optionally coloured) to the bmap
        lats, lons will be subset to subregion
    '''

    for y,x in zip(lats,lons):

        # Add dot to map
        mx,my = bmap(x, y)

        bmap.plot(mx, my, marker, markersize=markersize,
                  **bmapargs)

def add_marker_to_map(bmap, mask, lats, lons, marker='o', landonly=False, **bmapargs):
    '''
        Add marker (optionally coloured) to the bmap
        lats, lons will be subset to subregion
        bmapargs can include (EG)
            markersize=5, color='r',
    '''

    oceanmask=util.oceanmask(lats,lons, inlands=False)

    for y in range(len(lats)):
        for x in range(len(lons)):
            # Don't do oceanic squares
            if landonly and oceanmask[y,x]:
                continue

            # Add mark to map (if true in mask array)
            if mask[y,x]:
                mx,my = bmap(lons[x], lats[y])
                bmap.plot(mx, my, marker=marker, **bmapargs)

def remove_spines(ax, top=False, right=False, bottom=False, left=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)