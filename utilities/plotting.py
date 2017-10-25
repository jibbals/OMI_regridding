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
import matplotlib.dates as mdates
#import matplotlib.colors as mcolors #, colormapping
from matplotlib.colors import LogNorm # for lognormal colour bar

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

from utilities.JesseRegression import RMA
from utilities.utilities import regrid
import utilities.fio as fio
sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=True

# S W N E
__AUSREGION__=[-45, 108.75, -7, 156.25] # picked from lons_e and lats_e in GMAO.py
__GLOBALREGION__=[-69, -178.75, 69, 178.75]

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
    matplotlib.rcParams['image.cmap'] = 'PuRd' #'inferno_r'       # Colormap default

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


def createmap(data, lats, lons, edges=False ,
              vmin=None, vmax=None, latlon=True,
              region=__GLOBALREGION__, aus=False, linear=False,
              clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
              pname=None,title=None,suptitle=None, smoothed=False,
              cmapname=None, fillcontinents=None):
    '''
        Pass in data[lati,loni], lats[lati], lons[loni]
    '''
    if __VERBOSE__:
        print("createmap called: %s"%str(title))
        #print("Data %s, %d lats and %d lons"%(str(data.shape),len(lats), len(lons)))

    # Create a basemap map with region as inputted
    if aus: region=__AUSREGION__
    lllat=region[0]; urlat=region[2]; lllon=region[1]; urlon=region[3]
    m=Basemap(llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon,
              resolution='i', projection='merc')

    # Set vmin and vmax if necessary
    if vmin is None:
        vmin=1.05*np.nanmin(data)
    if vmax is None:
        vmax=0.95*np.nanmax(data)

    ## basemap pcolormesh uses data edges
    ##
    lats_e,lons_e=lats,lons
    if not edges:
        if __VERBOSE__: print("Making edges from lat/lon mids")
        nlat,nlon=len(lats), len(lons)
        lats_e=regularbounds(lats)
        lons_e=regularbounds(lons)
        assert nlat == len(lats_e)-1, "regularbounds failed: %d -> %d"%(nlat, len(lats_e))
        assert nlon == len(lons_e)-1, "regularbounds failed: %d -> %d"%(nlon, len(lons_e))

    ## midpoints, derive simply from edges
    lats=(lats_e[0:-1] + lats_e[1:])/2.0
    lons=(lons_e[0:-1] + lons_e[1:])/2.0

    ## interpolate for smoothed output if desired
    ##
    if smoothed:
        factor=5
        if __VERBOSE__: print("Smoothing data, by factor of %d"%factor)
        # 'increase' resolution
        nlats = factor*data.shape[0]
        nlons = factor*data.shape[1]
        lonsi = np.linspace(lons[0],lons[-1],nlons)
        latsi = np.linspace(lats[0],lats[-1],nlats)

        # also increase resolution of our edge lats/lons
        lats_e=regularbounds(latsi);
        lons_e=regularbounds(lonsi)
        lonsi, latsi = np.meshgrid(lonsi, latsi)
        # Smoothe data to increased resolution
        data = interp(data,lons,lats,lonsi,latsi)

    # Make edges into 2D meshed grid
    mlons_e,mlats_e=np.meshgrid(lons_e,lats_e)

    errmsg="pcolormesh needs edges for lat/lon (array: %s, lats:%s)"%(str(np.shape(data)),str(np.shape(mlats_e)))
    assert mlats_e.shape[0] == data.shape[0] + 1, errmsg

    if cmapname is None:
        cmapname = matplotlib.rcParams['image.cmap']

    cmap=plt.cm.cmap_d[cmapname]
    pcmeshargs={'vmin':vmin, 'vmax':vmax, 'clim':(vmin, vmax),
                'latlon':latlon, 'cmap':cmap}

    if not linear:
        if __VERBOSE__:print("createmap() is removing negatives")
        pcmeshargs['norm']=LogNorm()
        data[data<=0]=np.NaN

    #force nan into any pixel with nan results, so color is not plotted there...
    mdata=np.ma.masked_invalid(data) # mask non-finite elements
    #mdata=data # masking occasionally throws up all over your face

    if __VERBOSE__:
        shapes=tuple([ str(np.shape(a)) for a in [mlats_e, mlons_e, mdata, mdata.mask] ])
        print("lats: %s, lons: %s, data: %s, mask: %s"%shapes)

    cs=m.pcolormesh(mlons_e, mlats_e, mdata, **pcmeshargs)
    # colour limits for contour mesh
    cs.set_clim(vmin,vmax)

    # draw coastline and equator(no latlon labels)
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0])

    if fillcontinents is not None:
        m.fillcontinents(fillcontinents)

    # add titles and cbar label
    if title is not None:
        plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    cb=None
    if colorbar:
        cbargs={'size':'5%','pad':'1%','extend':'both','format':cbarfmt}
        cb=m.colorbar(cs,"bottom", **cbargs)
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

def plot_swath(day, region=__AUSREGION__, reprocessed=False,
              edges=False , vmin=None, vmax=None,
              aus=False, linear=False, clabel='molec/cm2', colorbar=True,
              cbarfmt=None, cbarxtickrot=None,
              pname=None,title=None,suptitle=None, smoothed=False,
              cmapname=None, fillcontinents=None):
    '''
        Wrapper to plot gridded swath output for a day
    '''


    #swaths=fio.read_omhcho_day(day)
    dkey=['VC_OMI_RSC','VCC'][reprocessed]
    swaths=fio.read_omhchorp(day,oneday=True,keylist=[dkey,'latitude','longitude'])
    data=swaths[dkey]
    lats=swaths['latitude']
    lons=swaths['longitude']

    return createmap(data, lats, lons, edges=edges ,
              vmin=vmin, vmax=vmax, latlon=True,
              region=region, aus=aus, linear=linear,
              clabel=clabel, colorbar=colorbar, cbarfmt=cbarfmt,
              cbarxtickrot=cbarxtickrot, pname=pname,title=title,
              suptitle=suptitle, smoothed=smoothed,
              cmapname=cmapname, fillcontinents=fillcontinents)


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
    x,y=bmap(xs,ys)
    bmap.plot(x, y, latlon=False, color=color, linewidth=linewidth)

def plot_corellation(X,Y, lims=[1e12,2e17], logscale=True, legend=True,
                     colour='k',linecolour='r', diag=True, oceanmask=None,
                     verbose=False):
    X=np.array(X)
    Y=np.array(Y)
    nans=np.isnan(X) + np.isnan(Y)
    lims0=np.array(lims); lims=np.array(lims)

    if oceanmask is None:
        plt.scatter(X[~nans], Y[~nans])
        m,b,r,CI1,CI2=RMA(X[~nans], Y[~nans]) # get regression
        plt.plot(lims, m*np.array(lims)+b,color=linecolour,
                 label='Y = %.5fX + %.2e, r=%.5f, n=%d'%(m,b,r,np.sum(~nans)))
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
        if m*lims[0] + x0 < 0 and logscale:
            lims[0] = -x0/m + 100
        if (lm*lims[0] + lx0 < 0) and logscale:
            lims[0] = -lx0/lm + 100

        #plot lobf and label
        plt.plot( lims, lm*lims+lx0, color='k', linewidth=2,
                label='Land: Y = %.5fX + %.2e; r=%.5f'%(lm,lx0,lr))
        plt.plot( lims, m*lims+x0, color='blue',
                label='Ocean: Y = %.5fX + %.2e, r=%.5f'%(m,x0,r))

    if verbose:
        print('Land: Y = %.5fX + %.2e; r=%.5f'%(lm,lx0,lr))
        print('with CI ranges of slope %2.5f, %2.5f'%(lci1[0][0],lci1[0][1]))
        print('with CI ranges of intercept %1.5e, %1.5e'%(lci1[1][0],lci1[1][1]))
        print('min, max land X: %.3e,%.3e'%(np.min(X[lmask]),np.max(X[lmask])) )
        print('min, max land Y: %.3e,%.3e'%(np.min(Y[lmask]),np.max(Y[lmask])) )

    if legend:
        plt.legend(loc=2,scatterpoints=1, fontsize=22,frameon=False)
    if logscale:
        plt.yscale('log'); plt.xscale('log')
    plt.ylim(lims0); plt.xlim(lims0)
    if diag:
        plt.plot(lims0,lims0,'--',color=colour,label='1-1') # plot the 1-1 line for comparison

def plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, xtickrot=30, dfmt='%Y%m', xticks=None, **pltargs):
    ''' plot values over datetimes '''
    dates = mdates.date2num(datetimes)
    #plt.plot_date(dates, values, **pltargs)
    plt.plot(dates,values,**pltargs)

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

def compare_maps(datas,lats,lons,pname,titles=['A','B'], suptitle=None,
                 clabel=None, region=__AUSREGION__, vmin=None, vmax=None,
                 rmin=-200.0, rmax=200., amin=None, amax=None, contourf=False,
                 linear=False, alinear=True, rlinear=True, **pltargs):
    '''
        Plot two maps and their relative and absolute differences
    '''
    A=datas[0]
    B=datas[1]
    Alats=lats[0]
    Alons=lons[0]
    Blats=lats[1]
    Blons=lons[1]
    if vmax is None:
        vmax = np.nanmax(np.array(np.nanmax(A),np.nanmax(B)))
    if vmin is None:
        vmin = np.nanmin(np.array(np.nanmin(A),np.nanmin(B)))
    if amax is None:
        amax = vmax
    if amin is None:
        amin = -vmax


    # regrid the lower resolution data
    if len(Alats) > len(Blats):
        B = regrid(B,Blats,Blons,Alats,Alons)
        Blats,Blons=Alats,Alons
    if len(Alats) < len(Blats):
        A = regrid(A,Alats,Alons,Blats,Blons)
        Alats,Alons=Blats,Blons
    lats=Alats
    lons=Alons

    # set up plot
    f,axes=plt.subplots(2,2,figsize=(16,14))
    plt.sca(axes[0,0])
    args={'region':region, 'clabel':clabel, 'linear':linear,
          'lats':lats, 'lons':lons, 'contourf':contourf, 'title':titles[0],
         'vmin':vmin, 'vmax':vmax}
    createmap(A, **args)

    plt.sca(axes[0,1])
    args['title']=titles[1]
    createmap(B, **args)

    plt.sca(axes[1,0])
    args['title']="%s - %s"%(titles[0],titles[1])
    args['vmin']=amin; args['vmax']=amax
    args['linear']=alinear
    createmap(A-B, **args)

    plt.sca(axes[1,1])
    args['title']="100*(%s-%s)/%s"%(titles[0], titles[1], titles[1])
    args['vmin']=rmin; args['vmax']=rmax
    args['linear']=rlinear
    args['clabel']="%"
    createmap((A-B)*100.0/B, suptitle=suptitle, pname=pname, **args)

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

def displaymap(region=__AUSREGION__, subregions=[], labels=[], colors=[],
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

    # add subregions and little lables:
    for r,l,c in zip(subregions, labels,colors):
        plot_rec(m,r,color=c)
        lon,lat=r[1],r[2]
        x,y = m(lon,lat)
        plt.text(x+100,y-100,l,fontsize=fontsize,color=c)

    return m



if __name__=='__main__':
    from datetime import datetime
    plot_swath(datetime(2005,1,20),title="neg_swath",pname="Figs/Checks/neg_swath.png")
