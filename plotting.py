#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:28:15 2016

Hold functions which will generally plot or print stuff

@author: jesse
"""

import numpy as np
import matplotlib
from mpl_toolkits.basemap import Basemap #, maskoceans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm # for lognormal colour bar
from JesseRegression import RMA
#import matplotlib.patches as mpatches

# S W N E
__AUSREGION__=[-50,100,-5,160,]
__GLOBALREGION__=[-80, -179, 80, 179]

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


def lat_lon_range(lats,lons,region):
    '''
    returns indices of lats, lons which are within region: list=[S,W,N,E]
    '''
    S,W,N,E = region
    lats,lons=np.array(lats),np.array(lons)
    latinds1=np.where(lats>=S)[0]
    latinds2=np.where(lats<=N)[0]
    latinds=np.intersect1d(latinds1,latinds2, assume_unique=True)
    loninds1=np.where(lons>=W)[0]
    loninds2=np.where(lons<=E)[0]
    loninds=np.intersect1d(loninds1,loninds2, assume_unique=True)
    return latinds, loninds

def findvminmax(data,lats,lons,region):
    '''
    return vmin, vmax of data[lats,lons] within region: list=[SWNE]
    '''
    latinds,loninds=lat_lon_range(lats,lons,region)
    data2=data[latinds,:]
    data2=data2[:,loninds]
    vmin=np.nanmin(data2)
    vmax=np.nanmax(data2)
    return vmin,vmax

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

def createmap(data,lats,lons, vmin=None, vmax=None, latlon=True,
              region=__GLOBALREGION__, aus=False, colorbar=True, linear=False,
              clabel=None,pname=None,title=None,suptitle=None):
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

    # fix the lats and lons to 2dimensional meshes(if they're not already)
    if len(lats.shape) == 1:
        lonsnew,latsnew=np.meshgrid(lons,lats)
    else:
        latsnew,lonsnew=(lats,lons)

    pcmeshargs={'latlon':latlon, 'vmin':vmin, 'vmax':vmax, 'clim':(vmin,vmax)}
    if not linear:
        pcmeshargs['norm']=LogNorm()

    cs=m.pcolormesh(lonsnew, latsnew, data, **pcmeshargs)
    #, latlon=latlon, vmin=vmin, vmax=vmax, norm=norm, clim=(vmin,vmax))

    # colour limits for contour mesh
    cs.cmap.set_under('white')
    cs.cmap.set_over('pink')
    cs.set_clim(vmin,vmax)

    # draw coastline and equator(no latlon labels)
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0])

    # add title and cbar label
    if title is not None:
        plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)
    cb=None
    if colorbar:
        cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
        if clabel is not None:
            cb.set_label(clabel)

    # if a plot name is given, save and close figure
    if pname is not None:
        plt.savefig(pname)
        print("Saved "+pname)
        plt.close()
        return

    # if no colorbar is wanted then don't return one (can be set externally)
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

def plot_time_series(datetimes,values,ylabel=None,xlabel=None, pname=None, legend=False, title=None, dfmt='%Y%m', **pltargs):
    ''' plot values over datetimes '''
    dates = mdates.date2num(datetimes)
    #plt.plot_date(dates, values, **pltargs)
    plt.plot(dates,values,**pltargs)

    #Handle ticks:
    #plt.gcf().autofmt_xdate()
    plt.xticks(rotation=55)
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
    if pname is not None:
        plt.savefig(pname)
        print('%s saved'%pname)
        plt.close()

