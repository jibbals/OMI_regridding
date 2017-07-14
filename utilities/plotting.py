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
from mpl_toolkits.basemap import Basemap #, maskoceans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors #, colormapping
from matplotlib.colors import LogNorm # for lognormal colour bar

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

from utilities.JesseRegression import RMA
from utilities.utilities import regrid
sys.path.pop(0)

###############
### GLOBALS ###
###############

__VERBOSE__=False

# S W N E
__AUSREGION__=[-47,106,-5,158,]
__GLOBALREGION__=[-80, -179, 80, 179]

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
    matplotlib.rcParams['image.cmap'] = 'Inferno'       # Colormap default

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
              region=__GLOBALREGION__, aus=False, linear=False,
              clabel=None, colorbar=True, left='white', right='pink',
              pname=None,title=None,suptitle=None, contourf=False,
              cmap=None):
    if __VERBOSE__:
        print("createmap called")
        print("Data %s, %d lats and %d lons"%(str(data.shape),len(lats), len(lons)))

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
    #force nan into any pixel with nan results, so color is not plotted there...
    nans=np.isnan(data)
    lonsnew[nans]=np.NaN
    latsnew[nans]=np.NaN

    pcmeshargs={'latlon':latlon, 'vmin':vmin, 'vmax':vmax,
                'clim':(vmin,vmax), 'cmap':cmap}
    if not linear:
        pcmeshargs['norm']=LogNorm()

    if contourf:
        locator=None
        if not linear:
            locator=ticker.LogLocator()
        cs=m.contourf(lonsnew, latsnew, data, locator=locator, **pcmeshargs)
    else:
        cs=m.pcolormesh(lonsnew, latsnew, data, **pcmeshargs)


    # colour limits for contour mesh
    cs.cmap.set_under(left)
    cs.cmap.set_over(right)
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
