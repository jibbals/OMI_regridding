# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# module for hdf eos 5
#import h5py
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# local imports:
import plotting as pp
from Data.GC_fio import get_tropchem_data, get_UCX_data
from GC_class import GC_output
from JesseRegression import RMA

##################
#####GLOBALS######
##################

################
###FUNCTIONS####
################

def compare_tropcol_tc_ucx(date=datetime(2005,1,1)):
    ''' maps of UCX and tropchem tropospheric columns '''

    # Read UCX and TC datasets
    ucx=GC_output(date,UCX=True)
    tc=GC_output(date,UCX=False)
    lats=tc.lats
    lons=tc.lons

    # Get hcho and isop for comparisons
    keys=['hcho','isop']
    tc_trop_cols=tc.get_trop_columns(keys=keys)
    for k in keys:
        tc_trop_cols[k] = np.mean(tc_trop_cols[k], axis=2) # mean of last dim
    ucx_trop_cols=ucx.get_trop_columns(keys=keys)

    # Set up limits
    rlims={'hcho':(-20,20),'isop':(-50,100)}
    alims={'hcho':(1e14,5e16),'isop':(1e9,5e16)}

    for ii,k in enumerate(keys):
        u=ucx_trop_cols[k]; t=tc_trop_cols[k]
        print("%5s   |  UCX          | tropchem"%k)
        print(" mean   | %.3e     |   %.3e"%(np.mean(u),np.mean(t)))
        print(" min    | %.3e     |   %.3e"%(np.min(u),np.min(t)))
        print(" max    | %.3e     |   %.3e"%(np.max(u),np.max(t)))
        print(" std    | %.3e     |   %.3e"%(np.std(u),np.std(t)))

        # Plot trop col differences for hcho
        amin,amax = alims[k]
        rmin,rmax = rlims[k]
        f,axes=plt.subplots(2,2,figsize=(16,14))
        plt.sca(axes[0,0])
        pp.createmap(u,lats,lons,aus=True,clabel='molec/cm2',
                     ptitle="%s UCX"%k, vmin=amin, vmax=amax)
        plt.sca(axes[0,1])
        pp.createmap(t,lats,lons,aus=True,clabel='molec/cm2',
                     ptitle="%s trop"%k, vmin=amin, vmax=amax)
        plt.sca(axes[1,0])
        pp.createmap(u-t,lats,lons,aus=True,clabel='molec/cm2',
                     ptitle="UCX - trop", linear=True)
        plt.sca(axes[1,1])
        pp.createmap((u-t)*100.0/u,lats,lons, vmin=rmin,vmax=rmax,aus=True,
                    linear=True,clabel='%',ptitle="100*(UCX-trop)/UCX",
                    pname='Figs/GC/UCX_vs_tropchem_%s_%s'%(k,tc.dstr))

def compare_surface_tc_ucx(date=datetime(2005,1,1)):
    ''' maps of UCX and tropchem surface HCHO'''
    ausregion=pp.__AUSREGION__ # [S W N E]
    dstr=date.strftime("%Y%m%d")

    # First get tropchem data:
    #
    tdat=get_tropchem_data(date=date,monthavg=True,surface=True)
    thcho=tdat['IJ-AVG-$CH2O']
    tlat=tdat['latitude']
    tlon=tdat['longitude']
    # determine min and max:
    tvmin,tvmax = np.nanmin(thcho), np.nanmax(thcho)
    print("Global tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))
    tvmin,tvmax = pp.findrange(thcho,tlat,tlon, ausregion)
    print("Aus tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))

    # Then get UCX data:
    #
    udat=get_UCX_data(date=date,surface=True)
    uhcho=udat['IJ_AVG_S__CH2O']
    ulat=udat['lat']
    ulon=udat['lon']
    assert (np.array_equal(ulat,tlat)) and (np.array_equal(ulon,tlon)), "LATS AND LONS DIFFER"

    # determine min and max:
    uvmin,uvmax = np.nanmin(uhcho), np.nanmax(uhcho)
    print("Global UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    uvmin,uvmax = pp.findrange(uhcho,ulat,ulon, ausregion)
    print("Aus UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    vmin,vmax=np.min([uvmin,tvmin]),np.max([uvmax,tvmax])

    # Figures with 4 subplots
    f,axes=plt.subplots(2,2,figsize=(14,14))
    kwargs={'vmin':vmin,'vmax':vmax,'linear':True, 'aus':True}
    # first is tropchem
    plt.sca(axes[0,0])
    m,cs,cb=pp.createmap(thcho,tlat,tlon, **kwargs)
    plt.title('tropchem surface')
    cb.set_label('ppbv')

    # second is UCX
    plt.sca(axes[0,1])
    m,cs,cb=pp.createmap(uhcho,ulat,ulon, **kwargs)
    plt.title('UCX surface')
    cb.set_label('ppbv')

    # Third is diffs:
    plt.sca(axes[1,0])
    m,cs,cb = pp.createmap(uhcho-thcho, tlat, tlon, **kwargs)
    plt.title('UCX - tropchem')
    cb.set_label('ppbv')

    # Fourth is rel diff:
    plt.sca(axes[1,1])
    kwargs['vmin']=-10; kwargs['vmax']=10
    m,cs,cb = pp.createmap((uhcho-thcho)/thcho*100, tlat, tlon, **kwargs)
    plt.title('100*(UCX - tropchem)/tropchem')
    cb.set_label('% difference')


    pname='Figs/GC/tropchem_hcho_%s.png'%dstr
    plt.suptitle('HCHO %s'%dstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close()


def isop_hcho_RMA(gc):
    '''
        compares isop emission [atom_C/cm2/s] against hcho vert_column [molec_hcho/cm2]
        as done in Palmer et al. 2003
    '''

    # Retrieve data
    isop = gc.E_isop # atom_C/cm2/s
    hcho = gc.get_trop_columns(keys=['hcho'])['hcho'] # molec/cm2
    dates= gc.dates # datetime list

    # plot name
    pname1='Figs/GC/E_isop_vs_hcho_series_%s.png'%gc.dstr
    pname2='Figs/GC/E_isop_vs_hcho_map_%s.png'%gc.dstr

    # subset region
    region=gc._get_region(aus=True, region=None)
    lats,lons=gc.lats,gc.lons
    lati,loni = pp.lat_lon_range(lats,lons,region)
    isop_sub=isop[lati,:,:]
    isop_sub=isop_sub[:,loni,:]
    hcho_sub=hcho[lati,:,:]
    hcho_sub=hcho_sub[:,loni,:]

    f,axes=plt.subplots(2,1,figsize=(10,14))
    # plot time series (spatially averaged)
    ts_isop=np.mean(isop_sub,axis=(0,1))
    ts_hcho=np.mean(hcho_sub,axis=(0,1))
    plt.sca(axes[0])
    pp.plot_time_series(dates,ts_isop,ylabel=r'E$_{isoprene}$ [atom$_C$ cm$^{-2}$ s$^{-1}$]',
        title='time series for %s'%gc.dstr, dfmt="%m%d", color='r')
    twinx=axes[0].twinx()
    plt.sca(twinx)
    pp.plot_time_series(dates,ts_hcho,ylabel=r'$\Omega_{HCHO}$ [ molec$_{HCHO}$ cm$^{-2}$ ]',
        xlabel='time', dfmt="%m%d", color='m')

    plt.sca(axes[1])
    plt.autoscale(True)
    # plot a sample of ii_max scatter plots and their regressions
    ii=0; ii_max=9
    colours=[cm.rainbow(i) for i in np.linspace(0, 0.9, ii_max)]
    randlatis= np.random.choice(lati, size=30)
    randlonis= np.random.choice(loni, size=30)
    # loop over random lats and lons
    for xi,yi in zip(randlonis,randlatis):
        if ii==ii_max: break
        lat=gc.lats[yi]; lon=gc.lons[xi]
        X=isop[yi,xi,:]; Y=hcho[yi,xi,:]
        if np.isclose(np.mean(X),0.0): continue
        lims=np.array([np.min(X),np.max(X)])
        plt.scatter(X,Y,color=colours[ii])
        m,b,r,CI1,CI2=RMA(X, Y) # get regression
        plt.plot(lims, m*np.array(lims)+b,color=colours[ii],
            label='Y[%5.1fS,%5.1fE] = %.1eX + %.2e, r=%.2f'%(-1*lat,lon,m,b,r))
        ii=ii+1
    plt.xlabel(r'E$_{isop}$ [atom$_C$ cm$^{-2}$ s$^{-1}$]')
    plt.ylabel(r'$\Omega_{HCHO}$ [molec$_{HCHO}$ cm$^{-2}$]')
    plt.title('Sample of regressions over Australia')
    plt.legend(loc=0,fontsize=9) # show legend
    plt.savefig(pname1)
    print("Saved "+pname1)
    plt.close()

    # Now plot the slope and r^2 on the map of Aus:
    fig,axes=plt.subplots(2,1,figsize=(10,14))
    dims=hcho.shape[0:2] # lat,lon dims
    r2=np.zeros(dims)
    slope=np.zeros(dims)
    offset=np.zeros(dims)
    for yi in lati:
        for xi in loni: # loop over lats and lons
            lat=gc.lats[yi]; lon=gc.lons[xi]
            X=isop[yi,xi,:]; Y=hcho[yi,xi,:]
            if np.isclose(np.mean(X),0): # if we're on an all ocean square
                continue
            m,b,r,CI1,CI2=RMA(X, Y) # get regression slope and r values
            r2[yi,xi]=r**2
            slope[yi,xi]=m
            offset=b
    plt.sca(axes[0]) # first plot slope
    vmin=1e-7
    slope[slope < vmin] = np.NaN # Nan the zeros and negatives for nonlinear plot
    pp.createmap(slope, lats, lons, vmin=vmin, aus=True, linear=False,
                clabel=r'm for: $\Omega_{HCHO}$ = m E$_{isop}$ + b')
    plt.sca(axes[1]) # then plot r2 and save figure
    pp.createmap(r2,lats,lons,vmin=0.01,vmax=1.0,aus=True,linear=True,clabel=r'r$^2$',
        ptitle="HCHO trop column vs isoprene emissions %s"%gc.dstr, pname=pname2)

def E_isop_series(gc, aus=False, region=None):
    ''' Plot E_isop time series '''

    data=gc.E_isop
    # subset to region
    reg=gc._get_region(aus,region)
    lati,loni=pp.lat_lon_range(gc.lats,gc.lons,reg)
    data=data[lati,:]
    data=data[:,loni]

    # Average over space lats and lons
    data=np.mean(data, axis=(0,1))

    # plot name
    pname='Figs/GC/E_isop_series_%s%s.png'%(['','aus_'][aus], gc.dstr)

    f=plt.figure(figsize=(10,8))
    pp.plot_time_series(gc.dates, data, xlabel='time', dfmt="%m%d",
        ylabel=r'E$_{isop}$ [atom$_C$ cm$^{-2}$ s$^{-1}$]',
        pname=pname, legend=True, title='Emissions of Isoprene (%s)'%gc.dstr)


def E_isop_map(gc, aus=False, region=None):
    ''' basemap plot of E_isop
        region=[S,W,N,E] boundaries
    '''
    region=gc._get_region(aus, region)
    data=np.mean(gc.E_isop,axis=2) # average over time

    pname='Figs/GC/E_isop_%s%s.png'%(['','aus_'][aus], gc.dstr)
    pp.createmap(data,gc.lats,gc.lons, region=region, vmin=1e10,vmax=1e13,
        ptitle='Emissions of isoprene', clabel=r'atom$_C$ cm$^{-2}$ s$^{-1}$',
        pname=pname)

# If this script is run directly:
if __name__=='__main__':
    pp.InitMatplotlib()
    #compare_surface_tc_ucx()
    compare_tropcol_tc_ucx()

    # scripts mapping stuff
    date=datetime(2005,1,1)
    #tc=GC_output(date,UCX=False)
    #E_isop_map(tc,aus=True)
    #E_isop_series(tc,aus=True)
    #isop_hcho_RMA(tc)
