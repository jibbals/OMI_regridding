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
import utilities.plotting as pp
from utilities.GC_fio import get_tropchem_data, get_UCX_data
from utilities.JesseRegression import RMA
from classes.GC_class import GC_output

##################
#####GLOBALS######
##################

################
###FUNCTIONS####
################

def compare_tc_ucx():
    '''
        Check UCX vs Tropchem from special 200501 runs
    '''
    # read two files directly
    ucx=get_UCX_data(test=True)
    trp=get_tropchem_data(monthavg=True, test=True)
    
    print("UCX shape: %s"%str(np.shape(ucx['E_isop_bio'])))
    print("Tropchem shape: %s"%str(np.shape(trp['E_isop_bio'])))
    # first compare biogsrce-isop:
    

def compare_tc_ucx_old(date=datetime(2005,1,1)):
    '''
        maps of UCX and tropchem tropospheric columns
        Can also produce similar maps for surface values rather than column amounts
    '''

    # Read UCX and TC datasets
    ucx=GC_output(date,UCX=True)
    tc=GC_output(date,UCX=False)
    lats=tc.lats
    lons=tc.lons

    # Get hcho and isop for comparisons
    data={} # data['tc']['isop'] = molec/cm2 [lat,lon]

    keys=['hcho','isop']
    rlims={'hcho':(-20,20),'isop':(-50,100)}
    for sf in [True,False]:
        if sf: # surface only
            data['tc']=tc.ppbv_to_molec_cm2(keys=keys)
            data['ucx']=ucx.ppbv_to_molec_cm2(keys=keys)
            for run in ['tc','ucx']:
                for k in keys:
                    data[run][k] = data[run][k][0,:] # surface layer only

            alims={'hcho':(1e13,5e15),'isop':(1e9,5e15)}

        else: # whole tropospheric column
            data['tc']  = tc.get_trop_columns(keys=keys)
            data['ucx'] = ucx.get_trop_columns(keys=keys)

            alims={'hcho':(1e14,5e16),'isop':(1e9,5e16)}

        # Mean of the month for tropchem run
        for k in keys:
            data['tc'][k] = np.mean(data['tc'][k], axis=2) # mean of time dim

        for ii,k in enumerate(keys):
            u=data['ucx'][k]; t=data['tc'][k]
            if sf:
                print("Surface only:")
            else:
                print("Troposphere")
            print("%5s   |  UCX          | tropchem"%k)
            print(" mean   | %.3e     |   %.3e"%(np.mean(u),np.mean(t)))
            print(" min    | %.3e     |   %.3e"%(np.min(u),np.min(t)))
            print(" max    | %.3e     |   %.3e"%(np.max(u),np.max(t)))
            print(" std    | %.3e     |   %.3e"%(np.std(u),np.std(t)))


            # Plot values and differences for each key
            units=r'molec cm$^{-2}$'
            pname='Figs/GC/UCX_vs_tropchem_%s_%s_%s'%(['trop','surf'][sf],k,tc.dstr)
            amin,amax = alims[k]
            rmin,rmax = rlims[k]
            f,axes=plt.subplots(2,2,figsize=(16,14))
            plt.sca(axes[0,0])
            pp.createmap(u,lats,lons,aus=True,clabel=units,
                         title="%s UCX"%k, vmin=amin, vmax=amax)
            plt.sca(axes[0,1])
            pp.createmap(t,lats,lons,aus=True,clabel=units,
                         title="%s tropchem"%k, vmin=amin, vmax=amax)
            plt.sca(axes[1,0])
            pp.createmap(u-t,lats,lons,aus=True,clabel=units,
                         title="UCX - tropchem", linear=True)
            plt.sca(axes[1,1])
            pp.createmap((u-t)*100.0/u,lats,lons, vmin=rmin,vmax=rmax,aus=True,
                        linear=True,clabel='%',title="100*(UCX-tc)/UCX",
                        suptitle='%s %s %s'%(['trop','surf'][sf], k, tc.dstr),
                        pname=pname)


def isop_hcho_RMA(gc):
    '''
        compares isop emission [atom_C/cm2/s] against hcho vert_column [molec_hcho/cm2]
        as done in Palmer et al. 2003
        Also plots sample of regressions over Australia
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
        title="HCHO trop column vs isoprene emissions %s"%gc.dstr, pname=pname2)

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
        title='Emissions of isoprene', clabel=r'atom$_C$ cm$^{-2}$ s$^{-1}$',
        pname=pname)

# If this script is run directly:
if __name__=='__main__':
    pp.InitMatplotlib()
    compare_tc_ucx()
    #compare_surface_tc_ucx()
    #compare_tc_ucx()

    # scripts mapping stuff
    date=datetime(2005,1,1)
    #tc=GC_output(date,UCX=False)
    #E_isop_map(tc,aus=True)
    #E_isop_series(tc,aus=True)
    #isop_hcho_RMA(tc)
