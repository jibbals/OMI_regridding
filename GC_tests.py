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
from classes.gchcho import gchcho

##################
#####GLOBALS######
##################

################
###FUNCTIONS####
################

def check_smearing():
    '''
        Calculate smearing using halfisoprene and tropchem trac_avg files..
    '''
    print("TODO")

def check_tropchem_monthly():
    '''
        see if monthly averaged output matches month of averaged daily averages
    '''
    # Read month of daily averages
    trop_mavg_fromdays=get_tropchem_data(date=datetime(2005,1,1),runtype='tropchem',monthavg=True)
    
    # read monthly average:
    trop_mavg=get_tropchem_data(date=datetime(2005,1,1),runtype='tropchem', fname='trac_avg_200501_month.nc')
    
    print("Keys in daily output")
    print(trop_mavg_fromdays.keys())
    print("Keys in monthly output")
    print(trop_mavg.keys())
    
    for key in ['E_isop_bio','hcho','isop','N_air']:
        print(key)
        print('from days, from monthly')
        mavg=trop_mavg[key] ;davg=trop_mavg_fromdays[key]
        print(davg.shape,mavg.shape)
        print(np.nanmean(davg),np.nanmean(mavg))
    
    # Also compare against UCX month average?

def compare_tc_ucx(date=datetime(2005,1,1),extra=False,fnames=None,suffix=None):
    '''
        Check UCX vs Tropchem
        set Extra to true to look at E_isop_biog and OH (uses specific output file 20050101
        set fnames=[tropchem.nc,ucx.nc] outputs to test specific files
    '''
    ymstr=date.strftime("%Y%m")
    region=pp.__GLOBALREGION__
    
    # Read the netcdf files (output specified for this test)
    # UCX FILE:
    #ucx=GC_output(date,UCX=True, fname='UCX_trac_avg_20050101.nc')
    if fnames is None:
        ucx=GC_output(date,UCX=True)
        trp=GC_output(date,UCX=False,monthavg=True)
    else:
        trp_fname,ucx_fname=fnames#
        trp=GC_output(date,UCX=False, monthavg=True, fname=trp_fname)
        ucx=GC_output(date,UCX=True, fname=ucx_fname)
    
    lats_all=ucx.lats
    lons_all=ucx.lons

    all_keys=['hcho','isop','E_isop_bio','OH','O3','NO2']
    keys=[]
    for key in all_keys:
        if hasattr(ucx,key) and hasattr(trp,key): # look at fields in both outputs
            keys.append(key)
            print('%s will be compared at surface '%(key))
    #keys=['hcho','isop']
    units={ 'hcho'      :'ppbv',        #r'molec cm$^{-2}$',
            'E_isop_bio':r'atom C cm$^{-2}$ s$^{-1}$',
            'OH'        :r'molec cm$^{-3}$',
            'isop'      :r'ppbv',
            'O3'        :r'ppbv',
            'NO2'       :r'ppbv',}
    cbarfmt={}; cbarxtickrot={}
    rlims={'hcho'       :(-20,20),   
           'E_isop_bio' :(-50,50),
           'OH'         :(-50,50),
           'isop'       :(-70,70),
           'O3'         :(-50,50),
           'NO2'        :(-50,50),}
    dlims={'hcho'       :(-.5,.5),
           'E_isop_bio' :(-1e12,1e12),
           'OH'         :(-8e5,8e5),
           'isop'       :(-6,6),
           'O3'         :(-40,40),
           'NO2'        :(-100,100),}
    alims={'hcho'       :(None,None),
           'E_isop_bio' :(None,None),
           'OH'         :(1e6,4e6),
           'isop'       :(None,None),
           'O3'         :(None,None),
           'NO2'        :(None,None),}
    for key in keys:
        cbarfmt[key]=None; cbarxtickrot[key]=None
    cbarfmt['OH']="%.1e"; cbarxtickrot['OH']=30
    
    ucx_data=ucx.get_field(keys=keys,region=region)
    trp_data=trp.get_field(keys=keys,region=region)
    lats=ucx_data['lats'];lons=ucx_data['lons']

    print("Region: ",region)
    for key in keys:
        print(key)
        dats=[ucx_data[key], trp_data[key]]
        for di,dat in enumerate(dats):
            pre=['UCX  ','trop '][di]
            print(pre+"%s shape: %s"%(key,str(np.shape(dat))))
            print("    min/mean/max: %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
            # Just look at surface from here
            if len(dat.shape)==3:
                dat=dat[0]
                print("    at surface  : %.1e/ %.1e /%.1e"%(np.min(dat),np.mean(dat),np.max(dat)))
                dats[di]=dat

        # whole tropospheric column
        #data['tc']  = tc.get_trop_columns(keys=keys)
        #data['ucx'] = ucx.get_trop_columns(keys=keys)

        # Plot values and differences for each key
        suffix=[suffix,''][suffix is None]
        pname='Figs/GC/UCX_vs_trp_glob_%s_%s%s.png'%(trp.dstr,key,suffix)
        u=dats[0];t=dats[1]
        amin,amax = alims[key]
        rmin,rmax = rlims[key]
        dmin,dmax = dlims[key]
        args={'region':region,'clabel':units[key], 'vmin':amin, 'vmax':amax,
              'linear':True, 'cmapname':'PuRd', 'cbarfmt':cbarfmt[key],
              'cbarxtickrot':cbarxtickrot[key]}

        f,axes=plt.subplots(2,2,figsize=(16,14))

        plt.sca(axes[0,0])
        pp.createmap(u,lats,lons, title="%s UCX"%key, **args)

        plt.sca(axes[0,1])
        pp.createmap(t,lats,lons, title="%s tropchem"%key, **args)

        plt.sca(axes[1,0])
        args['vmin']=dmin; args['vmax']=dmax
        args['cmapname']='coolwarm'
        pp.createmap(u-t,lats,lons,title="UCX - tropchem", **args)

        plt.sca(axes[1,1])
        args['vmin']=rmin; args['vmax']=rmax; args['clabel']='%'
        pp.createmap((u-t)*100.0/u, lats, lons, title="100*(UCX-tc)/UCX",
                     suptitle='%s %s %s'%('surface', key, trp.dstr), pname=pname, **args)



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

def check_shapefactors(date=datetime(2005,1,1)):
    '''
        Check shapefactors for shapefactor nc files
    '''
    sf=gchcho(date)
    def check_column(key):
        blah=getattr(sf,key)
        print("%s: %s : %.2e"%(key, blah.shape, np.nanmean(blah)))
        while len(blah.shape) > 1:
            blah=blah[:,0]
        print(blah)
    #print("Shape: ", sf.Shape_s.shape, np.nanmean(sf.Shape_s))
    #print("N_HCHO: ", sf.N_HCHO.shape, np.nanmean(sf.N_HCHO))
    #print("pmids: ", sf.pmids.shape)
    for key in ['Shape_s','N_HCHO','zmids','boxH','sigmas','pmids','pedges']:
        check_column(key)

# If this script is run directly:
if __name__=='__main__':
    pp.InitMatplotlib()
    #check_shapefactors()
    #check_tropchem_monthly()
    
    # Compare explicit dates:
    for cdate in [ datetime(2004,7,1), ]:
        yymm=cdate.strftime("%Y%m")
        fnames= [ "trac_avg_%s.nc"%yymm, "trac_avg_UCX_%s.nc"%yymm] 
        compare_tc_ucx(cdate,fnames=fnames)
    
    #compare_tc_ucx(datetime(2005,1,1),
    #               fnames=['trac_avg_200501_month.nc','trac_avg_UCX_200501.nc'],
    #               suffix='_rerun')
    
    #compare_surface_tc_ucx()
    #compare_tc_ucx()

    # scripts mapping stuff
    date=datetime(2005,1,1)
    #tc=GC_output(date,UCX=False)
    #E_isop_map(tc,aus=True)
    #E_isop_series(tc,aus=True)
    #isop_hcho_RMA(tc)
