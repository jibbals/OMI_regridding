# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# local modules
from Data.GC_fio import date_from_gregorian, read_tropchem, read_HEMCO_diag
import plotting as pp
from plotting import __AUSREGION__
from JesseRegression import RMA

##################
#####GLOBALS######
##################
# map for gc output names to simple names, and then reversed
Simplenames={'Tau':'taus','Pressure':'press','latitude':'lats','longitude':'lons', 
    'BXHGHT-$BXHEIGHT':'boxH','BXHGHT-$N(AIR)':'N_air','DXYPDXYP':'area',
    'IJ-AVG-$ISOP':'isop', 'IJ-AVG-$CH2O':'hcho', 'PEDGE-$PSURF':'psurf',
    'TR-PAUSETP-LEVEL':'tplev', 'IJ-AVG-$MVK':'mvk', 'IJ-AVG-$MACR':'macr',
    'IJ-AVG-$NO2':'no2','BIOGSRCEISOP':'E_isop'}
    #,'ISOP_BIOG':'E_isop'}
Tracernames={v: k for k, v in Simplenames.items()}
################
###FUNCTIONS####
################

class GC_tropchem:
    '''
        Class holding and manipulating tropchem GC output
        # dimensions = [[lev,] lat, lon[, time]]
        self.hcho  = molec/1e9molecA
        self.isop  =   '       '
        self.boxH  = box heights (m)
        self.psurf = pressure surfaces (hPa)
        self.area  = XY grid area (m2)
        self.N_air = air dens (molec_air/m3)
        self.E_isop= "atoms C/cm2/s"

    '''
    def __init__(self, date):
        ''' Read data for date into self '''
        self.dstr=date.strftime("%Y%m")
        tavg_file=read_tropchem(date)
        # Save all keys in Simplenames dict to class using associated simple names:
        for key,val in Simplenames.items():
            setattr(self, val, tavg_file.variables[key][:])
        tavg_file.close()
        
        # add some peripheral stuff
        self.n_lats=len(self.lats)
        self.n_lons=len(self.lons)
        
        # set dates and E_dates:
        self.dates=date_from_gregorian(self.taus)
        
        # set tropospheric columns for HCHO:
        # self.trop_hcho=(self.get_trop_columns(keys=['hcho']))['hcho']
        
    
    def get_trop_columns(self, keys=['hcho'], metres=False):
        ''' Return tropospheric column amounts in molec/cm2 [or molec/m2] '''
        data={}
        
        # where is tropopause and how much of next box we want
        trop=np.floor(self.tplev).astype(int)
        extra=self.tplev - trop
        
        # for each key, work out trop columns
        for key in keys:
            ppbv=getattr(self,key)
            # if key is Isoprene: we have PPBC instead of PPBV
            if key=='isop':
                ppbv=ppbv/5.0 # convert PPBcarbon to PPBisoprene
            
            dims=np.shape(ppbv)
            
            # ppbv * 1e-9 * molec_air/m3 * m * [m2/cm2]
            scale=[1e-4,1.0][metres]
            X = ppbv * 1e-9 * self.N_air * self.boxH * scale # molec/area
            
            out=np.zeros(dims[1:])
            for lat in range(dims[1]):
                for lon in range(dims[2]):
                    for t in range(dims[3]):
                        tropi=trop[lat,lon,t]
                        out[lat,lon,t]= np.sum(X[0:tropi,lat,lon,t])+extra[lat,lon,t]*X[tropi,lat,lon,t]
            data[key]=out
        return data
    
    def month_average(self, keys=['hcho','isop']):
        ''' Average the time dimension '''
        n_t=len(self.taus)
        out={}
        for v in keys:
            attr=getattr(self, v)
            dims=np.shape(attr)
            if (dims[-1]==n_t) or (dims[-1]==len(self.E_taus)):
                out[v]=np.nanmean(attr, axis=len(dims)-1) # average the last dimension
        
        return out
    def _get_region(self,aus=False, region=None):
        ''' region for plotting '''
        if region is None:
            region=[-89,-179,89,179]
            if aus:
                region=__AUSREGION__
        return region
 
    def plot_RMA_isop_hcho(self,pname=None):
        ''' 
            compares isop emission [atom_C/cm2/s] against hcho vert_column [molec_hcho/cm2]
            as done in Palmer et al. 2003
        '''
        
        # Retrieve data
        isop = self.E_isop # atom_C/cm2/s
        hcho = self.get_trop_columns(keys=['hcho'])['hcho'] # molec/cm2
        dates= self.dates # datetime list
        
        # plot name
        if pname is None:
            pname='Figs/GC/GC_E_isop_vs_hcho_%s.png'%self.dstr
        
        # subset region
        region=self._get_region(aus=True, region=None)
        lati,loni = pp.lat_lon_range(self.lats,self.lons,region)
        isop_sub=isop[lati,:,:]
        isop_sub=isop_sub[:,loni,:]
        hcho_sub=hcho[lati,:,:]
        hcho_sub=hcho_sub[:,loni,:]
        
        f,axes=plt.subplots(2,1,figsize=(10,14))
        # plot time series (spatially averaged)
        ts_isop=np.mean(isop_sub,axis=(0,1))
        ts_hcho=np.mean(hcho_sub,axis=(0,1))
        plt.sca(axes[0])
        pp.plot_time_series(dates,ts_isop,ylabel=r'E_{isoprene} [atom_C cm^{-2} s^{-1}]',title='time series', color='r')
        twinx=axes[0].twinx()
        plt.sca(twinx)
        pp.plot_time_series(dates,ts_hcho,ylabel=r'\Omega_{HCHO} [ molec_{HCHO} cm^{-2} ]', xlabel='time', color='m')
        
        plt.sca(axes[1])
        # plot a sample of 6 scatter plots and their regressions
        ii=0
        colours=[cm.rainbow(i) for i in np.linspace(0, 0.9, 6)] 
        for yi in np.random.choice(lati, size=(3,)):
            for xi in np.random.choice(loni, size=(2,)): # loop over 3 random lats and 2 random lons
                lat=self.lats[yi]; lon=self.lons[xi]
                X=isop[yi,xi,:]; Y=hcho[yi,xi,:]
                lims=np.array([np.min(X),np.max(X)])
                plt.scatter(X,Y,color=colours[ii])
                m,b,r,CI1,CI2=RMA(X, Y) # get regression
                plt.plot(lims, m*np.array(lims)+b,color=colours[ii],
                    label='Y[%5.1fS,%5.1fE] = %.5fX + %.2e, r=%.5f, n=%d'%(-1*lat,lon,m,b,r,len(X)))
                ii=ii+1
        plt.savefig(pname)
        print("Saved "+pname)
        plt.close()
        
    def plot_series_E_isop(self, aus=False, region=None, pname=None):
        ''' Plot E_isop time series '''
        
        # Average over space lats and lons
        data=np.mean(self.E_isop,axis=(0,1))
        print(np.mean(data))
        pp.plot_time_series(self.E_dates,data, xlabel='time',ylabel=r'E_{isoprene} [atom_C cm^{-2} s^{-1}]', pname=pname, legend=True, title='Emissions of Isoprene')
        
    def plot_map_E_isop(self, aus=False, region=None):
        ''' basemap plot of E_isop 
            region=[S,W,N,E] boundaries
        '''
        region=self._get_region(aus, region)
        data=np.mean(self.E_isop,axis=2) # average over time
        
        pname='Figs/GC/E_isop_%s%s.png'%(['','aus_'][aus], self.dstr)
        pp.createmap(data,self.lats,self.lons, vmin=1e10,vmax=1e13, ptitle='Emissions of isoprene',
            clabel='atom_C/cm2/s', pname=pname)
    
def check_units():
    '''
        
    '''
    N_ave=6.02214086*1e23 # molecs/mol
    airkg= 28.97*1e-3 # ~ kg/mol of dry air
    gc=GC_tropchem(datetime(2005,1,1))
    # N_air is molec/m3 in User manual, and ncfile: check it's sensible:
    nair=np.mean(gc.N_air[0]) # surface only
    airmass=nair/N_ave * airkg  # kg/m3 at surface
    print("Mean surface N_air=%e molec/m3"%nair)
    print(" = %.3e mole/m3, = %4.2f kg/m3"%(nair/N_ave, airmass ))
    assert (airmass > 0.9) and (airmass < 1.5), "surface airmass should be around 1.25kg/m3"
     
    # Boxheight is in metres in User manual and ncfile: check surface height
    print("Mean surface boxH=%.4f"%np.mean(gc.boxH[0]))
    assert (np.mean(gc.boxH[0]) > 10) and (np.mean(gc.boxH[0]) < 500), "surface level should be around 100m"
    
    # Isop is ppbC in manual , with 5 mole C / mole tracer (?), and 12 g/mole
    trop_cols=gc.get_trop_columns(keys=['hcho','isop'])
    trop_isop=trop_cols['isop']
    print("Tropospheric isoprene %s mean = %e molec/cm2"%(str(trop_isop.shape),np.nanmean(trop_isop)))
    print("What's expected for this?")
    trop_hcho=trop_cols['hcho']
    print("Tropospheric HCHO %s mean = %e molec/cm2"%(str(trop_hcho.shape),np.nanmean(trop_hcho)))
    print("What's expected for this?")
    
    # E_isop is atom_C/cm2/s, around 5e12?

def check_diag():
    '''
    '''
    gc=GC_tropchem(datetime(2005,1,1))
    E_isop_hourly=gc.E_isop
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


if __name__=='__main__':
    #check_diag()
    #check_units()
    gc=GC_tropchem(datetime(2005,1,1))
    gc.plot_RMA_isop_hcho()
    gc.plot_map_E_isop()
