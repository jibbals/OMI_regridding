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

# local modules
from Data.GC_fio import date_from_gregorian, read_tropchem, read_HEMCO_diag
import plotting as pp
from plotting import __AUSREGION__

##################
#####GLOBALS######
##################
# map for gc output names to simple names, and then reversed
Simplenames={'Tau':'tau','Pressure':'press','latitude':'lat','longitude':'lon', 
    'BXHGHT-$BXHEIGHT':'boxH','BXHGHT-$N(AIR)':'N_air','DXYPDXYP':'area',
    'IJ-AVG-$ISOP':'isop', 'IJ-AVG-$CH2O':'hcho', 'PEDGE-$PSURF':'psurf',
    'TR-PAUSETP-LEVEL':'tplev', 'IJ-AVG-$MVK':'mvk', 'IJ-AVG-$MACR':'macr',
    'IJ-AVG-$NO2':'no2','ISOP_BIOG':'E_isop'}
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
        self.E_isop= kg/m2/s
    '''
    def __init__(self, date):
        ''' Read data for date into self '''
        tavg_file=read_tropchem(date)
        diagkey='ISOP_BIOG'
        # Save all keys in Simplenames dict to class using associated simple names:
        for key,val in Simplenames.items():
            # one tracer is read from combined hemco_diags:
            if key==diagkey: continue
            setattr(self, val, tavg_file.variables[key][:])
        tavg_file.close()
        
        # Save the hemco_diag E_isop:
        diagfile=read_HEMCO_diag(date)
        e_isop=diagfile.variables[diagkey][:]
        e_tau=diagfile.variables['time'][:]
        # move time dimension to the end: making it [lat,lon,time]
        e_isop=np.rollaxis(e_isop,0,2)
        setattr(self, 'E_tau', e_time) # emissions are saved hourly (not daily)
        setattr(self, Simplenames[diagkey], e_isop)
        diagfile.close()
        
        # add some peripheral stuff
        self.n_lats=len(self.lat)
        self.n_lons=len(self.lon)
        
        # set dates and E_dates:
        self.dates=date_from_gregorian(self.tau)
        self.e_dates=date_from_gregorian(self.E_tau)
        
        # set tropospheric columns for HCHO:
        # self.trop_hcho=(self.get_trop_columns(keys=['hcho']))['hcho']
        
    def get_daily_E_isop(self):
        ''' Return daily averaged E_isop '''
        print("Test: running daily_E_isop")
        days= np.array([ d.day for d in self.e_dates ])
        print(days.shape)
        lastday=np.max(days)
        print(lastday)
        e_isop=np.zeros((n_lats,n_lons,lastday))
        print(e_isop.shape)
        
        for i in range(lastday):
            dayinds=np.where(days==i+1)[0]
            print(dayinds.shape)
            e_isop[:,:,i] = np.mean(self.E_isop[:,:,dayinds],axis=2)
        return e_isop
    
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
        n_t=len(self.tau)
        out={}
        for v in keys:
            attr=getattr(self, v)
            dims=np.shape(attr)
            if (dims[-1]==n_t) or (dims[-1]==len(self.E_tau)):
                out[v]=np.nanmean(attr, axis=len(dims)-1) # average the last dimension
        
        return out
    
    def plot_map_E_isop(self, region=None):
        ''' basemap plot of E_isop '''
        
        if region is None:
            
        
    
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

def check_diag():
    '''
    '''
    gc=GC_tropchem(datetime(2005,1,1))
    E_isop_hourly=gc.E_isop
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


if __name__=='__main__':
    check_diag()
    #check_units()
