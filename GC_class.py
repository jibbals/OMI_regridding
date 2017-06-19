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
#from Data.GC_fio import date_from_gregorian, read_tropchem
import Data.GC_fio as gcfio
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

# UCX version:
SimpleUCXnames={'time':'taus','lev':'press','lat':'lats','lon':'lons','IJ_AVG_S__ISOP':'isop',
    'IJ_AVG_S__CH2O':'hcho','PEDGE_S__PSURF':'psurf','BXHGHT_S__BXHEIGHT':'boxH',
    'BXHGHT_S__AD':'AD','BXHGHT_S__AVGW':'avgW','BXHGHT_S__N_AIR_':'N_air',
    'DXYP__':'area','TR_PAUSE__TP_LEVEL':'tplev'}
TracerUCXnames={v: k for k, v in SimpleUCXnames.items()}

################
#####CLASS######
################

class GC_output:
    '''
        Class holding and manipulating tropchem GC output
        # tropchem dims [[lev,] lat, lon[, time]]
        # UCX dims: lev = 72; alt059 = 59; lat = 91; lon = 144;
        self.hcho  = PPBV
        self.isop  = PPBC (=PPBV*5)
        self.boxH  = box heights (m)
        self.psurf = pressure surfaces (hPa)
        self.area  = XY grid area (m2)
        self.N_air = air dens (molec_air/m3)
        self.E_isop= "atoms C/cm2/s" # ONLY tropchem

    '''
    def __init__(self, date, UCX=False):
        ''' Read data for date into self '''
        self.dstr=date.strftime("%Y%m")

        # READ DATA, Tropchem or UCX file
        self.UCX=False
        if UCX:
            self.UCX=True
            tavg_data=gcfio.get_UCX_data(date, keys=SimpleUCXnames.keys(),
                                         surface=False)
            for key,val in SimpleUCXnames.items():
                setattr(self, val, tavg_data[key])

            self.taus=gcfio.gregorian_from_dates([date])

        else: # READ TROPCHEM DATA:
            tavg_file=gcfio.read_tropchem(date)
            # Save data using names mapped from the Simplenames dict:
            for key,val in Simplenames.items():
                # Some tropchem attributes lost the 1e9 scaling
                # during the bpch2coards process
                if key in gcfio.tropchem_scaled_keys:
                    setattr(self, val, tavg_file.variables[key][:]*gcfio.tropchem_scale)
                else:
                    setattr(self, val, tavg_file.variables[key][:])

            #Close the file
            tavg_file.close()

        # add some peripheral stuff
        self.n_lats=len(self.lats)
        self.n_lons=len(self.lons)

        # set dates and E_dates:
        self.dates=gcfio.date_from_gregorian(self.taus)

    def ppbv_to_molec_cm2(self,keys=['hcho'],metres=False):
        ''' return dictionary with data in format molecules/cm2 [or /m2]'''
        out={}
        for k in keys:
            ppbv=getattr(self,k)
            if k=='isop':
                ppbv=ppbv/5.0 # ppb carbon to ppb isoprene

            # ppbv * 1e-9 * molec_air/m3 * m * [m2/cm2]
            scale=[1e-4, 1.0][metres]
            out[k] = ppbv * 1e-9 * self.N_air * self.boxH * scale # molec/area
        return out

    def get_trop_columns(self, keys=['hcho'], metres=False):
        ''' Return tropospheric column amounts in molec/cm2 [or molec/m2] '''
        data={}

        # where is tropopause and how much of next box we want
        trop=np.floor(self.tplev).astype(int)
        extra=self.tplev - trop

        Xdata=self.ppbv_to_molec_cm2(keys=keys,metres=metres)
        # for each key, work out trop columns
        for key in keys:

            dims=np.shape(Xdata[key])
            X=Xdata[key]

            out=np.zeros(dims[1:])
            for lat in range(dims[1]):
                for lon in range(dims[2]):
                    if self.UCX:
                        tropi=trop[lat,lon]
                        out[lat,lon]=np.sum(X[0:tropi,lat,lon])+extra[lat,lon]*X[tropi,lat,lon]
                    else:
                        for t in range(dims[3]):
                            tropi=trop[lat,lon,t]
                            out[lat,lon,t]= np.sum(X[0:tropi,lat,lon,t])+extra[lat,lon,t]*X[tropi,lat,lon,t]

            data[key]=out
        return data

    def month_average(self, keys=['hcho','isop']):
        ''' Average the time dimension '''
        out={}
        if self.UCX: #UCX already month averaged
            for v in keys:
                attr=getattr(self,v)
                out[v]=attr
            return out
        n_t=len(self.taus)
        for v in keys:
            attr=getattr(self, v)
            dims=np.shape(attr)
            if (dims[-1]==n_t) or (dims[-1]==len(self.E_taus)):
                out[v]=np.nanmean(attr, axis=len(dims)-1) # average the last dimension

        return out
    def get_surface(self, keys=['hcho']):
        ''' Get the surface layer'''
        out={}
        for v in keys:
            out[v]=(getattr(self,v))[0]
        return out

    def _get_region(self,aus=False, region=None):
        ''' region for plotting '''
        if region is None:
            region=[-89,-179,89,179]
            if aus:
                region=__AUSREGION__
        return region


################
###FUNCTIONS####
################

def check_units():
    '''
        N_air (molecs/m3)
        boxH (m)
        trop_cols (molecs/cm2)
        E_isop (TODO)

    '''
    N_ave=6.02214086*1e23 # molecs/mol
    airkg= 28.97*1e-3 # ~ kg/mol of dry air
    gc=GC_output(datetime(2005,1,1))
    # N_air is molec/m3 in User manual, and ncfile: check it's sensible:
    nair=np.mean(gc.N_air[0]) # surface only
    airmass=nair/N_ave * airkg  # kg/m3 at surface
    print("Mean surface N_air=%e molec/m3"%nair)
    print(" = %.3e mole/m3, = %4.2f kg/m3"%(nair/N_ave, airmass ))
    assert (airmass > 0.9) and (airmass < 1.5), "surface airmass should be around 1.25kg/m3"

    # Boxheight is in metres in User manual and ncfile: check surface height
    print("Mean surface boxH=%.2fm"%np.mean(gc.boxH[0]))
    assert (np.mean(gc.boxH[0]) > 10) and (np.mean(gc.boxH[0]) < 500), "surface level should be around 100m"

    # Isop is ppbC in manual , with 5 mole C / mole tracer (?), and 12 g/mole
    trop_cols=gc.get_trop_columns(keys=['hcho','isop'])
    trop_isop=trop_cols['isop']
    print("Tropospheric isoprene %s mean = %.2e molec/cm2"%(str(trop_isop.shape),np.nanmean(trop_isop)))
    print("What's expected for this ~1e12?")
    trop_hcho=trop_cols['hcho']
    print("Tropospheric HCHO %s mean = %.2e molec/cm2"%(str(trop_hcho.shape),np.nanmean(trop_hcho)))
    print("What's expected for this ~1e15?")

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
    check_units()

