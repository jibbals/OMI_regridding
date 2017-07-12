# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Reads ncfiles created by bpch2coards, from tropchem and UCX V10.01 so far.

Currently reads one month at a time... may update to N days

History:
    Created in the summer of '69 by jwg366
    Mon 10/7/17: Added verbose flag and started history.
'''
## Modules
import numpy as np
from datetime import datetime
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm

# Read in including path from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# 'local' modules
import utilities.GC_fio as gcfio
import utilities.utilities as util
from utilities import plotting as pp
from utilities.plotting import __AUSREGION__
from utilities.JesseRegression import RMA
from classes.variable_names_mapped import GC_trac_avg

# remove the parent folder from path. [optional]
sys.path.pop(0)

##################
#####GLOBALS######
##################

__VERBOSE__=True # For file-wide print statements

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
        ''' Read data for ONE MONTH into self '''
        self.dstr=date.strftime("%Y%m")

        # READ DATA, Tropchem or UCX file
        self.UCX=UCX
        read_file_func   = gcfio.get_tropchem_data
        read_file_params = {'date':date}
        if UCX:
            read_file_func = gcfio.get_UCX_data
        else: # if it's tropchem we want to take month avg:
            read_file_params['monthavg']=True
        
        # Read the file in
        tavg_data=read_file_func(**read_file_params)
        
        # Save the data to this class.
        for key in tavg_data.keys():
            setattr(self, key, tavg_data[key])
            if __VERBOSE__:
                print("GC_output reading %s"%key)
        self.taus=util.gregorian_from_dates([date])
        
        # add some peripheral stuff
        self.n_lats=len(self.lats)
        self.n_lons=len(self.lons)

        # set dates and E_dates:
        self.dates=util.date_from_gregorian(self.taus)

    def get_field(self, keys=['hcho', 'E_isop'], region=pp.__AUSREGION__):
        '''
        Return fields subset to a specific region [S W N E]
        '''
        lati,loni = util.lat_lon_range(self.lats,self.lons,region)
        out={'lats':self.lats[lati],'lons':self.lons[loni]}
        for k in keys:
            out[k] = getattr(self, k)
            out[k] = out[k][lati,:]
            out[k] = out[k][:,loni]
        return out

    def model_slope(self, region=pp.__AUSREGION__):
        '''
            Use RMA regression between E_isop and tropcol_HCHO to determine S:
                HCHO = S * E_isop + b
            Note: Slope = Yield_isop / k_hcho

            Return {'lats','lons','r':reg, 'b':bg, 'slope':slope}

        '''
        # if this calc is already done, short cut it
        if hasattr(self, 'modelled_slope'):
            if __VERBOSE__:
                print("Slope has already been modelled, re-returning")
            return self.modelled_slope
            # obj.attr_name exists.
        hcho = self.get_trop_columns(keys=['hcho'])['hcho']
        isop = self.E_isop

        lats,lons = self.lats, self.lons
        lati,loni = util.lat_lon_range(lats,lons,region)

        isop = isop[lati, :]
        isop = isop[:, loni]
        hcho = hcho[lati, :]
        hcho = hcho[:, loni]

        sublats, sublons = lats[lati], lons[loni]
        n_x = len(loni)
        n_y = len(lati)
        slope  = np.zeros([n_y,n_x]) + np.NaN
        bg     = np.zeros([n_y,n_x]) + np.NaN
        reg    = np.zeros([n_y,n_x]) + np.NaN

        for xi in range(n_x):
            for yi in range(n_y):
                # Y = m X + B
                X=isop[yi, xi, :]
                Y=hcho[yi, xi, :]

                # Skip ocean or no emissions squares:
                if np.isclose(np.mean(X),0.0): continue

                # get regression
                m,b,r,CI1,CI2=RMA(X, Y)
                slope[yi,xi] = m
                bg[yi,xi] = b
                reg[yi,xi] = r

        print('model_yield')
        print(np.nanmean(slope))

        # Return all the data and the lats/lons of our subregion:
        self.modelled_slope={'lats':sublats,'lons':sublons,'r':reg, 'b':bg, 'slope':slope}
        return self.modelled_slope


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
    gc=GC_output(datetime(2005,1,1))
    E_isop_hourly=gc.E_isop
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


if __name__=='__main__':
    #check_diag()
    check_units()

