# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Reads ncfiles created by bpch2coards, from tropchem and UCX V10.01 so far.

Currently reads one month at a time... may update to N days

History:
    Created in the summer of '69 by jwg366
    Mon 10/7/17: Added verbose flag and started history.
'''
###############
### Modules ###
###############

import numpy as np
from datetime import datetime
from scipy.constants import N_A as N_avegadro
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
import utilities.GMAO as gmao
#from classes.variable_names_mapped import GC_trac_avg

# remove the parent folder from path. [optional]
sys.path.pop(0)

##################
#####GLOBALS######
##################

__VERBOSE__=True # For file-wide print statements

# MAP GC TAVG output to nicer names:
_iga='IJ-AVG-$_'
_bxh='BXHGHT-$_'
_GC_names_to_nice = { 'time':'time','lev':'press','lat':'lats','lon':'lons',
    # IJ_AVGs: in ppbv, except isop (ppbC)
    _iga+'NO':'NO', _iga+'O3':'O3', _iga+'MVK':'MVK', _iga+'MACR':'MACR',
    _iga+'ISOPN':'isopn', _iga+'IEPOX':'iepox', _iga+'NO2':'NO2', _iga+'NO3':'NO3',
    _iga+'NO2':'NO2', _iga+'ISOP':'isop', _iga+'CH2O':'hcho',
    # Biogenic sources: atoms C/cm2/s
    'BIOGSRCE_ISOP':'E_isop_bio',
    # burning sources: atoms C/cm2/s
    'BIOBSRCE_CH2O':'E_hcho_burn',
    # Other diagnostics:
    'PEDGE-$_PSURF':'psurf',
    _bxh+'BXHEIGHT':'boxH', # metres
    _bxh+'AD':'AD', # air mass in grid box, kg
    _bxh+'AVGW':'avgW', # Mixing ratio of H2O vapor, v/v
    _bxh+'N(AIR)':'N_air', # Air density: molec/m3
    'DXYP_DXYP':'area', # gridbox surface area: m2
    'TR-PAUSE_TP-LEVEL':'tplev',
    'TR-PAUSE_TPLEV':'tplev', # this one is added to satellite output manually
    'TR-PAUSE_TP-HGHT':'tpH', # trop height: km
    'TR-PAUSE_TP-PRESS':'tpP', # trop Pressure: mb
    # Many more in trac_avg_yyyymm.nc, not read here yet...
    'CHEM-L=$_OH':'OH', # OH molec/cm3: (time, alt059, lat, lon) : 'chemically produced OH'
    }

################
#####CLASS######
################

class GC_common:
    '''
        Class for GEOS-Chem output, inherited by tavg and sat
    '''
    def __init__(self, date, keys, run='tropchem'):
        ''' Read data for ONE MONTH into self
            run= 'tropchem'|'halfisop'|'UCX'
        '''
        self.dstr=date.strftime("%Y%m")

        # Initialise to zeros:
        self.run=run
        #self.hcho  = 0      #PPBV
        #self.isop  = 0      #PPBC (=PPBV*5)
        #self.O_hcho= 0      # column hcho molec/cm2
        #self.boxH  = 0      #box heights (m)
        #self.psurf = 0      #pressure surfaces (hPa)
        #self.area  = 0      #XY grid area (m2)
        #self.N_air = 0      #air dens (molec_air/m3)
        #self.E_isop_bio = 0 #"atoms C/cm2/s" # ONLY tropchem
        self.attrs={}       # Attributes from bpch file

        data,attrs = gcfio.read_tavg(date,run=run,keys=keys)

        # Data has shape like [[time,]lon,lat[,lev]]

        # Save the data to this class.
        for key in data.keys():
            if key in _GC_names_to_nice.keys():
                setattr(self, _GC_names_to_nice[key], data[key])
                self.attrs[_GC_names_to_nice[key]] = attrs[key]
            else:
                setattr(self, key, data[key])
                self.attrs[key]=attrs[key]

            if __VERBOSE__:
                print("GC_tavg reading %s %s"%(key,data[key].shape))

        # If possible calculate the column hcho too
        # molec/cm2 = ppbv * 1e-9 * molec_A / cm3 * H(cm)
        n_dims=len(np.shape(self.hcho))
        print("n_dims = %d, hcho=%.2e"%(n_dims,np.mean(self.hcho)))
        self.O_hcho = np.sum(self.ppbv_to_molec_cm2(keys=['hcho',])['hcho'],axis=n_dims-1)

        # add some peripheral stuff
        self.n_lats=len(self.lats)
        self.n_lons=len(self.lons)

        # Convert from numpy.datetime64 to datetime
        # '2005-01-01T00:00:00.000000000'
        if not hasattr(self,'time'):
            self.time=[date.strftime("%Y-%m-%dT%H:%M:%S.000000000")]
        self.dates=[datetime.strptime(str(d),'%Y-%m-%dT%H:%M:%S.000000000') for d in self.time]


        #self._has_time_dim = len(self.E_isop_bio.shape) > 2
        self._has_time_dim = len(self.dates) > 1

        # Determine emissions in kg/s from atom_C / cm2 / s
        if isinstance(self.E_isop_bio,type(np.array(0))):
            E=self.E_isop_bio # atom C / cm2 / s
            SA=self.area * 1e-6  # m2 -> km2
            # kg/atom_isop = grams/mole * mole/molec * kg/gram
            kg_per_atom = util.isoprene_grams_per_mole * 1.0/N_avegadro * 1e-3
            #          isop/C * cm2/km2 * km2 * kg/isop
            conversion= 1./5.0 * 1e10 * SA * kg_per_atom
            self.E_isop_bio_kgs=E*conversion

        assert all(self.lats == gmao.lats_m), "LATS DON'T MATCH GMAO 2x25 MIDS"
        self.lats_e=gmao.lats_e
        assert all(self.lons == gmao.lons_m), "LONS DON'T MATCH GMAO 2x25 MIDS"
        self.lons_e=gmao.lons_e

    # Common Functions:
    def date_index(self, date):
        ''' Return index of date '''
        whr=np.where(np.array(self.dates) == date) # returns (matches_array,something)
        return whr[0][0] # We just want the match

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
        '''
            Return tropospheric column amounts in molec/cm2 [or molec/m2]
        '''
        data={}

        # where is tropopause and how much of next box we want
        trop=np.floor(self.tplev).astype(int)
        extra=self.tplev - trop

        Xdata=self.ppbv_to_molec_cm2(keys=keys,metres=metres)
        # for each key, work out trop columns
        for key in keys:
            X=Xdata[key]
            dims=np.array(np.shape(X))
            if __VERBOSE__:
                print("%s has shape %s"%(key,str(dims)))
            # Which index is time,lon,lat,lev?
            timei=0; loni=1; lati=2
            out=np.zeros(dims[[timei,loni,lati]])
            if dims[0] > 40: # no time dimension
                lati=1; loni=0
                out=np.zeros(dims[[loni, lati]])

            for lat in range(dims[lati]): # loop over lats
                for lon in range(dims[loni]): # loop over lons
                    try:
                        if len(dims)==3:
                            tropi=trop[lon,lat]
                            out[lon,lat]=np.sum(X[lon,lat,0:tropi])+extra[lon,lat] * X[lon,lat,tropi]
                        else:
                            for t in range(dims[timei]):
                                tropi=trop[t,lon,lat]
                                out[t,lon,lat] = np.sum(X[t,lon,lat,0:tropi]) + \
                                    extra[t,lon,lat] * X[t,lon,lat,tropi]
                    except IndexError as ie:
                        print((tropi, lat, lon))
                        print("dims: %s"%str(dims))
                        print(np.shape(out))
                        print(np.shape(X))
                        print(np.shape(extra))
                        print(ie)
                        raise(ie)

            data[key]=out
        return data

    def month_average(self, keys=['hcho','isop']):
        ''' Average the time dimension '''
        out={}
        n_t=len(self.dates)
        for v in keys:
            data=getattr(self, v)
            dims=np.shape(data)
            if self._has_time_dim:
                out[v]=np.nanmean(data, axis=0) # average the time dim
            else:
                out[v]=data
        return out
    def get_surface(self, keys=['hcho']):
        ''' Get the surface layer'''
        out={}
        for v in keys:
            if self._has_time_dim:
                out[v]=(getattr(self,v))[:,:,:,0]
            else:
                out[v]=(getattr(self,v))[:,:,0]
        return out

    def _get_region(self,aus=False, region=None):
        ''' region for plotting '''
        if region is None:
            region=[-89,-179,89,179]
            if aus:
                region=__AUSREGION__
        return region

    def get_field(self, keys=['hcho', 'E_isop_bio'], region=pp.__AUSREGION__):
        '''
        Return fields subset to a specific region [S W N E]
        '''
        lati, loni = util.lat_lon_range(self.lats,self.lons,region)
        # TODO use edges from mids function (utils?)
        lati_e = np.append(lati,np.max(lati)+1)
        loni_e = np.append(loni,np.max(loni)+1)
        out={'lats':self.lats[lati],
             'lons':self.lons[loni],
             'lats_e':self.lats_e[lati_e],
             'lons_e':self.lons_e[loni_e],
             'lati':lati,
             'loni':loni}
        # DATA LIKE [[time,]lon,lat[,lev]]
        try:
            for k in keys:
                out[k] = getattr(self, k)
                ndims=len(out[k].shape)
                if ndims==4:
                    out[k] = out[k][:,:,lati,:]
                    out[k] = out[k][:,loni,:,:]
                elif ndims==3:
                    if out[k].shape[0] < 40:
                        out[k] = out[k][:,loni,:]
                        out[k] = out[k][:,:,lati]
                    else:
                        out[k] = out[k][loni,:,:]
                        out[k] = out[k][:,lati,:]
                else:
                    out[k] = out[k][loni,:]
                    out[k] = out[k][:,lati]
        except IndexError as ie:
            print(k)
            print(np.shape(out[k]))
            print(np.shape(lati),np.shape(loni))
            raise ie
        return out

class GC_tavg(GC_common):
    '''
        Class holding and manipulating tropchem GC output
        # tropchem dims [time,lon,lat,lev]
        # UCX dims: lev = 72; alt059 = 59; lat = 91; lon = 144;
        # LATS AND LONS ARE BOX MIDPOINTS
        self.hcho  = PPBV
        self.isop  = PPBC (=PPBV*5)
        self.boxH  = box heights (m)
        self.psurf = pressure surfaces (hPa)
        self.area  = XY grid area (m2)
        self.N_air = air dens (molec_air/m3)
        self.E_isop_bio= "atoms C/cm2/s" # ONLY tropchem

    '''
    def __init__(self,date,keys=gcfio.__tavg_mainkeys__,run='tropchem'):
        # Call GC_common initialiser with tavg_mainkeys and tropchem by default
        super(GC_tavg,self).__init__(date,keys=keys,run=run)

    #TODO: define method to create a GC fire mask
    #def firemask(self,):


class GC_sat(GC_common):
    '''
        Class for reading and manipulating satellite overpass output!
    '''
    def __init__(self,date,keys=gcfio.__sat_mainkeys__,run='tropchem'):
        super(GC_sat,self).__init__(date,keys=keys,run=run)
        # fix TR-PAUSE_TPLEV output:
        if len(self.tplev.shape)==3:
            self.tplev=self.tplev[:,:,0]
        if len(self.tplev.shape)==4:
            self.tplev=self.tplev[:,:,:,0]


    def model_slope(self, region=pp.__AUSREGION__):
        '''
            Use RMA regression between E_isop and tropcol_HCHO to determine S:
                HCHO = S * E_isop + b
            Notes:
                Slope = Yield_isop / k_hcho
                HCHO: molec/cm2
                E_isop: Atom C/cm2/s


            Return {'lats','lons','r':reg, 'b':bg, 'slope':slope}

        '''
        # if this calc is already done, short cut it
        if hasattr(self, 'modelled_slope'):
            # unless we're looking at a new area
            if self.modelled_slope['region'] == region:
                if __VERBOSE__:
                    print("Slope has already been modelled, re-returning")
                return self.modelled_slope

        hcho = self.get_trop_columns(keys=['hcho'])['hcho']
        isop = self.E_isop_bio # Atom C/cm2/s

        lats,lons = self.lats, self.lons
        lati,loni = util.lat_lon_range(lats,lons,region=region)

        isop = isop[:, lati, :]
        isop = isop[:, :, loni]
        hcho = hcho[:, lati, :]
        hcho = hcho[:, :, loni]

        sublats, sublons = lats[lati], lons[loni]
        n_x = len(loni)
        n_y = len(lati)
        slope  = np.zeros([n_y,n_x]) + np.NaN
        bg     = np.zeros([n_y,n_x]) + np.NaN
        reg    = np.zeros([n_y,n_x]) + np.NaN

        for xi in range(n_x):
            for yi in range(n_y):
                # Y = m X + B
                X=isop[:, yi, xi]
                Y=hcho[:, yi, xi]

                # Skip ocean or no emissions squares:
                if np.isclose(np.mean(X), 0.0): continue

                # get regression
                m, b, r, CI1, CI2=RMA(X, Y)
                slope[yi, xi] = m
                bg[yi, xi] = b
                reg[yi, xi] = r

        if __VERBOSE__:
            print('GC_tavg.model_yield() calculates avg. slope of %.2e'%np.nanmean(slope))

        # Return all the data and the lats/lons of our subregion:
                            # lats and lons for slope, (and region for later)
        self.modelled_slope={'lats':sublats,'lons':sublons, 'region':region,
                             # indexes of lats/lons for slope
                             'lati':lati, 'loni':loni,
                             # regression, background, and slope
                             'r':reg, 'b':bg, 'slope':slope}
        return self.modelled_slope
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
    gc=GC_tavg(datetime(2005,1,1))

    #data in form [time,lon,lat,lev]
    gcm=gc.month_average(keys=['hcho','N_air'])
    hcho=gcm['hcho']
    nair=gcm['N_air']

    # Mean surface HCHO in ppbv
    hcho=np.mean(hcho[:,:,0])
    print("Surface HCHO in ppbv: %6.2f"%hcho)

    # N_air is molec/m3 in User manual, and ncfile: check it's sensible:
    nair=np.mean(nair[:,:,0])
    airmass=nair/N_ave * airkg  # kg/m3 at surface
    print("Mean surface N_air=%e molec/m3"%nair)
    print(" = %.3e mole/m3, = %4.2f kg/m3"%(nair/N_ave, airmass ))
    assert (airmass > 0.9) and (airmass < 1.5), "surface airmass should be around 1.25kg/m3"

    # Boxheight is in metres in User manual and ncfile: check surface height
    print("Mean surface boxH=%.2fm"%np.mean(gc.boxH[0]))
    assert (np.mean(gc.boxH[:,:,:,0]) > 10) and (np.mean(gc.boxH[:,:,:,0]) < 500), "surface level should be around 100m"

    # Isop is ppbC in manual , with 5 mole C / mole tracer (?), and 12 g/mole
    trop_cols=gc.get_trop_columns(keys=['hcho','isop'])
    trop_isop=trop_cols['isop']
    print("Tropospheric isoprene %s mean = %.2e molec/cm2"%(str(trop_isop.shape),np.nanmean(trop_isop)))
    print("What's expected for this ~1e12?")
    trop_hcho=trop_cols['hcho']
    print("Tropospheric HCHO %s mean = %.2e molec/cm2"%(str(trop_hcho.shape),np.nanmean(trop_hcho)))
    print("What's expected for this ~1e15?")

    # E_isop_bio is atom_C/cm2/s, around 5e12?

def check_diag():
    '''
    '''
    gc=GC_tavg(datetime(2005,1,1))
    E_isop_hourly=gc.E_isop_bio
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


if __name__=='__main__':
    #check_diag()
    check_units()

