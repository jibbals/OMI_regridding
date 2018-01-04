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
from datetime import datetime, timedelta
from scipy.constants import N_A as N_Avogadro
from glob import glob
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm

# Read in including path from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# 'local' modules
from utilities import GC_fio
import utilities.utilities as util
from utilities import plotting as pp
from utilities.plotting import __AUSREGION__
from utilities.JesseRegression import RMA
import utilities.GMAO as GMAO
#from classes.variable_names_mapped import GC_trac_avg

# remove the parent folder from path. [optional]
sys.path.pop(0)

##################
#####GLOBALS######
##################

__VERBOSE__=True # For file-wide print statements

# filepaths:
rdir='Data/GC_Output/'
sat_path  = {'tropchem':rdir+'geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s.bpch',
             'halfisop':rdir+'geos5_2x25_tropchem_halfisoprene/satellite_output/ts_satellite.%s.bpch',
             'UCX':rdir+'UCX_geos5_2x25/satellite_output/ts_satellite.%s.bpch',
             'biogenic':rdir+'geos5_2x25_tropchem_biogenic/satellite_output/sat_biogenic.%s.bpch',}
tavg_path = {'tropchem':rdir+'geos5_2x25_tropchem/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'halfisop':rdir+'geos5_2x25_tropchem_halfisoprene/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'UCX':rdir+'UCX_geos5_2x25/trac_avg/trac_avg_geos5_2x25_UCX_updated.%s'}

# GC OUTPUT NAMES:
__coords__ = ['lev','lon','lat','time']
__ijavg__  = ['IJ-AVG-$_ISOP',
              'IJ-AVG-$_CH2O',
              'IJ-AVG-$_NO2',       # NO2 in ppbv
              'TIME-SER_AIRDEN',    # Named differently in satellite output
              'CHEM-L=$_OH',]       # OH concentrations?
__emiss__  = ['BIOGSRCE_ISOP',      # biogenic source of isoprene () []
              'BIOBSRCE_CH20',      # biomass burning hcho source () []
              'ANTHSRCE_NO', ]      # anthro source of NO (molec/cm2/s) [t,lat,lon,1]
__other__  = ['PEDGE-$_PSURF',      # pressure at surface of each gridbox (hPa)
              'BXHGHT-$_BXHEIGHT',  # box height (?)
              'BXHGHT-$_AD',        # Air density (?)
              'BXHGHT-$_AVGW',      # water ??
              'BXHGHT-$_N(AIR)',    # air density (?)
              'DXYP_DXYP',          # gridbox horizontal area (m?)
              'TR-PAUSE_TP-LEVEL',  #
              'TR-PAUSE_TPLEV',    # Added satellite output for ppamf
              'DAO-3D-$_TMPU',      # Temperature field (Kelvin)
              'DAO-FLDS_TS', ]      # Surf Temp (Kelvin)
__gc_allkeys__ = __coords__ + __ijavg__ + __emiss__ + __other__


# MAP GC output to nicer names:
_ija='IJ-AVG-$_'
_bxh='BXHGHT-$_'
_GC_names_to_nice = { 'time':'time','lev':'press','lat':'lats','lon':'lons',
    # IJ_AVGs: in ppbv, except isop (ppbC)
    _ija+'NO':'NO', _ija+'O3':'O3', _ija+'MVK':'MVK', _ija+'MACR':'MACR',
    _ija+'ISOPN':'isopn', _ija+'IEPOX':'iepox', _ija+'NO2':'NO2', _ija+'NO3':'NO3',
    _ija+'NO2':'NO2', _ija+'ISOP':'isop', _ija+'CH2O':'hcho',
    # Biogenic sources:
    'BIOGSRCE_ISOP':'E_isop_bio', # atoms C/cm2/s
    'ISOP_BIOG':'E_isop_bio', # kgC/cm2/s (from Hemco_diagnostic output)
    # burning sources: atoms C/cm2/s
    'BIOBSRCE_CH2O':'E_hcho_burn',
    # Other diagnostics:
    'PEDGE-$_PSURF':'psurf',
    _bxh+'BXHEIGHT':'boxH', # metres
    _bxh+'AD':'AD', # air mass in grid box, kg
    _bxh+'AVGW':'avgW', # Mixing ratio of H2O vapor, v/v
    _bxh+'N(AIR)':'N_air', # Air density: molec/m3 from tavg output
    'TIME-SER_AIRDEN':'N_air', # Air density: molec/cm3 from satellite output
    'DXYP_DXYP':'area', # gridbox surface area: m2
    'TR-PAUSE_TP-LEVEL':'tplev',
    'TR-PAUSE_TPLEV':'tplev', # this one is added to satellite output manually
    'TR-PAUSE_TP-HGHT':'tpH', # trop height: km
    'TR-PAUSE_TP-PRESS':'tpP', # trop Pressure: mb
    # Many more in trac_avg_yyyymm.nc, not read here yet...
    'CHEM-L=$_OH':'OH', # OH molec/cm3: (time, alt059, lat, lon) : 'chemically produced OH'
    'DAO-3D-$_TMPU':'temp', # temperature
    'DOA-FLDS_TS':'surftemp', # surface temperature, Kelvin
    }

################
#####CLASS######
################

class GC_base:
    '''
        Class for GEOS-Chem output, inherited by tavg and sat
    '''
    def __init__(self, data,attrs, nlevs=47):
        ''' Read data for ONE MONTH into self
            run= 'tropchem'|'halfisop'|'UCX'
            output= 'satellite'|'tavg'
        '''
        self.ntimes=1
        if 'time' in data:
            self.ntimes=len(data['time'])
        self.nlats=91
        self.nlons=144
        self.nlevs=nlevs

        # Initialise to zeros:
        #self.hcho  = 0      #PPBV
        #self.isop  = 0      #PPBC (=PPBV*5)
        #self.O_hcho= 0      # column hcho molec/cm2
        #self.boxH  = 0      #box heights (m)
        #self.psurf = 0      #pressure surfaces (hPa)
        #self.area  = 0      #XY grid area (m2)
        #self.N_air = 0      #air dens (molec_air/m3)
        #self.E_isop_bio = 0 #"atoms C/cm2/s" # ONLY tropchem
        self.attrs={}       # Attributes from bpch file

        # Data could have any shape, we fix to time,lat,lon,lev
        # for each key in thefile
        for key in data.keys():

            # Make sure array has dims: [[time,]lats,lons[,levs]]
            arr=data[key]
            if len(arr.shape) > 1:
                arr = util.reshape_time_lat_lon_lev(arr,self.ntimes,self.nlats,self.nlons,self.nlevs)

            # Fix air density units to molec/m3, in case they are in molec/cm3
            if key == 'TIME-SER_AIRDEN':
                surf_air=np.nanmean(arr[:,:,:,0])
                if __VERBOSE__:
                    print(key,np.shape(arr),'surface N_air:',surf_air)
                if key in attrs:
                    print("attrs:",attrs[key])
                    if 'unit' in attrs[key]:
                        # we want molec/m3
                        attrs[key]['orig_unit']=attrs[key]['unit']
                else:
                    if __VERBOSE__:
                        print(key,' has no attributes!!, assuming molec/cm3 or molec/m3')
                    attrs[key]={'units':'molec/m3'}

                if surf_air < 1e23: # probably molec/cm3
                    arr=arr*1e6

                attrs[key]['units']='molec/m3'

            nkey=key
            if key in _GC_names_to_nice.keys():
                nkey=_GC_names_to_nice[key]

            setattr(self, nkey, arr)
            # just use 'units' not 'unit'
            if 'unit' in attrs[key]:
                attrs[key]['units'] = attrs[key]['unit'] # save unit to units
                attrs[key].pop('unit',None) # delete 'unit' attribute

            self.attrs[nkey] = attrs[key]



            if __VERBOSE__:
                print("GC_tavg reading %s %s"%(key,data[key].shape))

        # Grab area if file doesn't have it
        if not hasattr(self,'area'):
            self.area=GMAO.area_m2

        # Calculate total columns hcho
        # molec/cm2 = ppbv * 1e-9 * molec_A / cm3 * H(cm)
        if hasattr(self,'hcho'):
            n_dims=len(np.shape(self.hcho))
            print("CHECK:hcho %s, mean = %.2e %s"%(str(np.shape(self.hcho)),np.mean(self.hcho),self.attrs['hcho']['units']))
            self.O_hcho = np.sum(self.ppbv_to_molec_cm2(keys=['hcho',])['hcho'],axis=n_dims-1)
            self.attrs['O_hcho']={'units':'molec/cm2','desc':'Total column HCHO'}
            print("CHECK:O_hcho %s, mean=%.2e %s"%(str(self.O_hcho.shape),np.mean(self.O_hcho),self.attrs['O_hcho']['units']))


        # Convert from numpy.datetime64 to datetime
        if not hasattr(self,'time'):
            self.time=[attrs['init_date'].strftime("%Y-%m-%dT%H:%M:%S.000000000")]
        self.dates=util.datetimes_from_np_datetime64(self.time)
        self.dstr=self.dates[0].strftime("%Y%m")

        # flag to see if class has time dimension
        self._has_time_dim = len(self.dates) > 1

        assert all(self.lats == GMAO.lats_m), "LATS DON'T MATCH GMAO 2x25 MIDS"
        self.lats_e=GMAO.lats_e
        assert all(self.lons == GMAO.lons_m), "LONS DON'T MATCH GMAO 2x25 MIDS"
        self.lons_e=GMAO.lons_e

    # Common Functions:
    def date_index(self, date):
        ''' Return index of date '''
        return util.date_index(date,self.dates)

    def lat_lon_index(self,lat,lon):
        '''  return lati, loni '''
        return util.lat_lon_index(lat,lon,self.lats,self.lons)

    def ppbv_to_molec_cm2(self,keys=['hcho']):
        ''' return dictionary with data in format molecules/cm2 [or /m2]'''
        out={}
        N_air=self.N_air

        for k in keys:
            ppbv=getattr(self,k)
            if k=='isop':
                ppbv=ppbv/5.0 # ppb carbon to ppb isoprene

            # ppbv * 1e-9 * molec_air/m3 * boxH(m) * m2/cm2
            scale=0.0
            out[k] = ppbv * 1e-9 * N_air * self.boxH * 1e-4 # molec/cm2
        return out

    def get_trop_columns(self, keys=['hcho']):
        '''
            Return tropospheric column amounts in molec/cm2
        '''
        if __VERBOSE__:
            print('retrieving trop column for ',keys)
        data={}

        # where is tropopause and how much of next box we want
        trop=np.floor(self.tplev).astype(int)
        extra=self.tplev - trop

        Xdata=self.ppbv_to_molec_cm2(keys=keys)
        # for each key, work out trop columns
        for key in keys:
            X=Xdata[key]
            dims=np.array(np.shape(X))
            if __VERBOSE__:
                print("%s has shape %s"%(key,str(dims)))
            # Which index is time,lat,lon,lev?
            timei=0;lati=1;loni=2
            out=np.zeros(dims[[timei,lati,loni]])
            if dims[0] > 40: # no time dimension
                lati=0; loni=1
                out=np.zeros(dims[[lati, loni]])

            for lat in range(dims[lati]): # loop over lats
                for lon in range(dims[loni]): # loop over lons
                    try:
                        if len(dims)==3:
                            tropi=trop[lat,lon]
                            out[lat,lon]=np.sum(X[lat,lon,0:tropi])+extra[lat,lon] * X[lat,lon,tropi]
                        else:
                            for t in range(dims[timei]):
                                tropi=trop[t,lat,lon]
                                out[t,lat,lon] = np.sum(X[t,lat,lon,0:tropi]) + \
                                    extra[t,lat,lon] * X[t,lat,lon,tropi]
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
        for v in keys:
            data=getattr(self, v)
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
        # DATA LIKE [[time,]lat,lon[,lev]]
        try:
            for k in keys:
                out[k] = getattr(self, k)
                ndims=len(out[k].shape)
                if ndims==4:
                    out[k] = out[k][:,:,loni,:]
                    out[k] = out[k][:,lati,:,:]
                elif ndims==3:
                    if out[k].shape[0] < 40:
                        out[k] = out[k][:,lati,:]
                        out[k] = out[k][:,:,loni]
                    else:
                        out[k] = out[k][lati,:,:]
                        out[k] = out[k][:,loni,:]
                else:
                    out[k] = out[k][lati,:]
                    out[k] = out[k][:,loni]
        except IndexError as ie:
            print(k)
            print(np.shape(out[k]))
            print(np.shape(lati),np.shape(loni))
            raise ie
        return out

    def model_slope(self, region=pp.__AUSREGION__, overpass_hour=13):
        '''
            Use RMA regression between E_isop and tropcol_HCHO to determine S:
                HCHO = S * E_isop + b
            Notes:
                Slope = Yield_isop / k_hcho
                HCHO: molec/cm2
                E_isop: Atom C/cm2/s


            Return {'lats','lons','r':regression, 'b':bg, 'slope':slope}

        '''
        # if this calc is already done, short cut it
        if hasattr(self, 'modelled_slope'):
            # unless we're looking at a new area
            if self.modelled_slope['region'] == region:
                if __VERBOSE__:
                    print("Slope has already been modelled, re-returning")
                return self.modelled_slope

        # Read MEGAN outputs from biogenic run:
        megan=Hemco_diag(self.dates[0], month=True)
        isop = megan.daily_LT_averaged(hour=overpass_hour) # lat/lon kgC/cm2/s [days,lat,lon]
        # use hcho from calling class
        hcho = self.get_trop_columns(keys=['hcho'])['hcho']
        # used to be atom C/cm2/s

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

    def plot_hcho_columns(self, region=pp.__AUSREGION__):
        ''' plot map of hcho columns over a region '''
        data=self.O_hcho

        data=self.time_averaged(day0,dayn,keys=[key,],)[key]
        lati,loni=util.lat_lon_range(self.lats,self.lons,region=region)
        data=data[lati,:]
        data=data[:,loni]
        lats=self.lats[lati]
        lons=self.lons[loni]

    def _get_region(self,aus=False, region=None):
        ''' region for plotting '''
        if region is None:
            region=[-89,-179,89,179]
            if aus:
                region=__AUSREGION__
        return region

    def _set_E_isop_bio_kgs(self):
        # Determine emissions in kg/s from atom_C / cm2 / s
        E=self.E_isop_bio * 0.2 # atom C -> atom isop
        gpm=util.__grams_per_mole__['isop']
        SA=self.area * 1e-6  # m2 -> km2
        # kg/atom_isop = grams/mole * mole/molec * kg/gram
        kg_per_atom = gpm * 1.0/N_Avogadro * 1e-3
        # cm2/km2 * km2 * kg/isop
        conversion= 1e10 * SA * kg_per_atom

        if self._has_time_dim:
            self.E_isop_bio_kgs = np.zeros(np.shape(E))
            for t in range(len(self.dates)):
                self.E_isop_bio_kgs[t]=E[t] * conversion
        else:
            self.E_isop_bio_kgs=E * conversion
        self.attrs['E_isop_bio_kgs']={'units':'kg/s',}

class GC_tavg(GC_base):
    '''
        Class holding and manipulating tropchem GC output
        # tropchem dims [time,lat,lon,lev]
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
    def __init__(self,date,keys=__gc_allkeys__,run='tropchem',nlevs=47):
        # Call GC_base initialiser with tavg_mainkeys and tropchem by default

        # Determine path of file:
        dstr = date.strftime("%Y%m%d0000")
        path = tavg_path[run]%dstr

        # read data/attrs and initialise class:
        data,attrs = GC_fio.read_bpch(path,keys=keys,multi='*' in path)
        attrs['init_date']=date

        super(GC_tavg,self).__init__(data,attrs,nlevs=nlevs)

        # add E_isop_bio_kgs:
        if hasattr(self,'E_isop_bio'):
            self._set_E_isop_bio_kgs()


class GC_sat(GC_base):
    '''
        Class for reading and manipulating satellite overpass output!
    '''
    def __init__(self,date, keys=__gc_allkeys__, run='tropchem',nlevs=47):

        # Determine path of files:
        dstr=date.strftime("%Y%m*")
        path=sat_path[run]%dstr

        # read data/attrs and initialise class:
        data,attrs = GC_fio.read_bpch(path,keys=keys,multi='*' in path)

        # may need to handle missing time dim...
        if not 'time' in data:
            tmp=data['IJ-AVG-$_CH2O']
            dates=[date]
            times=util.datetimes_from_np_datetime64(dates,reverse=True)
            if len(tmp.shape) == 4:
                dates=util.list_days(date,month=True)
                times=util.datetimes_from_np_datetime64(dates,reverse=True)
            data['time']=np.array(times)
            attrs['time']={'desc':'Overpass date (np.datetime64)'}

        attrs['init_date']=date
        super(GC_sat,self).__init__(data,attrs,nlevs=nlevs)

        # fix TR-PAUSE_TPLEV output:
        if hasattr(self,'tplev'):
            if len(self.tplev.shape)==3:
                self.tplev=self.tplev[:,:,0]
            if len(self.tplev.shape)==4:
                self.tplev=self.tplev[:,:,:,0]

        # fix emissions shape:
        if hasattr(self, 'E_isop_bio'):
            E=self.E_isop_bio
            Eshape=np.shape(E)
            print('Eshape:',Eshape,np.nanmax(E))

            if Eshape[-1] == self.nlevs:
                if __VERBOSE__:
                    print('fixing E_isop_bio shape to [[t,]lat,lon] from ',Eshape)
                if len(Eshape)==3:
                    self.E_isop_bio = E[:,:,0]
                elif len(Eshape)==4:
                    self.E_isop_bio = E[:,:,:,0]

            # also calculate emissions in kg/s
            self._set_E_isop_bio_kgs()
        # fix dates:
        files=glob(path)
        if len(files) > 1:
            self._has_time_dim=True
            #ASSUME WE HAVE ALL DAYS IN THIS MONTH:
            self.dates=util.list_days(day0=date, month=True)


class Hemco_diag(GC_base):
    '''
        class just for Hemco_diag output and manipulation
    '''
    def __init__(self,day0, dayn=None, month=False):

        if __VERBOSE__:
            print('Reading Hemco_diag files:')

        # read data/attrs and initialise class:
        data,attrs=GC_fio.read_Hemco_diags(day0,dayn,month=month)
        attrs['init_date']=day0
        attrs['n_dims']=len(np.shape(data['ISOP_BIOG']))
        super(Hemco_diag,self).__init__(data,attrs)

        self.n_dates=len(self.dates)
        self.n_days=len(util.list_days(self.dates[0],self.dates[-1]))

        # times and local times:
        self.local_time_offset=util.local_time_offsets(self.lons,
                                                       n_lats=len(self.lats),
                                                       astimedeltas=False)

        # for conversion from kgC/m2/s to atomC/cm2/s
        # g/kg * m2/cm2 / (gram/mole * mole/molec)
        gram_per_molec = util.__grams_per_mole__['carbon'] / N_Avogadro
        self.kgC_per_m2_to_atomC_per_cm2 = 1e3 * 1e-4 / gram_per_molec

        # make E_isop_bio into atom C/cm2/s
        assert self.attrs['E_isop_bio']['units']=='kgC/m2/s', 'E_isop_bio units are %s'%self.attrs['E_isop_bio']['units']
        self.E_isop_bio = self.E_isop_bio*self.kgC_per_m2_to_atomC_per_cm2
        self.attrs['E_isop_bio']['units']='atomC/cm2/s'

        if __VERBOSE__:
            for k in data:
                print('  %10s %10s %s'%(k, data[k].shape, attrs[k]))

    def daily_LT_averaged(self,hour=13):
        '''
            Average over an hour of each day
            input hour of 1pm gives data from 1pm-2pm
        '''

        # need to subtract an hour from each of our datetimes, putting 24 hour into each 'day'
        # this makes 0500 represent 0500-0600 instead of 0400-0500 as it is in output files
        dates=np.array(self.dates) - timedelta(seconds=3600)
        # this undoes the problem of our 24th hour being stored in
        # the following day's 00th hour

        # Need to get a daily output time dimension
        days=util.list_days(dates[0],dates[-1])
        di=0
        prior_day=days[0]

        # if this has already been calculated then return that.
        if hasattr(self, 'E_isop_bio_LT'):
            if self.attrs['E_isop_bio_LT']['hour'] == hour:
                return days, self.E_isop_bio_LT

        isop=self.E_isop_bio # atomC/cm2/s
        out=np.zeros([len(days),len(self.lats),len(self.lons)])+np.NaN
        sanity=np.zeros(out.shape)
        LTO=self.local_time_offset
        for i,date in enumerate(dates):
            GMT=date.hour # current GMT
            if (date.day > prior_day.day) or (date.month > prior_day.month) or (date.year > prior_day.year):
                prior_day=date
                if (not np.all(sanity[di]==1)) or (np.any(np.isnan(out[di]))):
                    print('ERROR: should get one hour from each day!')
                    print(date, hour, GMT)
                    print(sanity[di,0,:])
                    print(dates[0::12], days)
                    assert False, ''
                di=di+1
            # local time is gmt+offset (24 hour time)
            LT=(GMT+LTO + 24) % 24 # 91,144 (lats by lons)

            #print(date.strftime('%m%dT%H'),'   lon matches:',np.sum(LT[0,:]==hour))

            out[di,LT==hour] = isop[i,LT==hour]
            sanity[di,LT==hour] = sanity[di,LT==hour]+1


        self.E_isop_bio_LT=np.squeeze(out)
        self.attrs['E_isop_bio_LT']={'desc':'map for each day of global data at specific hour local time',
                                     'hour':hour,
                                     'units':self.attrs['E_isop_bio']['units']}
        return days, np.squeeze(out)

    def plot_daily_emissions_cycle(self,lat=-31,lon=150,pname=None,color='r'):
        ''' take a month and plot the emissions over the day'''
        import matplotlib.pyplot as plt

        if pname is None:
            pname='Figs/GC/E_megan_%s.png'%self.dates[0].strftime("%Y%m")

        assert self.n_days > 1, "plot needs more than one day"

        # lat lon gives us one grid box
        lati,loni=self.lat_lon_index(lat,lon)
        offset=self.local_time_offset[0,loni] # get time offset from longitude

        data=self.E_isop_bio[:,lati,loni]

        # figure, first do whole timeline:
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 4]})
        plt.sca(a0)
        plt.plot(data, color=color)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off

        # then show daily cycle
        plt.sca(a1)
        pp.plot_daily_cycle(self.dates,data,houroffset=offset)
        #arr=np.zeros([24,self.n_days])

        plt.ylabel('E_isop_biogenic [kgC/cm2/s]')
        plt.xlabel('hour(LT)')
        plt.suptitle(self.dates[0].strftime("%b %Y"))
        plt.tight_layout()
        plt.savefig(pname)
        print("SAVED FIGURE:",pname)

class GC_biogenic:
    def __init__(self,month): #=datetime(2005,1,1)):
        '''  Read biogenic output and hemco for e_isop and o_hcho  '''
        # first get hemco output for this month
        self.hemco=Hemco_diag(month,month=True)
        # also get satellite output
        self.sat_out=GC_sat(month,run='biogenic')

    def model_slope(self, region=pp.__AUSREGION__, overpass_hour=13):
        '''
            Use RMA regression between E_isop and tropcol_HCHO to determine S:
                HCHO = S * E_isop + b
            Notes:
                Slope = Yield_isop / k_hcho
                HCHO: molec/cm2
                E_isop: kgC/cm2/s
                    Converted to atom C/cm2/s in script.

            Return {'lats','lons','r':regression, 'b':bg, 'slope':slope}

        '''
        #
        if __VERBOSE__:
            print('model_slope called for biogenic class')

        # if this calc is already done, short cut it
        if hasattr(self, 'modelled_slope'):
            # unless we're looking at a new area
            if self.modelled_slope['region'] == region:
                if __VERBOSE__:
                    print("Slope has already been modelled, re-returning")
                return self.modelled_slope

        # Read MEGAN outputs from biogenic run:
        megan=self.hemco
        # satellite output for hcho concentrations:
        sat_out=self.sat_out

        # grab satellite overpass E_isop and trop column hcho
        days,isop = megan.daily_LT_averaged(hour=overpass_hour) # lat/lon kgC/cm2/s [days,lat,lon]?
        # isop should be atom C/cm2/s
        assert megan.attrs['E_isop_bio']['units'] == 'atomC/cm2/s', 'units are bad in E_isop_bio %s'%megan.attrs['E_isop_bio']['units']

        hcho = sat_out.get_trop_columns(keys=['hcho'])['hcho'] # ppbv -> molec/cm2

        # what lats and lons do we want?
        lats,lons = sat_out.lats, sat_out.lons
        lati,loni = util.lat_lon_range(lats,lons,region=region)
        assert all(sat_out.lats == megan.lats), "output lats don't match"
        # subset the data
        isop = isop[:, lati, :]
        isop = isop[:, :, loni]
        hcho = hcho[:, lati, :]
        hcho = hcho[:, :, loni]
        sublats, sublons = lats[lati], lons[loni]

        # Convert to atomC/cm2/s
        #isop=isop * self.hemco.kgC_per_m2_to_atomC_per_cm2
        #assert self.hemco.attrs['E_isop_bio']['units'] == 'kgC/cm2/s', 'units are bad for E_isop_bio (%s)'%self.hemco.attrs['E_isop_bio']

        print("in slope function: ")
        print("  nanmean E_isop = %.2e %s"%(np.nanmean(isop),'atom C/cm2/s'))
        print("  nanmean trop_hcho = %.2e %s"%(np.nanmean(hcho),'molec/cm2'))

        # arrays to hold the month's slope, background, and regression coeff
        n_x = len(loni)
        n_y = len(lati)
        slope  = np.zeros([n_y,n_x]) + np.NaN
        bg     = np.zeros([n_y,n_x]) + np.NaN
        reg    = np.zeros([n_y,n_x]) + np.NaN

        # regression for each lat/lon gives us slope
        for xi in range(n_x):
            for yi in range(n_y):
                # Y = m X + B
                X=isop[:, yi, xi]
                Y=hcho[:, yi, xi]

                # Skip ocean or no emissions squares:
                # When using kgC/cm2/s, we are always close to zero (1e-11 order)
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

def check_units(d=datetime(2005,1,1)):
    '''
        N_air (molecs/m3)
        boxH (m)
        trop_cols (molecs/cm2)
        E_isop (TODO)

    '''
    N_ave=6.02214086*1e23 # molecs/mol
    airkg= 28.97*1e-3 # ~ kg/mol of dry air
    for gc in [GC_tavg(d), GC_sat(d)]:
        print('---')
        print('')

        #data in form [time,lat,lon,lev]
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
        print("Mean surface boxH=%.2fm"%np.mean(gc.boxH[:,:,:,0]))
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


def check_diag(d=datetime(2005,1,1)):
    '''
    '''
    gc=GC_tavg(d)
    E_isop_hourly=gc.E_isop_bio
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


if __name__=='__main__':
    #check_diag()
    check_units(datetime(2005,2,1))



