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

#from glob import glob

# 'local' modules
from utilities import GC_fio, masks, fio
import utilities.utilities as util
from utilities import plotting as pp
from utilities.plotting import __AUSREGION__
from utilities.JesseRegression import RMA
import utilities.GMAO as GMAO


##################
#####GLOBALS######
##################

__VERBOSE__=False # For file-wide print statements

# filepaths:
rdir='Data/GC_Output/'
sat_path  = {'tropchem':rdir+'geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s.bpch',
             'halfisop':rdir+'geos5_2x25_tropchem_halfisoprene/satellite_output/ts_satellite.%s.bpch',
             'noisop':rdir+'geos5_2x25_tropchem_noisoprene/satellite_output/ts_satellite.%s.bpch',
             'UCX':rdir+'UCX_geos5_2x25/satellite_output/ts_satellite.%s',
             'biogenic':rdir+'geos5_2x25_tropchem_biogenic/satellite_output/sat_biogenic.%s.bpch',
             'new_emissions':rdir+'new_emissions/satellite_output/ts_satellite_altered.%s.bpch',
             }
#             'new_emiss':rdir+'new_emissions/satellite_output/ts_satellite_altered.%s.bpch',}
tavg_path = {'tropchem':rdir+'geos5_2x25_tropchem/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'halfisop':rdir+'geos5_2x25_tropchem_halfisoprene/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'noisop':rdir+'geos5_2x25_tropchem_noisoprene/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'UCX':rdir+'UCX_geos5_2x25/trac_avg/trac_avg_geos5_2x25_UCX_updated.%s',
             'nochem':rdir+'nochem/trac_avg/trac_avg.geos5_2x25_tropchem.%s',
             'new_emissions':rdir+'new_emissions/trac_avg/trac_avg.geos5_2x25_tropchem_altered.%s',
              }
             #'new_emiss':rdir+'new_emissions/trac_avg/trac_avg.geos5_2x25_tropchem_altered.%s',}

# GC OUTPUT NAMES:
__coords__ = ['lev','lon','lat','time']
__ijavg__  = ['IJ-AVG-$_ISOP',
              'IJ-AVG-$_CH2O',
              'IJ-AVG-$_NO2',       # NO2 in ppbv
              'IJ-AVG-$_O3',        # O3 in ppbv
              'TIME-SER_AIRDEN',    # Named differently in satellite output
              'CHEM-L=$_OH',]       # OH concentrations?
__emiss__  = [#'BIOGSRCE_ISOP',      # biogenic source of isoprene () []
              'BIOBSRCE_CH20',      # biomass burning hcho source () []
              'ANTHSRCE_NO',        # anthro source of NO (molec/cm2/s) [t,lat,lon,1]
              'NO-SOIL_NO',         # soil emissions of NO (molec/cm2/s) [t,lat,lon,1]
              'BIOBSRCE_NO',        # fire no (molec/cm2/s) [t,lat,lon,1]
             ]
__other__  = ['PEDGE-$_PSURF',      # pressure at surface of each gridbox (hPa)
              'BXHGHT-$_BXHEIGHT',  # box height (?)
              'BXHGHT-$_AD',        # Air density (kg)
              'BXHGHT-$_AVGW',      # water ??
              'BXHGHT-$_N(AIR)',    # number density of air (?)
              'DXYP_DXYP',          # gridbox horizontal area (m?)
              'PORL-L=$_LHCHO',     # HCHO loss in mol/cm3/s
              'PORL-L=$_LISO',      # isoprene loss in mol/cm3/s
              'TR-PAUSE_TP-LEVEL',  #
              'TR-PAUSE_TPLEV',    # Added satellite output for ppamf
              'DAO-3D-$_TMPU',      # Temperature field (Kelvin)
              'DAO-FLDS_TS', ]      # Surf Temp (Kelvin)


__gc_allkeys__ = __ijavg__ + __emiss__ + __other__

__gc_tropcolumn_keys__ = ['TIME-SER_AIRDEN',    # air density for tracavg output
                          'PEDGE-$_PSURF',      # pressure at surface of each gridbox (hPa)
                          'BXHGHT-$_BXHEIGHT',  # box height (?)
                          'BXHGHT-$_AD',        # Air density (kg)
                          'BXHGHT-$_AVGW',      # water ??
                          'BXHGHT-$_N(AIR)',    # number density of air (?)
                          'DXYP_DXYP',          # gridbox horizontal area (m?)
                          'TR-PAUSE_TP-LEVEL',  #
                          'TR-PAUSE_TPLEV',]    # Added satellite output for ppamf

# MAP GC output to nicer names:
_ija='IJ-AVG-$_'
_bxh='BXHGHT-$_'
_GC_names_to_nice = { 'time':'time','lev':'press','lat':'lats','lon':'lons',
    # IJ_AVGs: in ppbv, except isop (ppbC)
    _ija+'NO':'NO', _ija+'O3':'O3', _ija+'MVK':'MVK', _ija+'MACR':'MACR',
    _ija+'ISOPN':'isopn', _ija+'IEPOX':'iepox', _ija+'NO3':'NO3',
    _ija+'NO2':'NO2', _ija+'ISOP':'isop', _ija+'CH2O':'hcho',
    # Biogenic sources:
    'BIOGSRCE_ISOP':'E_isop_bio', # atoms C/cm2/s
    'ISOP_BIOG':'E_isop_bio', # kgC/cm2/s (from Hemco_diagnostic output)
    'NO-SOIL_NO':'NO_soil', # molec/cm2/s (soil nox)
    # burning sources: atoms C/cm2/s
    'BIOBSRCE_CH2O':'E_hcho_burn',
    # Other diagnostics:
    'PEDGE-$_PSURF':'psurf',
    _bxh+'BXHEIGHT':'boxH', # metres
    _bxh+'AD':'AD', # air mass in grid box, kg
    _bxh+'AVGW':'avgW', # Mixing ratio of H2O vapor, v/v
    _bxh+'N(AIR)':'N_air', # Air density: molec/cm3 from tavg output
    'TIME-SER_AIRDEN':'N_air', # Air density: molec/cm3 from satellite output
    'DXYP_DXYP':'area', # gridbox surface area: m2
    'TR-PAUSE_TP-LEVEL':'tplev',
    'TR-PAUSE_TPLEV':'tplev', # this one is added to satellite output manually
    'TR-PAUSE_TP-HGHT':'tpH', # trop height: km
    'TR-PAUSE_TP-PRESS':'tpP', # trop Pressure: mb
    'PORL-L=$_LHCHO':'Lhcho', # loss of hcho mol/cm3/s
    'PORL-L=$_LISO':'Lisop', # loss of isop mol/cm3/s
    # Many more in trac_avg_yyyymm.nc, not read here yet...
    'CHEM-L=$_OH':'OH', # OH molec/cm3: (time, alt059, lat, lon) : 'chemically produced OH'
                        # alt059 is 38 levels for no reason
    'DAO-3D-$_TMPU':'temp', # temperature
    'DAO-FLDS_TS':'surftemp', # surface temperature, Kelvin
    }


################
#####CLASS######
################

class GC_base:
    '''
        Class for GEOS-Chem output, inherited by tavg and sat
    '''
    def __init__(self, data,attrs, nlevs=47, nlats=91, nlons=144):
        '''

        '''
        self.ntimes=1
        if 'time' in data:
            if isinstance(data['time'], list):
                self.ntimes=len(data['time'])
            elif isinstance(data['time'], np.ndarray):
                self.ntimes = data['time'].size
            if __VERBOSE__:
                print( "CHECK data['time']:")
                print(type(data['time']))
                print(np.shape(data['time']))
                print(data['time'])
            
        self.nlats=nlats
        self.nlons=nlons
        self.nlevs=nlevs

        # Initialise to zeros:
        #self.hcho  = 0      #PPBV
        #self.isop  = 0      #PPBC (=PPBV*5)
        #self.O_hcho= 0      # column hcho molec/cm2
        #self.boxH  = 0      #box heights (m)
        #self.psurf = 0      #pressure surfaces (hPa)
        #self.area  = 0      #XY grid area (m2)
        #self.N_air = 0      #air dens (molec_air/cm3)
        #self.E_isop_bio = 0 #"atoms C/cm2/s" # ONLY tropchem
        self.attrs={}       # Attributes from bpch file

        if hasattr(self,'area'):
            if len(self.area.shape) > 2:
                self.area = self.area[0] # remove time dimension

        # Data could have any shape, we fix to time,lat,lon,lev
        # for each key in thefile
        for key in data.keys():

            # Make sure array has dims: [[time,]lats,lons[,levs]]
            arr=data[key]
            if len(arr.shape) > 1:
                if key in ['CHEM-L=$_OH', 'PORL-L=$_LHCHO','PORL-L=$_LISO']:
                    levdim=len(arr.shape)-1
                    keylevels=arr.shape[levdim] # sometimes 38 levels instead of 47...
                else:
                    keylevels=self.nlevs
                if __VERBOSE__:
                    print (np.shape(arr),self.ntimes,self.nlats,self.nlons,keylevels)
                    print (key)
                arr = util.reshape_time_lat_lon_lev(arr,self.ntimes,self.nlats,self.nlons,keylevels)

            # Fix air density units to molec/cm3, in case they are in molec/m3
            if (key == 'TIME-SER_AIRDEN') or (key == 'BXHGHT-$_N(AIR)'):
                # grab surface air density to check units
                surf_air=np.nanmean(arr[:,:,0])
                if len(arr.shape)==4:
                    surf_air=np.nanmean(arr[:,:,:,0]) # maybe has time dimension

                if __VERBOSE__:
                    print(key,np.shape(arr),'surface N_air:',surf_air)
                if key in attrs:
                    if __VERBOSE__:
                        print("attrs          :     values")
                        _blah = [ print('%15s:%15s'%(k, v)) for k,v in attrs[key].items() ]
                    if 'unit' in attrs[key]:
                        # we want molec/cm3
                        attrs[key]['orig_unit']=attrs[key]['unit']
                else:
                    if __VERBOSE__:
                        print(key,' has no attributes!!, assuming molec/cm3 or molec/m3')
                    attrs[key]={'units':'molec/cm3'}

                if surf_air > 1e23: # probably molec/m3
                    arr=arr*1e-6
                    if __VERBOSE__:
                        print(key,' being changed from molec/m3 to molec/cm3')
                        print(key,'shape:', np.shape(arr))#, 'surface N_air:', arr)

                attrs[key]['units']='molec/cm3'

            nkey=key
            if key in _GC_names_to_nice.keys():
                nkey=_GC_names_to_nice[key]

            setattr(self, nkey, arr)
            # just use 'units' not 'unit'
            if 'unit' in attrs[key]:
                if not ('units' in attrs[key].keys()):
                    attrs[key]['units'] = attrs[key]['unit'] # save unit to units
                attrs[key].pop('unit',None) # delete 'unit' attribute

            self.attrs[nkey] = attrs[key]

            if __VERBOSE__:
                print("READING %s(now %s) %s(now %s)"%(key,nkey,data[key].shape,np.shape(getattr(self,nkey))))
                if self.attrs[nkey] is not None:
                    for k,v in self.attrs[nkey].items():
                        if k in ['full_name','original_shape','units','orig_unit','axis','standard_name']:
                            print('    %15s:%15s'%(k, v))

        # Grab area if file doesn't have it
        if not hasattr(self,'area'):
            self.area=GMAO.area_m2

        # Calculate total columns hcho
        # molec/cm2 = ppbv * 1e-9 * molec_A / cm3 * H(cm)
        if all([hasattr(self,attr) for attr in ['hcho','N_air','boxH']]):
            n_dims=len(np.shape(self.hcho))
            if __VERBOSE__:
                print("CHECK:hcho %s, mean = %.2e %s"%(str(np.shape(self.hcho)),np.mean(self.hcho),self.attrs['hcho']['units']))
            self.O_hcho = np.sum(self.units_to_molec_cm2(keys=['hcho',])['hcho'],axis=n_dims-1)
            self.attrs['O_hcho']={'units':'molec/cm2','desc':'Total column HCHO'}
            if __VERBOSE__:
                print("CHECK:O_hcho %s, mean=%.2e %s"%(str(self.O_hcho.shape),np.mean(self.O_hcho),self.attrs['O_hcho']['units']))


        # Convert from numpy.datetime64 to datetime
        if not hasattr(self,'time'):
            self.time=[attrs['init_date'].strftime("%Y-%m-%dT%H:%M:%S.000000000")]
            self.dates=[datetime.strptime(self.time[0],"%Y-%m-%dT%H:%M:%S.000000000"),]
        else:
            self.dates=list(util.datetimes_from_np_datetime64(self.time))
        
        # debug
        #print('self.time',self.time)
        #print('self.dates',self.dates)
        self.dstr=self.dates[0].strftime("%Y%m")

        # flag to see if class has time dimension
        self._has_time_dim = len(self.dates) > 1
        
        # Check that all the lats match the GMAO resolution we use
        assert (set(self.lats) & set(GMAO.lats_m)) == set(self.lats), "LATS DON'T MATCH GMAO 2x25 MIDS"
        self.lats_e=GMAO.lats_e
        if self.nlats < 91:
            self.lats_e=util.edges_from_mids(self.lats)
        # same for lons, but I don't ever trim the lons I think
        assert all(self.lons == GMAO.lons_m), "LONS DON'T MATCH GMAO 2x25 MIDS"
        self.lons_e=GMAO.lons_e

    # Common Functions:
    def date_index(self, date):
        ''' Return index of date '''
        return util.date_index(date,self.dates)

    def lat_lon_index(self,lat,lon):
        '''  return lati, loni '''
        return util.lat_lon_index(lat,lon,self.lats,self.lons)

    def units_to_molec_cm2(self, keys=[]):
        ''' convert units to molec/cm2    '''
        out={}
        for key in keys:
            data=getattr(self,key)
            data_levels=data.shape[-1] # some arrays have 38 levels..?
            boxh=self.boxH # maybe in metres
            if str.strip(self.attrs['boxH']['units']) == 'm':
                boxh=boxh*100 # change to centimetres

            # subset of levels
            if len(data.shape) == 3:
                boxh=boxh[:,:,0:data_levels]
            elif len(data.shape) == 4:
                boxh=boxh[:,:,:,0:data_levels]

            units=str.strip(self.attrs[key]['units'])
            # MOLEC/CM3 TO MOLEC/CM2
            if  units == 'molec/cm3':
                out[key] = data*boxh
            # PPBV TO MOLEC/CM2
            elif units == 'ppbv' or units == 'ppbC':
                N_air=self.N_air # molec/cm3
                assert self.attrs['N_air']['units']=='molec/cm3', 'N_air units are NOT molec/cm3, they are %s'%self.attrs['N_air']['units']
                if units=='ppbC':
                    if key=='isop':
                        data=data/5.0 # ppb carbon to ppb isoprene
                    else:
                        assert False, '%s(%s) unsure how to convert to molec/cm3'%(key,units)

                # ppb * 1e-9 * molec_air/cm3 * boxH(cm)
                # scale=0.0
                out[key] = data * 1e-9 * N_air * boxh # molec/cm2
            else:
                assert False, "Haven't made conversion for %s(%s) to molec/cm2"%(key,units)

        return out

    def ppbv_to_molec_cm2(self, keys=['hcho']):
        ''' return dictionary with data in format molecules/cm2'''
        return self.units_to_molec_cm2(keys)

    def get_total_columns(self, keys=['hcho']):
        return self.get_trop_columns(keys, total_column=True)

    def get_trop_columns(self, keys=['hcho'], total_column=False):
        '''
            Return tropospheric column amounts in molec/cm2
        '''
        if __VERBOSE__:
            print('retrieving %s column for '%['trop','total'][total_column],keys)
            for key in keys:
                print(key, self.attrs[key]['units'])
        data={}

        # where is tropopause and how much of next box we want
        if total_column:
            trop=np.zeros([self.ntimes,self.nlats,self.nlons],dtype=np.int)
            trop[:,:,:]=self.nlevs - 1
            trop=np.squeeze(trop)
            extra = np.copy(trop)
            extra[:]=1
            # all but top level + 1* top level
        else:
            trop=np.floor(self.tplev).astype(int)
            extra=self.tplev - trop

        Xdata=self.units_to_molec_cm2(keys=keys)
        # for each key, work out trop columns
        for key in keys:
            X=Xdata[key]
            dims=np.array(np.shape(X))
            if __VERBOSE__:
                print("%s has shape %s"%(key,str(dims)))
            # Which index is time,lat,lon,lev?
            timei=0;lati=1;loni=2
            out=np.zeros(dims[[timei,lati,loni]])

            hastime= len(dims) > 3 # [[time],lat,lon,lev]

            if not hastime: # no time dim
                lati=0; loni=1
                out=np.zeros(dims[[lati, loni]])

            for lat in range(dims[lati]): # loop over lats
                for lon in range(dims[loni]): # loop over lons
                    try:
                        if not hastime:
                            tropi=trop[lat,lon]
                            if tropi > dims[2]:
                                tropi = dims[2]-1
                            out[lat,lon]=np.sum(X[lat,lon,0:tropi])+extra[lat,lon] * X[lat,lon,tropi]
                        else:
                            for t in range(dims[timei]):
                                tropi=trop[t,lat,lon]
                                if tropi > dims[3]:
                                    tropi = dims[3]-1
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


    def plot_hcho_columns(self, region=pp.__AUSREGION__):
        ''' plot map of hcho columns over a region '''
        data=np.nanmean(self.O_hcho, axis=0) # average over time dimension
        pp.createmap(data,self.lats,self.lons)

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

        #print(E.shape,conversion.shape, self.area.shape)
        #if len(conversion.shape) > 2:
        #    conversion=conversion[0]
        #print(E.shape,conversion.shape)

        if self._has_time_dim:
            self.E_isop_bio_kgs = np.zeros(np.shape(E))
            conversion=np.repeat(conversion[np.newaxis,:,:], len(self.dates), axis=0)

        self.E_isop_bio_kgs=E * conversion
        self.attrs['E_isop_bio_kgs']={'units':'kg/s',}

class GC_tavg(GC_base):
    '''
        Class holding and manipulating tropchem GC output
        # tropchem dims [time,lat,lon,lev]
        # UCX dims: lev = 72; alt059 = 59; lat = 91; lon = 144;
        # LATS AND LONS ARE BOX MIDPOINTS
        self.hcho  = PPBV (molec/molec_air)
        self.isop  = PPBC (=PPBV*5)
        self.boxH  = box heights (m)
        self.psurf = pressure surfaces (hPa)
        self.area  = XY grid area (m2)
        self.N_air = air dens (molec_air/cm3)
        self.E_isop_bio= "atoms C/cm2/s" # ONLY tropchem

    '''
    def __init__(self,day0,dayN=None,keys=__gc_allkeys__,run='tropchem',nlevs=None):
        # Call GC_base initialiser with tavg_mainkeys and tropchem by default

        # Determine path of files:
        dates=util.list_months(day0,dayN)
        dstrs=[ m.strftime("%Y%m%d0000") for m in dates]
        paths= [ tavg_path[run]%dstr for dstr in dstrs ]

        # read data/attrs and initialise class:
        data,attrs = GC_fio.read_bpch(paths,keys=keys)

        if nlevs is None:
            nlevs =47 # assume 47

        if run == 'UCX':
            nlevs = 72
            # reading monthly averages does not create a datetime array
            if not hasattr(data,'time'):
                data['dates']=np.array(dates)
                attrs['dates']={'units':'datetimes'}
                data['time']= np.array(util.datetimes_from_np_datetime64(dates,reverse=True))
                attrs['time']={'units':'datetime64','desc':'month starts for monthly averaged data'}



        attrs['init_date']=day0

        super(GC_tavg,self).__init__(data,attrs,nlevs=nlevs)

        # add E_isop_bio_kgs:
        if hasattr(self,'E_isop_bio'):
            self._set_E_isop_bio_kgs()

    def hcho_lifetime(self, month, region=pp.__AUSREGION__):
        '''
        Use tau = HCHO / Loss    to look at hcho lifetime over a month
        return lifetimes[day,lat,lon],
        '''
        print("CHECK: TRYING HCHO_LIFETIME:",month, region)

        d0=util.first_day(month)
        dN=util.last_day(month)

        # read hcho and losses from trac_avg
        # ch20 in ppbv, air density in molec/cm3,  Loss HCHO mol/cm3/s
        keys=['IJ-AVG-$_CH2O','BXHGHT-$_N(AIR)', 'PORL-L=$_LHCHO']
        #run = GC_class.GC_tavg(d0,dN, keys=keys) # [time, lat, lon, lev]
        print('     :READ GC_tavg')
        # TODO: Instead of surface use tropospheric average??
        hcho = self.hcho[:,:,:,0] # [time, lat, lon, lev=47] @ ppbv
        N_air = self.N_air[:,:,:,0] # [time, lat, lon, lev=47] @ molec/cm3
        Lhcho = self.Lhcho[:,:,:,0] # [time, lat, lon, lev=38] @ mol/cm3/s  == molec/cm3/s !!!!
        lats=self.lats
        lons=self.lons

        # [ppbv * 1e-9 * (molec air / cm3) / (molec/cm3/s)] = s
        tau = hcho * 1e-9 * N_air  /  Lhcho
        print('     :CALCULATED tau')
        # change to hours
        tau=tau/3600.
        #
        self.tau=tau

        if region is not None:
            subset=util.lat_lon_subset(lats,lons,region,[tau],has_time_dim=True)
            tau=subset['data'][0]
            lats=subset['lats']
            lons=subset['lons']
        print("CHECK: MANAGED HCHO_LIFETIME:",month)

        return tau, self.dates, lats, lons

class GC_sat(GC_base):
    '''
        Class for reading and manipulating satellite overpass output!
    '''
    def __init__(self,day0, dayN=None, keys=__gc_allkeys__, run='tropchem',nlevs=None):

        if nlevs is None:
            if run=='UCX':
                nlevs=72
            else:
                nlevs=47
                
        # Remove duplicate keys
        keys = list(set(keys))
        
        # Determine path of files:
        dates=util.list_days(day0,dayN)
        dstrs=util.list_days_strings(day0,dayN)
        paths= [ sat_path[run]%dstr for dstr in dstrs ]
        
        # need to handle reading of multiple years specially
        years=util.list_years(day0,dayN)
        nlats=91
        nlons=144
        if len(years) >1:
            # read data and attrs from bigger file created specially for this:
            data, attrs = GC_fio.read_overpass_file(day0,dayN, run=run, keys=keys)
            nlats=28 # multiyear file is trimmed
        else:
            # read data/attrs and initialise class:
            print("GC_sat reading keys:",keys)
            data,attrs = GC_fio.read_bpch(paths,keys=keys)
            
        times=util.datetimes_from_np_datetime64(dates,reverse=True)
        data['time']=np.array(times)
        attrs['time']={'desc':'Overpass date (np.datetime64)'}

        attrs['init_date']=day0
        
        # may need to handle missing time dim...
        super(GC_sat,self).__init__(data,attrs,nlats=nlats,nlevs=nlevs)

        # fix dates:
        self.has_time_dim= len(dates) > 1
        #self.dates=dates
        
        if hasattr(self,'boxH'):
            # box heights (m)
            h=self.boxH

            # Altitudes (m)
            level_axis=2
            if self.has_time_dim:
                level_axis=3
            self.zmids=np.cumsum(h,axis=level_axis)-h/2.0
            self.attrs['zmids']={'units':'m','desc':'altitude at middle of each level'}
            
        
        # fix TR-PAUSE_TPLEV output:
        if hasattr(self,'tplev'):
            if self._has_time_dim:
                self.tplev=self.tplev[:,:,:,0]
            else:
                self.tplev=self.tplev[:,:,0]

        # fix emissions shape:
        if hasattr(self, 'E_isop_bio'):
            E=self.E_isop_bio
            Eshape=np.shape(E)
            #print('Eshape:',Eshape,np.nanmax(E))
            if np.nanmax(E) == 0.0:
                self.E_isop_bio=None
            else:
                if Eshape[-1] == self.nlevs:
                    if __VERBOSE__:
                        print('fixing E_isop_bio shape to [[t,]lat,lon] from ',Eshape)
                    if len(Eshape)==3:
                        self.E_isop_bio = E[:,:,0]
                    elif len(Eshape)==4:
                        self.E_isop_bio = E[:,:,:,0]

                # also calculate emissions in kg/s
                self._set_E_isop_bio_kgs()

        # Calculate shape factors for faster AMF calculation later
        # Only if we have all the stuff and no time dimension:
        if all([hasattr(self, astr) for astr in ['hcho','N_air','psurf','boxH','O_hcho']]) and not self._has_time_dim:
            # Nhcho (molec/cm3) = vmr_hcho (ppb*1e-9) * Nair (molec/cm3)
            assert self.attrs['N_air']['units'] == 'molec/cm3', 'N_air is NOT molec/cm3'
            self.N_hcho = self.hcho * self.N_air * 1e-9
            self.attrs['N_hcho']={'units':'molec/cm3','desc':'HCHO number density'}

            # levels are final dimension
            arrshape=self.hcho.shape
            n_lats,n_lons,n_levs = arrshape

            # Column air (molec/cm2) = (molec/cm3 * m * 100 cm/m)
            self.O_air = np.sum(self.N_air * self.boxH * 100, axis=2)

            # Add attributes
            self.attrs['O_air']={'units':'molec/cm2','desc':'Total column Air'}

            self.add_pedges()
            pmids=self.pmids
            pedges=self.pedges
            TOA=GMAO.__TOA__

            # Top pmid SHOULD be a little higher than 0.01
            assert np.all(pmids[:,:,-1]> TOA), "Problem with pmid calculation"
            #if not np.all(pmids[:,:,-1]>0.1):
            #    print("WARNING: What's going on with TOA pressure? (should be 0.1)")
            #    print("       : average TOA pedge = %.2e hPa"%np.mean(pedges[:,:,-1]))
                #print(self.psurf[5,5,:])
                #print(pedges[5,5,:])
                #print(pmids[5,5,:])
                #print(pmids[:,:,-1])
                #print(pedges[:,:,-1])
                #print(pedges[:,:,-2])

            # surface pressure matching pmids array
            pbots=self.psurf
            psurf=pbots[:,:,0]
            psurfs = np.repeat(psurf[:,:,np.newaxis],n_levs,axis=2)

            # box heights (m)
            h=self.boxH

            # Altitudes in sigmas coordinates
            #self.smids = (pmids - 0.1) / (psurfs - 0.1)
            self.smids = (pmids - TOA) / (psurfs - TOA)
            self.attrs['smids']={'units':'unitless','desc':'sigma at middle of each level'}

            ## The shape factors!
            # First repeat total column amounts (molec/cm2) along level dimension
            # to avoid looping
            O_hcho_rep = np.copy(self.O_hcho)
            O_hcho_rep = np.repeat(O_hcho_rep[:,:,np.newaxis],n_levs, axis=2)
            O_air_rep = np.copy(self.O_air)
            O_air_rep = np.repeat(O_air_rep[:,:,np.newaxis],n_levs, axis=2)

            # S_z (1/m) = N_HCHO (molec/cm3) / O_HCHO (molec/cm2)  * 100(cm/m)
            self.Shape_z = self.N_hcho / O_hcho_rep * 100
            self.attrs['Shape_z']={'units':'1/m','desc':'Shape_z at each level'}

            # S_s (unitless) = vmr (molec hcho/air) Column_air (molec/cm2) / Column_H (molec/cm2)
            self.Shape_s = self.hcho * 1e-9 * (O_air_rep / O_hcho_rep)
            self.attrs['Shape_s']={'units':'unitless','desc':'Shape_sigma at each level'}

    def add_pedges(self):
        '''
            Add pedges to class
        '''

        # Pressure at bottom edge of each level
        pbots=self.psurf
        TOA=GMAO.__TOA__

        if self._has_time_dim:
            n_times, n_lats,n_lons,n_levs = self.psurf.shape
            pedges=np.ndarray([n_times,n_lats,n_lons,n_levs+1])

            pedges[:,:,:,:-1]=pbots
            pedges[:,:,:,-1] = TOA
            pmids=np.sqrt(pedges[:,:,:,:-1]*pedges[:,:,:,1:])

        else:
            n_lats,n_lons,n_levs = self.psurf.shape

            # pressure edges
            pedges=np.ndarray([n_lats,n_lons,n_levs+1])

            TOA=GMAO.__TOA__
            pedges[:,:,:-1]=pbots
            pedges[:,:,-1] = TOA
            pmids=np.sqrt(pedges[:,:,:-1]*pedges[:,:,1:])

        self.pedges=pedges
        self.attrs['pedges']={'units':'hPa','desc':'pressure edges'}
        self.pmids=pmids
        self.attrs['pmids']={'units':'hPa','desc':'pressure midpoints'}


    def calculate_AMF(self, w, w_pmids, AMF_G, lat, lon, plotname=None, debug_levels=False):
        '''
        Return AMF calculated using normalized shape factors
        uses both S_z and S_sigma integrations

        Determines
            AMF_z = \int_0^{\infty} { w(z) S_z(z) dz }
            AMF_s = \int_0^1 { w(s) S_s(s) ds } # this one uses sigma dimension
            AMF_Chris = \Sigma_i (Shape(P_i) * \omega(P_i) * \Delta P_i) /  \Sigma_i (Shape(P_i) * \omega(P_i) )

        update:20180516
            Moved from gchcho class to GC_sat class - which reads directly from bpch files
            This should make the gchcho creation unnecessary
        update:20161109
            Chris AMF calculated, along with normal calculation without AMF_G
            lowest level fix now moved to a function to make this method more readable
        update:20160905
            Fix lowest level of both GC box and OMI pixel to match the pressure
            of the least low of the two levels, and remove any level which is
            entirely below the other column's lowest level.
        update:20160829
            LEFT AND RIGHT NOW SET TO 0 ()
        OLD:
            Determine AMF_z using AMF_G * \int_0^{\infty} { w(z) S_z(z) dz }
            Determine AMF_sigma using AMF_G * \int_0^1 { w(s) S_s(s) ds }

        '''
        # column index, pressures, altitudes, sigmas:
        #
        lati, loni=self.lat_lon_index(lat,lon)

        # pressure edges and mids (hPa)
        pedges=self.pedges[lati,loni,:]
        pmids=self.pmids[lati,loni]

        # box heights (m)
        h=self.boxH[lati,loni]

        # Altitudes (m)
        zmids=self.zmids[lati,loni]

        # Altitudes in sigmas coordinates
        smids = self.smids[lati,loni]

        ## The shape factors!

        # S_z (1/m) = N_HCHO (molec/cm3) / O_HCHO (molec/cm2)  * 100(cm/m)
        S_z = self.Shape_z[lati,loni]

        # S_s (unitless) = vmr (molec hcho/air) Column_air (molec/cm2) / Column_H (molec/cm2)
        S_s = self.Shape_s[lati,loni]

        # Interpolate the shape factors to these new pressure levels:
        # S_s=np.interp(S_pmids, S_pmids_init[::-1], S_s[::-1])
        # also do S_z? currently I'm leaving this for comparison. AMF_z will be sanity check

        # calculate sigma edges
        sedges = (pedges - pedges[-1]) / (pedges[0]-pedges[-1])
        dsigma = sedges[0:-1]-sedges[1:]  # change in sigma at each level

        # Default left,right values (now zero)
        lv,rv=0.,0.

        # sigma midpoints for interpolation
        w_smids = (w_pmids - pedges[-1])/ (pedges[0]-pedges[-1])

        # convert w(press) to w(z) and w(s), on GEOS-Chem's grid
        #
        w_zmids = np.interp(w_pmids, pmids[::-1], zmids[::-1])
        w_z     = np.interp(zmids, w_zmids, w,left=lv,right=rv) # w_z does not account for differences between bottom levels of GC vs OMI pixel
        w_s     = np.interp(smids, w_smids[::-1], w[::-1],left=lv,right=rv)
        w_s_2   = np.interp(smids, w_smids[::-1], w[::-1]) # compare without fixed edges!

        # Integrate w(z) * S_z(z) dz using sum(w(z) * S_z(z) * height(z))
        AMF_z = np.sum(w_z * S_z * h)
        AMF_s= np.sum(w_s * S_s * dsigma)

        # Calculations with bottom relevelled
        # match he bottom levels in the pressure midpoints dimension
        w_pmids_new,S_pmids_new,w_new,S_s_new = util.match_bottom_levels(w_pmids,pmids,w,S_s)
        S_pedges_new = pedges.copy()
        for i in range(1,len(S_pedges_new)-1):
            S_pedges_new[i]=(S_pmids_new[i-1]*S_pmids_new[i]) ** 0.5
        S_sedges_new = (S_pedges_new - S_pedges_new[-1]) / (S_pedges_new[0]-S_pedges_new[-1])
        dsigma_new = S_sedges_new[0:-1]-S_sedges_new[1:]
        w_smids_new = (w_pmids_new - S_pedges_new[-1])/ (S_pedges_new[0]-S_pedges_new[-1])
        S_smids_new=(S_pmids_new - S_pedges_new[-1]) / (S_pedges_new[0]-S_pedges_new[-1])
        w_s_new     = np.interp(S_smids_new, w_smids_new[::-1], w_new[::-1], left=lv, right=rv)
        AMF_s_new = np.sum(w_s_new * S_s_new * dsigma_new)

        if plotname is not None:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FormatStrFormatter as fsf # formatter for labelling

            integral_s_old=np.sum(w_s_2 * S_s * dsigma)
            AMF_s_old= AMF_G * integral_s_old

            f,axes=plt.subplots(2,2,figsize=(14,12))
            # omega vs pressure, interpolated and not interpolated
            plt.sca(axes[0,0])
            plt.plot(w, w_pmids, linestyle='-',marker='o', linewidth=2, label='original',color='k')
            plt.title('$\omega$')
            plt.ylabel('p(hPa)'); plt.ylim([1015, 0.01]); plt.yscale('log')
            ax=plt.twinx(); ax.set_ylabel('z(m)'); plt.sca(ax)
            plt.plot(w_z, zmids, linestyle='-',marker='x', label='$\omega$(z)',color='fuchsia')
            # add legend from both axes
            h1,l1 = axes[0,0].get_legend_handles_labels()
            h2,l2 = ax.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2,loc=0)

            # omega vs sigma levels
            plt.sca(axes[0,1])
            plt.plot(w_s, smids, '.')
            plt.title('$\omega(\sigma)$')
            plt.ylabel('$\sigma$')
            plt.yscale('log')
            plt.ylim(1.01, 0.001)
            axes[0,1].yaxis.tick_right()
            axes[0,1].yaxis.set_label_position("right")
            # add AMF value to plot
            for yy,lbl in zip([0.6, 0.7, 0.8, 0.9], ['AMF$_z$=%5.2f'%AMF_z, 'AMF$_{\sigma}$=%5.2f'%AMF_s, 'AMF$_{\sigma}$(pre-fix)=%5.2f'%AMF_s_old, 'AMF$_{\sigma relevelled}$=%5.2f'%AMF_s_new]):
                plt.text(.1,yy,lbl,transform=axes[0,1].transAxes,fontsize=16)

            # shape factor z plots
            plt.sca(axes[1,0])
            plt.plot(S_z, pmids, '.',label='S$_z$(p)', color='k')
            plt.ylim([1015,0.01])
            plt.title('Shape')
            plt.ylabel('p(hPa)')
            plt.yscale('log')
            plt.xlabel('m$^{-1}$')
            # overplot omega*shape factor on second y axis
            ax=plt.twinx(); ax.set_ylabel('z(m)'); plt.sca(ax)
            plt.plot(S_z*w_z, zmids,label='$S_z(z) * \omega(z)$',color='fuchsia')
            # legend
            h1,l1 = axes[1,0].get_legend_handles_labels()
            h2,l2 = ax.get_legend_handles_labels()
            ax.legend(h1+h2,l1+l2,loc=0)
            axes[1,0].xaxis.set_major_formatter(fsf('%2.1e'))

            # sigma shape factor plots
            plt.sca(axes[1,1]);
            plt.title('Shape$_\sigma$')
            plt.plot(S_s, smids, label='S$_\sigma$', color='k')
            plt.plot(S_s*w_s, smids, label='S$_\sigma * \omega_\sigma$', color='fuchsia')
            plt.plot(S_s_new*w_s_new, S_smids_new, label='new S$_\sigma * \omega_\sigma$', color='orange')
            plt.plot(S_s*w_s_2, smids, '--', label='old product', color='cyan')
            plt.legend(loc=0)
            plt.ylim([1.05,-0.05])
            plt.ylabel('$\sigma$')
            plt.xlabel('unitless')

            plt.suptitle('amf calculation factors')
            f.savefig(plotname)
            print('%s saved'%plotname)
            plt.close(f)
        return (AMF_s, AMF_z)


class Hemco_diag(GC_base):
    '''
        class just for Hemco_diag output and manipulation
    '''
    def __init__(self,day0, dayn=None, month=False):

        if __VERBOSE__:
            print('Reading Hemco_diag files:')

        # read data/attrs and initialise class:
        datadict,attrs=GC_fio.read_Hemco_diags(day0,dayn,month=month)

        #        ## OLD WAY: now use yearly megan output files
        #        # For each month read the data
        #        data_list=[]
        #        months=util.list_months(day0,dayn)
        #        for month in months:
        #            data,attrs=GC_fio.read_Hemco_diags(month,util.last_day(month))
        #            data_list.append(data)
        #        datadict={}
        #
        #        # Combine the data
        #        for key in data_list[0].keys():
        #            if __VERBOSE__:
        #                print("Reading ",key, np.shape(data_list[0][key]))
        #
        #            # Read the dimensions
        #            if key in ['lat','lon']:
        #                datadict[key] = data_list[0][key]
        #            elif (key in ['time','ISOP_BIOG']):
        #
        #                # np array of the data [time, lats, lons]
        #                data=np.array(data_list[0][key])
        #
        #                # for each extra month, append onto time dim:
        #                for i in range(1,len(months)):
        #                    data=np.append(data,np.array(data_list[i][key]),axis=0)
        #
        #                datadict[key]=data
        #
        #            elif __VERBOSE__:
        #                print("KEY %s not being read from E_new dataset"%key )
        #
        #        attrs['init_date']=day0
        #        attrs['n_dims']=len(np.shape(datadict['ISOP_BIOG']))

        super(Hemco_diag,self).__init__(datadict,attrs)

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
        
        # LETS REMOVE OCEAN ZEROS, REPLACE WITH NANS
        oceanmask=util.oceanmask(self.lats,self.lons)
        oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],self.n_dates,axis=0)
        self.E_isop_bio[oceanmask3d] = np.NaN
        self.oceanmask=oceanmask
        self.oceanmask3d=oceanmask3d
        
        if __VERBOSE__:
            print("E_isop_bio oceanic squares replaced with np.NaN")
            for k in datadict:
                print('  %10s %10s %s'%(k, np.shape(datadict[k]), attrs[k]))

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
            # iterating over hours, handle case where we reach end of day or end of month or end of year
            if (date.day > prior_day.day) or (date.month > prior_day.month) or (date.year > prior_day.year):
                prior_day=date
                if (not np.all(sanity[di]==1)):# or (np.any(np.isnan(out[di]))):
                    if not np.all(sanity[di]==1):
                        print('ERROR: should get one hour from each day!')
                    # has sanity not been set? has something created a NAN output?
                    # Oceanic squares are NaN now, rather than zero.
                    #if np.any(np.isnan(out[di])):
                    #    print("ERROR: Should not have NANs in MEGAN output")
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

    def daily_averaged(self):
        '''
            return days, E_isop_bio (average of each day )
            RIGHT NOW JUST 24 HOURS FROM START...
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
        if hasattr(self, 'E_isop_bio_daily'):
            return days, self.E_isop_bio_daily

        isop=self.E_isop_bio # atomC/cm2/s
        out=np.zeros([len(days),len(self.lats),len(self.lons)])+np.NaN

        for i in np.arange(0,len(dates),24):

            # local time is gmt+offset (24 hour time)
            #LT=(GMT+LTO + 24) % 24 # 91,144 (lats by lons)

            #print(date.strftime('%m%dT%H'),'   lon matches:',np.sum(LT[0,:]==hour))

            out[di] = np.nanmean(isop[i:i+24],axis=0)
            # need to split into 15 degree portions to do it properly?

            di = di+1

        self.E_isop_bio_LT=np.squeeze(out)
        self.attrs['E_isop_bio_daily']={'desc':'24 hour averaged E_isop_bio',
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
        self.sat_out=GC_sat(month, dayN=util.last_day(month) ,run='biogenic')
        self.dates=self.sat_out.dates

    def model_slope(self, region=pp.__AUSREGION__, overpass_hour=13, return_X_and_Y=False):
        '''
            Use RMA regression between E_isop and tropcol_HCHO to determine S:
                HCHO = S * E_isop + b
            Notes:
                Slope = Yield_isop / k_hcho
                HCHO: molec/cm2
                E_isop: atomC/cm2/s
                b: molec/cm2
                S: s
                Slope_SF = Slope after filtering by the smear (\hat{S})

            Return {'lats','lons','r':regression, 'b':bg, 'slope':slope, 'slope_sf'}

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



        O_hcho=sat_out.O_hcho # should be very similar to hcho molec/cm2
        if __VERBOSE__:
            hcho = sat_out.get_trop_columns(keys=['hcho'])['hcho'] # ppbv -> molec/cm2
            print("satellite output: trop column vs total column")
            print("    %.2e    %.2e "%(np.nanmean(hcho),np.nanmean(O_hcho)))
        hcho = O_hcho # We will apply the slope to total columns from OMHCHORP.

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

        if __VERBOSE__:
            print("in slope function: ")
            print("  nanmean E_isop = %.2e %s"%(np.nanmean(isop),'atom C/cm2/s'))
            print("  nanmean trop_hcho = %.2e %s"%(np.nanmean(hcho),'molec/cm2'))

        # calculate slope etc
        slope, bg, reg, err = slope_calc(isop,hcho)

        # again for smear filtered
        # Read smearing mask from file
        d0,dN = self.dates[0], self.dates[-1]
        smearmask, smeardates, smearlats, smearlons = masks.get_smear_mask(d0,dN,region=region)
        assert np.all(smearlats == sublats), 'smear lats are different to model lats '+str(smearlats)+str(sublats)
        assert np.all(smearlons == sublons), 'smear lons are different to model lons '+str(smearlons)+str(sublons)
        # apply to isop and hcho
        isopsf = np.copy(isop)
        isopsf[smearmask] = np.NaN
        hchosf = np.copy(hcho)
        hchosf[smearmask] = np.NaN

        # calculate slope etc
        slopesf, bgsf, regsf, errsf = slope_calc(isopsf,hchosf)


        if __VERBOSE__:
            print('GC_tavg.model_yield() calculates avg. slope of %.2e'%np.nanmean(slope))

        # Return all the data and the lats/lons of our subregion:
                            # lats and lons for slope, (and region for later)
        self.modelled_slope={'lats':sublats,'lons':sublons, 'region':region,
                             # indexes of lats/lons for slope
                             'lati':lati, 'loni':loni,
                             # regression, background, and slope
                             'r':reg, 'b':bg, 'slope':slope, 'err':err,
                             'rsf':regsf, 'bsf':bgsf, 'slopesf':slopesf, 'errsf':errsf}
        if return_X_and_Y:
            self.modelled_slope['hcho']=hcho
            self.modelled_slope['isop']=isop
            self.modelled_slope['hchosf']=hchosf
            self.modelled_slope['isopsf']=isopsf
        return self.modelled_slope

################
###FUNCTIONS####
################


def check_units(d=datetime(2005,1,1)):
    '''
        N_air (molecs/cm3)
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
        nair=gcm['N_air'] # molec/cm3

        # Mean surface HCHO in ppbv
        hcho=np.mean(hcho[:,:,0])
        print("Surface HCHO in ppbv: %6.2f"%hcho)

        # N_air is molec/m3 in User manual, and ncfile: check it's sensible:
        nair=np.mean(nair[:,:,0])
        airmass=nair/N_ave * airkg  * 1e6 # kg/cm3 * cm3/m3 at surface
        print("Mean surface N_air=%e molec/cm3"%nair)
        print(" = %.3e mole/cm3, = %4.2f kg/m3"%(nair/N_ave, airmass ))
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

def slope_calc(isop,hcho):
    '''
        Function that calculate slope on 3d array inputs isop and hcho
    '''
    arrshape=np.shape(isop)
    #time, lats, lons
    n_t, n_y, n_x= arrshape
    
    slope  = np.zeros([n_y,n_x]) + np.NaN
    bg     = np.zeros([n_y,n_x]) + np.NaN
    reg    = np.zeros([n_y,n_x]) + np.NaN
    err = np.zeros([n_y,n_x,2]) + np.NaN
    
    # regression for each lat/lon gives us slope
    for xi in range(n_x):
        for yi in range(n_y):
            # Y = m X + B
            X=isop[:, yi, xi]
            Y=hcho[:, yi, xi]

            # Skip ocean or no emissions squares:
            # potential problem: When using kgC/cm2/s, we are always close to zero (1e-11 order)
            # however we are using molec/cm2/s
            if np.isclose(np.mean(X), 0.0): continue
            if np.all(np.isnan(X)): continue


            # get regression
            m, b, r, CI1, CI2=RMA(X, Y)
            slope[yi, xi] = m
            bg[yi, xi] = b
            reg[yi, xi] = r
            err[yi,xi] = CI1[0] # slope limits (CI: ricker method)

    return slope,bg,reg,err

def check_diag(d=datetime(2005,1,1)):
    '''
    '''
    gc=GC_tavg(d)
    E_isop_hourly=gc.E_isop_bio
    print(E_isop_hourly.shape())
    E_isop=gc.get_daily_E_isop()
    print(E_isop.shape())


def create_slope_file(d0=datetime(2005,1,1),dN=datetime(2012,12,31), region=pp.__AUSREGION__):
    '''
        Store slope before/after mya and smear filter is applied
    '''
    # date range as string
    dstr=d0.strftime('%Y%m%d')+'-'+dN.strftime('%Y%m%d')
    outfilename='Data/GC_Output/slopes.h5'

    # first read biogenic model month by month and save the slope, and X and Y for wollongong
    allmonths = util.list_months(d0,dN)
    alldays   = util.list_days(d0,dN)

    # things to store each month
    s           = []
    s_sf        = []
    h, h_sf     = [], [] # HCHO
    e, e_sf     = [], [] # e_isop
    r, r_sf     = [], [] # regression coeff
    ci, ci_sf   = [], [] # confidence interval for slope
    n, n_sf     = [], [] # number of entries making slope

    for i,month in enumerate(allmonths):
        # Retrieve data
        GC=GC_biogenic(month)

        # Get slope and stuff we want to plot
        model = GC.model_slope(return_X_and_Y=True, region=region)
        #di       = util.date_index(dates[0],alldays,dates[-1])
        #lati,loni = util.lat_lon_index(lat,lon, model['lats'],model['lons'])
        outlat,outlon = model['lats'], model['lons']

        h.append(model['hcho'])
        h_sf.append(model['hchosf'])
        e.append(model['isop'])
        e_sf.append(model['isopsf'])
        #print(type(model['hchosf']))
        #print(type(h_sf[-1]))
        #print(np.shape(h_sf[-1]))


        # Y=slope*X+b with regression coeff r
        r.append(model['r'])
        r_sf.append(model['rsf'])
        ci.append(model['err'])
        ci_sf.append(model['errsf'])
        s.append(model['slope'])
        s_sf.append(model['slopesf'])
        nan     = np.isnan(h[-1]) + np.isnan(e[-1])
        nansf   = np.isnan(h_sf[-1]) + np.isnan(e_sf[-1])
        n.append(np.sum(~nan, axis = 0))
        n_sf.append(np.sum(~nansf, axis = 0))
        print("CHECK: Shapes ",month)
        for arr in h,e,r,ci,n:
            print('  ',type(arr), np.shape(arr))
            print('    ',type(arr[-1]), np.shape(arr[-1]))

    # combine into arrays with time dim
    isop   = e[0]
    isopsf = e_sf[0]
    hcho   = h[0]
    hchosf = h_sf[0]
    for arr in e[1:]:
        isop = np.append(isop,arr,axis=0)
    for arr in e_sf[1:]:
        isopsf=np.append(isopsf,arr,axis=0)
    for arr in h[1:]:
        hcho=np.append(hcho,arr,axis=0)
    for arr in h_sf[1:]:
        hchosf=np.append(hchosf,arr,axis=0)

    # monthly metrics are easier to combine
    r       = np.stack(r,axis=0)
    r_sf    = np.stack(r_sf,axis=0)
    ci      = np.stack(ci,axis=0)
    ci_sf   = np.stack(ci_sf,axis=0)
    n       = np.stack(n,axis=0)
    n_sf    = np.stack(n_sf,axis=0)

    print("CHECK: FINAL Shapes ",month)
    for arr in hcho,isop,r,ci,n:
        print(np.shape(arr))

    # group into months and find slope on all januarys, ...
    shape=[12,len(outlat),len(outlon)]
    shapeci=[12,len(outlat),len(outlon),2]
    n_mya, n_sf_mya   = np.zeros(shape),np.zeros(shape)
    ci_mya, ci_sf_mya = np.zeros(shapeci),np.zeros(shapeci)
    s_mya, s_sf_mya   = np.zeros(shape),np.zeros(shape)
    r_mya, r_sf_mya   = np.zeros(shape),np.zeros(shape)

    for month in range(12):
        #mi = np.array([d.month == month+1 for d in allmonths])
        di = np.array([d.month == month+1 for d in alldays])
        X = isop[di]
        Y = hcho[di]
        X_sf = isopsf[di]
        Y_sf = hchosf[di]
        n_mya[month] = np.sum(~np.isnan(X*Y),axis=0)
        n_sf_mya[month] = np.sum(~np.isnan(X_sf*Y_sf),axis=0)

        s_mya[month], bg, r_mya[month], ci_mya[month] = slope_calc(X,Y)
        s_sf_mya[month], bg, r_sf_mya[month], ci_sf_mya[month] = slope_calc(X_sf,Y_sf)

    print("CHECK: FINAL Shapes ")
    for arr in s_mya,s_sf_mya,r_mya,ci_mya,n_mya:
        print(np.shape(arr))

    # finally save file
    arraydict={'slope':s, 'slope_sf':s_sf, 'slope_mya':s_mya, 'slope_sf_mya':s_sf_mya,
               'r':r, 'r_sf':r_sf, 'r_mya':r_mya, 'r_sf_mya':r_sf_mya,
               'ci':ci, 'ci_sf':ci_sf, 'ci_mya':ci_mya, 'ci_sf_mya':ci_sf_mya,
               'n':n, 'n_sf':n_sf, 'n_mya':n_mya, 'n_sf_mya':n_sf_mya,
               'dates':util.gregorian_from_dates(allmonths),
               'lats':outlat, 'lons':outlon}
    attrdict={'slope':{'units':'s', 'desc':'slope between biogenic emissions and HCHO from GEOS chem'},
              'r':{'desc':'RMA regression coefficient of slope'},
              'ci':{'desc':'95% CI for slope'},
              'n':{'desc':'entries making up slope'},
              'dates':{'desc':'gregorian dates, hours since 1985,1,1,0,0'},
              'lats':{'desc':'lat midpoints'},
              'lons':{'desc':'lon midpoints'},
              }

    fio.save_to_hdf5(outfilename, arraydict, fillvalue=np.NaN,
                 attrdicts=attrdict, fattrs={'date coverage':dstr},
                 verbose=__VERBOSE__)




if __name__=='__main__':
    #check_diag()
    #check_units(datetime(2005,2,1))

    print("shouldn't be here")

