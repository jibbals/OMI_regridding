#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO,GC_fio,fio


from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.gchcho import gchcho
import xbpch
import xarray

###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)
d1=datetime(2005,2,1)
region=pp.__AUSREGION__

dstr=d0.strftime("%Y%m%d")
yyyymm=d0.strftime("%Y%m")


satname="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.20050101.bpch"
satnames="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s*.bpch"%yyyymm
tracfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/tracerinfo.dat'
diagfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/diaginfo.dat'
Hemco_diag="Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.200501010100.nc"


dat,att=GC_fio.read_Hemco_diags(d0)
e=dat['ISOP_BIOG']
lons=dat['lon']
a=att['ISOP_BIOG']
e.shape # time, lat, lon
np.nanmean(e)
for k in a:
    print(k,':',a[k])
for k in dat:
    print(k, ':', dat[k].shape)


hd=GC_class.Hemco_diag(d0,month=False)
#hd.plot_daily_emissions_cycle()
enew=hd.daily_LT_averaged()
lats=hd.lats
lons=hd.lons
pp.basicmap(enew,lats,lons,aus=True,pname='test.png')
print(np.nanmean(enew))

