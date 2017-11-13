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
from classes.GC_class import GC_sat, GC_tavg
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

#HD=fio.read_netcdf(Hemco_diags)
#print(HD)
#
#Hemco_diags="Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.200501*.nc"
#HD=xarray.open_mfdataset(Hemco_diags)
#print(HD)

dat,att=GC_fio.read_Hemco_diags(d0)
e=dat['ISOP_BIOG']
a=att['ISOP_BIOG']
e.shape # time, lat, lon
np.nanmean(e)
for k in a:
    print(k,':',a[k])


