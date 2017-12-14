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
import Inversion


from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.omhchorp import omhchorp
from classes.gchcho import gchcho
from classes.campaign import campaign
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

# Read campaigns:
SPS1=campaign()
SPS2=campaign()
SPS1.read_SPS(1)
SPS2.read_SPS(2)

f, (ax1,ax2) = plt.subplots(1,2,sharey=True)
plt.sca(ax1)
SPS1.plot_series("SPS1")
plt.sca(ax2)
SPS2.plot_series("SPS2",'Figs/SPS_Series.png')
pp.plot_daily_cycle(SPS1.dates,SPS1.hcho,overplot=True)
dstr=d0.strftime("%Y%m%d")
yyyymm=d0.strftime("%Y%m")

pp.InitMatplotlib()

satname="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.20050101.bpch"
satnames="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s*.bpch"%yyyymm
tracfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/tracerinfo.dat'
diagfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/diaginfo.dat'
Hemco_diag="Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.200501010100.nc"
biosat_files="Data/GC_Output/geos5_2x25_tropchem_biogenic/satellite_output/sat_biogenic.%s*.bpch"%yyyymm

###
# read biogenic output
OMI=omhchorp(d0,)
print('before:: ',np.nanmean(OMI.VCC))
OMI.apply_fire_mask('VCC')
print('after:: ',np.nanmean(OMI.VCC))

pp.createmap(OMI.fire_mask_8,OMI.lats,OMI.lons,linear=True)
plt.savefig('test_fires.png')
#GC=GC_class.GC_biogenic(d0)
#MS=GC.model_slope()

#enew=Inversion.Emissions_1day(d0,GC,OMI)
#enew.keys()
