#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO
from utilities import GC_fio

from classes.E_new import E_new # E_new class
from classes.GC_class import GC_tavg
from classes.gchcho import gchcho
import xbpch


###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)
region=pp.__AUSREGION__

dstr=d0.strftime("%Y%m%d")
yyyymm=d0.strftime("%Y%m")

satnames="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s*.bpch"%yyyymm
tracfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/tracerinfo.dat'
diagfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/diaginfo.dat'

# READ MULTIPLE SAT OUT FILES:
# dask needs to be explicitely true
sat_m=xbpch.open_mfbpchdataset(satnames,tracerinfo_file=tracfile,diaginfo_file=diagfile, decode_cf=False,dask=True)
sat_d=None
print(sat_m.coords)
print(sat_d)

pname='testplot.png'




