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
fname='Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.20050101.bpch'
dname='Data/GC_Output/geos5_2x25_tropchem/satellite_output/diaginfo.dat'
tname='Data/GC_Output/geos5_2x25_tropchem/satellite_output/tracerinfo.dat'
keys=GC_fio.__sat_mainkeys__


#sat=GC_fio.read_bpch(fname,keys=keys)

sat=xbpch.open_bpchdataset(fname,tracerinfo_file=tname,diaginfo_file=dname,decode_cf=False)
sat=GC_fio.read_bpch(fname,keys)
print(sat.keys())



