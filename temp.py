#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:00:10 2017

@author: jesse
"""

## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
import fio
import reprocess
import plotting as pp
from JesseRegression import RMA
from omhchorp import omhchorp as omrp
from gchcho import match_bottom_levels


import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy

from datetime import datetime#, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
#import matplotlib.patches as mpatches

import timeit
import random

Ohcho='$\Omega_{HCHO}$'
Ovc='$\Omega_{VC}$'
Ovcc='$\Omega_{VCC}$'
Oomi='$\Omega_{OMI}$'
Ogc='$\Omega_{GC}$'

date=datetime(2005,1,1); lllat=-80; lllon=-179; urlat=80; urlon=179; pltname=""

# Grab OMI data
fname=fio.determine_filepath(date)[0]
print(fname)
print('reading swath')
omi_s=fio.read_omhcho(fname)
print('reading 1 day')
omi_1=fio.read_omhcho_day(date)
print('reading 8 days')
omi_8=fio.read_omhcho_8days(date)


counts= omi_8['counts']
print( "at most %d entries "%np.nanmax(counts) )
print( "%d entries in total"%np.nansum(counts) )
lonmids=omhchorp['longitude']
latmids=omhchorp['latitude']
lons,lats = np.meshgrid(lonmids,latmids)

# Plot
# SC, VC_omi, AMF_omi
# VCC, VC_gc, AMF_GC
# VC_OMI-GC, VC_GC-GC, GC_map
# cuncs, AMF-correlation, AMF_GCz
