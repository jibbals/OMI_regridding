#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
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
import pandas as pd

import regrid_swaths as rs

###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)



fires_per_area,lats,lons=fio.read_MOD14A1(d0,True)
fires,lats,lons=fio.read_MOD14A1(d0,False)
earth_sa=510e6 # 510.1 million km2
count_a=np.sum(fires)
count_b=np.mean(fires_per_area)*earth_sa*1e3
print(count_a,count_b)
print((count_a-count_b)/count_b)

pp.createmap(fires,lats,lons,title='MODIS Fires 20050102',
             pname='test_fires.png', clabel='fire pixels',
             linear=False,cmapname='Reds',vmin=1,vmax=3e6)

#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')
