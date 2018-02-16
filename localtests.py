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
import pandas as pd

###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)
mydfile='Data/MOD14A1_D_FIRE/2005/MOD14A1_D_FIRE_2005-01-02.CSV'
myd=pd.read_csv(mydfile)

myd.shape
data=myd.values
data[data>9000]=np.NaN # ocean squares!

# assume leftmost bottom is 0,0
lats=np.linspace(89.9,-89.9,1799)
lons=np.linspace(-180,179.9,3600)

pp.createmap(data,lats,lons,title='MODIS Fires 20050102',
             pname='test_fires.png', clabel='fire pixels/1000km$^2$',
             linear=True,)# cmapname='Reds')

#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')
