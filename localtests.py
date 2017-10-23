#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO
from utilities import GC_fio

from classes.E_new import E_new # E_new class
from classes.GC_class import GC_tavg

###############
### Globals ###
###############
__VERBOSE__=True


#####
## DO STUFF
#####
d0=datetime(2005,1,1)
dn=datetime(2005,2,1)
region=pp.__AUSREGION__

#tavg,attrs=GC_fio.read_tavg(d0)
#d=tavg['time'][0]
half=GC_tavg(d0,run='halfisop')
full=GC_tavg(d0,run='tropchem')
print("GC_tavg read")

print(half.attrs)


