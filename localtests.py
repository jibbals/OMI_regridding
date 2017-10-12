#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:15:43 2017

@author: jesse
"""

from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# local modules
import utilities.utilities as util
import utilities.plotting as pp
from utilities import GMAO
#from utilities import fio

from classes.E_new import E_new # E_new class


###############
### Globals ###
###############
__VERBOSE__=True


d0=datetime(2005,1,1)
dn=datetime(2005,2,1)
region=pp.__AUSREGION__

e=E_new(d0)
d,isop=e.get_series('E_isop',maskocean=False,region=region,testplot=True)
d,isop2=e.get_series('E_isop',maskocean=True,region=region)
plt.plot(isop,label='original')
plt.plot(isop2,label='ocean masked')
plt.legend()