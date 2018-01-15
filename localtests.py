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
d1=datetime(2005,1,31)
region=pp.__AUSREGION__

omi=omhchorp(d0,d1,keylist=['VCC','gridentries'],ignorePP=True)
omi.lats.shape
vccavg=np.nanmean(omi.VCC,axis=0)
lowvcc=util.regrid_to_lower(vccavg,omi.lats,omi.lons,GMAO.lats_e,GMAO.lons_e)

vmin,vmax=1e14,1e17
linear=False

plt.figure()
plt.subplot(211)
pp.createmap(vccavg,omi.lats,omi.lons,linear=linear,vmin=vmin,vmax=vmax)
plt.subplot(212)
pp.createmap(lowvcc,GMAO.lats_m,GMAO.lons_m,linear=linear,vmin=vmin,vmax=vmax)

#data,attrs=fio.read_omno2d(d0)
#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')
