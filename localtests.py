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
#full=GC_tavg(d0,run='tropchem')
print("GC_tavg read")
print(half.attrs)


lats=half.lats
lons=half.lons
lats_e=half.lats_e
lons_e=half.lons_e

print("hcho:",half.hcho.shape,np.mean(half.hcho))
half_month=half.month_average(keys=['hcho','O_hcho','E_isop_bio'])
h_O_hcho=half_month['O_hcho'] # molec/cm2
h_hcho=half_month['hcho'][:,:,0] # surface ppbv
h_E_isop=half_month['E_isop_bio'] # molec/cm2/s
print('hcho:',h_hcho.shape,np.mean(h_hcho))
print(half.attrs['hcho'])
print('E_isop:',h_E_isop.shape,np.mean(h_E_isop))
print(half.attrs['E_isop_bio'])

#S= (f_hcho - h_hcho) / (f_E_isop - h_E_isop) # s

#print("emissions from Full,Half:",np.sum(f_E_isop),np.sum(h_E_isop))
#print("O_hcho from Full, Half:",np.sum(f_hcho),np.sum(h_hcho))
#S[f_E_isop == 0.0] = np.NaN

#print("S shape:",S.shape)
#print("Average S:",np.nanmean(S))

region=pp.__AUSREGION__
dstr=d0.strftime("%Y%m%d")

pname='testplot.png'
print(lats,lons)
print(h_hcho.T)
pp.basicmap(h_hcho.T,lats_e,lons_e,linear=True,pname=pname,title='f_hcho (ppbv)')

pp.basicmap(h_O_hcho.T, lats,lons,linear=True,pname='testplot2.png',title='O_hcho (molec/cm2)')



