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
import tests

from classes.E_new import E_new # E_new class
from classes import GC_class
from classes.omhchorp import omhchorp
from classes.gchcho import gchcho
from classes.campaign import campaign
import xbpch
import xarray
import pandas as pd

import timeit

###############
### Globals ###
###############
__VERBOSE__=True

region=pp.__AUSREGION__

#####
## DO STUFF
#####
d0=datetime(2005,1,1)
dstr=d0.strftime('%Y%m%d')
mstr=d0.strftime('%Y%m')
latres=0.25
lonres=0.3125

dN=datetime(2005,1,5)

dates=util.list_days(d0,dN,month=False)


om=fio.read_omhchorp(datetime(2004,12,26),datetime(2005,1,1),keylist=['fires'],latres=latres,lonres=lonres)

# masks here made using default values...
start1=timeit.default_timer()
#firemask,_fdates,_flats,_flons=fio.make_fire_mask(d0, latres=latres,lonres=lonres)
start2=timeit.default_timer()
smokemask,_sdates,_slats,_slons=fio.make_smoke_mask(d0, latres=latres,lonres=lonres)
start3=timeit.default_timer()
#anthmask,_adates,_alats,_alons=fio.make_anthro_mask(d0,latres=latres,lonres=lonres)
end=timeit.default_timer()

#print("TIME: %6.2f seconds for firemask"%(start2-start1))
print("TIME: %6.2f seconds for smokemask"%(start3-start2))
#print("TIME: %6.2f seconds for anthromask"%(end-start3))


def emisssions_vs_firefilter(d0=datetime(2005,1,1)):
    '''
    '''
    Enew=E_new(d0,)
    # mean Emissions estimates vs eachother

    f,axes = plt.subplots(3,3,figsize=(15,15))
    region=[Enew.lats[0],Enew.lons[0],Enew.lats[-1],Enew.lons[-1]]

    lats=Enew.lats
    lons=Enew.lons
    arrs=[[Enew.E_VCC_OMI, Enew.E_VCC, Enew.E_VCC_PP],
          [Enew.E_VCC_OMI_f, Enew.E_VCC_f, Enew.E_VCC_PP_f],
          []]

    labels=[['OMI','GC','PP'],
            ['OMI_f','GC_f','PP_f'],
            ['fire-nofire','fire-nofire','fire-nofire']]

    linear=False
    vmin=1e10
    vmax=1e13
    for i in range(3):
        for j in range(3):
            plt.sca(axes[i,j])
            if i < 2:
                arr=np.nanmean(arrs[i][j], axis=0) # average over time
            elif i==2:
                arr=np.nanmean(arrs[1][j], axis=0) - np.nanmean(arrs[0][j], axis=0)
                vmin=-5e12; vmax=5e12; linear=True

            pp.createmap(arr,lats,lons,title=labels[i][j],
                         region=region,
                         vmin=vmin,vmax=vmax,linear=linear)

    pname='Figs/Emiss/FireFilter%s.png'%mstr
    plt.suptitle('Emissions with and without fire filter %s'%mstr,fontsize=25)
    plt.savefig(pname)
    plt.close()



# to be moved somewhere:
def firetest():
    fires_per_area,lats,lons=fio.read_MOD14A1(d0,True)
    fires,lats,lons=fio.read_MOD14A1(d0,False)
    earth_sa=510e6 # 510.1 million km2
    count_a=np.sum(fires)
    count_b=np.mean(fires_per_area)*earth_sa*1e3
    print(count_a,count_b)
    print((count_a-count_b)/count_b)

    region=[-20,-30,40,50]
    f,axes=plt.subplots(3,1)
    plt.sca(axes[0])
    pp.createmap(fires,lats,lons,title='MODIS Fires 20050102',
                 colorbar=None, region=region,
                 linear=False, vmin=1,vmax=3e6)

    # lats lons are .1x.1 degrees
    # Try lower and higher resolution function:
    hlats,hlons,hlat_e,hlon_e = util.lat_lon_grid(latres=0.25,lonres=0.3125)
    llats,llons,llat_e,llon_e = util.lat_lon_grid(latres=2.0,lonres=2.5)

    hfires=util.regrid(fires,lats,lons,hlats,hlons,groupfunc=np.nansum)
    lfires=util.regrid(fires,lats,lons,llats,llons,groupfunc=np.nansum)
    print(np.nansum(hfires))
    print(np.nansum(lfires))
    plt.sca(axes[1])
    pp.createmap(hfires,hlats,hlons,title='High res',
                 colorbar=False,region=region,
                 linear=False, vmin=1,vmax=3e6)

    plt.sca(axes[2])
    pp.createmap(lfires,llats,llons,title='Low res', region=region,
                 clabel='fire pixels', pname='test_fires.png',
                 linear=False,cmapname='Reds',vmin=1,vmax=3e6)


#
#pp.createmap(data['tropno2'],data['lats'],data['lons'],vmin=1e13, vmax=1e16,pname='testno2.png',
#             title='OMNO2d for 2005, jan, 1',clabel='trop NO2 (molec/cm2)')
