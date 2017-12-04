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
d1=datetime(2005,2,1)
region=pp.__AUSREGION__

dstr=d0.strftime("%Y%m%d")
yyyymm=d0.strftime("%Y%m")

pp.InitMatplotlib()

satname="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.20050101.bpch"
satnames="Data/GC_Output/geos5_2x25_tropchem/satellite_output/ts_satellite_omi.%s*.bpch"%yyyymm
tracfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/tracerinfo.dat'
diagfile='Data/GC_Output/geos5_2x25_tropchem/satellite_output/diaginfo.dat'
Hemco_diag="Data/GC_Output/geos5_2x25_tropchem_biogenic/Hemco_diags/E_isop_biog.200501010100.nc"
biosat_files="Data/GC_Output/geos5_2x25_tropchem_biogenic/satellite_output/sat_biogenic.%s*.bpch"%yyyymm

#dat,attr=GC_fio.read_bpch(path=biosat_files,keys=GC_fio.__sat_mainkeys__,multi=True)
#dat['IJ-AVG-$_CH2O'].shape

#Inversion.store_emissions_month(d1)

# test GC_tavg plotting
GC=GC_class.GC_sat(d0,)

#plt.subplot(121)
#pp.createmap(GC.O_hcho[0],GC.lats,GC.lons,aus=True,GC_shift=True,
#             title='O_hcho',clabel=GC.attrs['O_hcho']['units'])
#plt.subplot(122)
#pp.createmap(GC.N_air[0,:,:,0],GC.lats,GC.lons,aus=True,GC_shift=True,
#             clabel=GC.attrs['N_air']['units'], title='N_air',
#             linear=True)

# READ OMI
month=d0
dayn=util.last_day(month)
OMI=omhchorp(month,dayn=dayn)

# Check data
print ('OMI (VCC) molec/cm2',OMI.VCC.shape)
OMIhcho=OMI.time_averaged(month,dayn,keys=['VCC'])['VCC']# molec/cm2
print('month average globally:',np.nanmean(OMIhcho))

print("GC (O_hcho)",GC.attrs['O_hcho']['units'], GC.O_hcho.shape)
GChcho=np.nanmean(GC.O_hcho,axis=0) # time averaged for the month
print("month average globally:",np.nanmean(GChcho))

plt.figure(figsize=(12,12))
plt.subplot(221)
pp.createmap(GChcho,GC.lats,GC.lons,aus=True,GC_shift=True, linear=True,
             vmin=0,vmax=2e16, cmapname='rainbow',
             title=r'GC $\Omega_{HCHO}$', clabel=r'molec cm$^{-2}$')

plt.subplot(222)
pp.createmap(OMIhcho,OMI.lats,OMI.lons,aus=True, linear=True,
             vmin=0,vmax=2e16, cmapname='rainbow',
             title=r'OMI $\Omega_{HCHO}$',clabel=r'molec cm$^{-2}$')

# regrid GChcho onto higher resolution
lats,lons=OMI.lats,OMI.lons
GChcho=util.regrid(GChcho,GC.lats,GC.lons,lats,lons)

diff=GChcho-OMIhcho
rdiff=(GChcho-OMIhcho)/OMIhcho

plt.subplot(212)
pp.createmap(diff,lats,lons, region=[-50,100,-5,170], GC_shift=True, linear=True,
             vmin=-1.5e16, vmax=1.5e16, cmapname='seismic',
             title='GC - OMI')

#plt.subplot(224)
#pp.createmap(rdiff,lats,lons, aus=True, GC_shift=True, linear=True,
#             vmin=-1.5, vmax=1.5,cmapname='seismic',
#             title='(GC - OMI)/OMI')
#plt.suptitle("GEOS-Chem vs OMI total column HCHO")
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
pname='Figs/GC_vs_OMI_hcho.png'
plt.savefig(pname)
print("SAVED ",pname)
plt.close()

# test stupid LT avg function
#GC=GC_class.Hemco_diag(d0)
#days,isop=GC.daily_LT_averaged()
#
#isop=isop*GC.kgC_per_m2_to_atomC_per_cm2
#np.nanmean(isop)
#m,cs,cb=pp.basicmap(isop,GC.lats,GC.lons,linear=True)
#cs.set_clim(0,np.nanmax(isop)*0.9)
#GC=GC_class.GC_sat(d0,run='biogenic')
#GC=GC_class.GC_biogenic(d0)
#print(GC)

#
#dat,att=GC_fio.read_Hemco_diags(d0)
#e=dat['ISOP_BIOG']
#lons=dat['lon']
#a=att['ISOP_BIOG']
#e.shape # time, lat, lon
#np.nanmean(e)
#for k in a:
#    print(k,':',a[k])
#for k in dat:
#    print(k, ':', dat[k].shape)
#
#
#hd=GC_class.Hemco_diag(d0,month=False)
##hd.plot_daily_emissions_cycle()
#days,E_megan=hd.daily_LT_averaged()
#lats=hd.lats
#lons=hd.lons
#pp.basicmap(E_megan,lats,lons,aus=True,pname='test.png')
#print(np.nanmean(E_megan))

