#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:02:39 2018

@author: jesse
"""
from datetime import datetime
import csv
import numpy as np
import os.path, time

# fio and data classes
from utilities import fio, GC_fio
from classes import GC_class

##################
### GLOBALS ######
##################

__keysofinterest__ = ['hcho','isop','boxH','psurf','lats','lons','OH','NO2',
                      'AD','N_air','N_hcho',
                      'O_hcho','O_air','Shape_s','Shape_z',
                      'E_isop_bio']
__attrsofinterest__ = ['full_name', 'standard_name', 'units',
                       'original_shape','axis']

def summarise_class(dataclass, classname, keys=__keysofinterest__):
    lines=[]
    lines.append("===========================================\n")
    lines.append("======= %15s =======\n"%classname)
    lines.append("===========================================\n")
    lines.append("Modified: %s \n"%(str([mod_time for mod_time in dataclass.modification_times])))
    lines.append("=============\n")
    for key in keys:
        if key in vars(dataclass).keys():
            data=getattr(dataclass,key)
            surfmean=np.nanmean(data)
            if len(data.shape) == 3:
                surfmean=np.nanmean(data[:,:,0])
            elif len(data.shape) == 4:
                surfmean=np.nanmean(data[:,:,:,0])
            lines.append("%s %s \n     surface mean=%.2e\n"%(key, str(data.shape), surfmean))
            for k,v in dataclass.attrs[key].items():
                if k in __attrsofinterest__:
                    lines.append('    %15s:%15s\n'%(k, v))
    return lines

def write_GC_units(day=datetime(2005,1,1)):
    '''
        Check unites read in from GEOS-Chem satellite, tavg, and HEMCO_DIAG files
    '''
    outfname="Data/GC_units_summary.txt"
    outfile=open(outfname,"w")

    # Summarise GC_sat datafiles
    for runtype in ['tropchem','halfisop','biogenic']:
        dat = GC_class.GC_sat(day,run=runtype)
        outfile.writelines(summarise_class(dat,"GC sat_output (%s)"%runtype))

    # Summarise GC_tavg datafiles
    for runtype in ['tropchem','halfisop','nochem']:
        dat = GC_class.GC_tavg(day,run=runtype)
        outfile.writelines(summarise_class(dat,"GC tavg (%s)"%runtype))

    # Look at HEMCO diagnostic outputs
    dat = GC_class.Hemco_diag(day)
    outfile.writelines(summarise_class(dat,"GC Hemco_diag"))

    print("SAVED FILE: ",outfname)
    outfile.close()

def test_fires_fio():
    '''
    Test 8 day average fire interpolation and fio
    '''
    day = datetime(2005,1,1)
    ## get normal and interpolated fire
    orig, lats, lons=fio.read_8dayfire(day)

    lonres,latres = 0.3125, 0.25
    regr, lats2, lons2=fio.read_8dayfire_interpolated(day, latres=latres, lonres=lonres)

    check_array(orig)
    check_array(regr)

    # Number checks..
    assert np.max(orig) == np.max(regr), "Maximum values don't match..."
    print ("Mean orig = %4.2f\nMean regridded = %4.2f"%(np.mean(orig[orig>-1]), np.mean(regr[regr>-1])))

    ## PLOTTING
    ##

    # EG plot of grids...
    plt.figure(0,figsize=(10,8))
    # tasmania lat lon = -42, 153
    m0=Basemap(llcrnrlat=-45, urcrnrlat=-37, llcrnrlon=142, urcrnrlon=150,
              resolution='i',projection='merc')
    m0.drawcoastlines()
    m0.fillcontinents(color='coral',lake_color='aqua')
    m0.drawmapboundary(fill_color='white')

    # original grid = thick black
    d=[1000,1]
    m0.drawparallels(lats-0.25, linewidth=2.5, dashes=d)
    m0.drawmeridians(lons-0.25, linewidth=2.5, dashes=d)
    # new grid = thin brown
    m0.drawparallels(lats2-latres/2.0, color='brown', dashes=d)
    m0.drawmeridians(lons2-lonres/2.0, color='brown', dashes=d)
    plt.title('Original grid(black) vs new grid(red)')
    plt.savefig('Figs/AQUAgrids.png')
    plt.close()
    ## Regridded Data plot comparison
    # plot on two subplots
    fig=plt.figure(1, figsize=(14,9))
    fig.patch.set_facecolor('white')

    # mesh lon/lats
    mlons,mlats = np.meshgrid(lons,lats)
    mlons2,mlats2 = np.meshgrid(lons2,lats2)
    axtitles=['original (0.5x0.5)',
              'regridded to %1.4fx%1.4f (latxlon)' % (latres,lonres)]

    ax1=plt.subplot(121)
    m1,cs1, cb1 = pp.linearmap(orig,mlats,mlons, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)
    ax2=plt.subplot(122)
    m2,cs2, cb2 = pp.linearmap(regr,mlats2,mlons2, vmin=1, vmax=10,
        lllat=-57, urlat=1, lllon=110, urlon=170)

    for i in range(2):
        [ax1,ax2][i].set_title(axtitles[i])
        [cb1, cb2][i].set_label('Fire Count (8 day)')

    plt.suptitle('AQUA 2005001',fontsize=20)
    plt.tight_layout()
    plt.savefig('Figs/AQUA2005001.png')
    plt.close()

def read_multiple_years():
    '''
        Test reading of multiple years of data
    '''
    
    jan1_05 = datetime(2005,1,1)
    dec1_05 = datetime(2005,12,1)
    dec31_05 = datetime(2005,12,31)
    jan1_06 = datetime(2006,1,1)
    dec31_06 = datetime(2006,12,31)
    
    # Works for new_emissions, something up with tropchem output!?
    for run in ['tropchem',]: #['new_emissions','tropchem']:
        # can read span over 05/06?
        print("READING: %s run dec05->jan06"%run)
        tropchem_sat = GC_class.GC_sat(dec31_05,dayN=jan1_06,run=run)
        print("WORKED: %s run dec05->jan06"%run)
        # can read multiple years?
        print("READING: %s run jan05->dec06"%run)
        tropchem_sat = GC_class.GC_sat(jan1_05,dayN=dec31_06,run=run)
        print("WORKED: %s run jan05->dec06"%run)

    
