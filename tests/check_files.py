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