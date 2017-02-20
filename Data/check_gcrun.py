# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslade Feb 6 2015

Check the ncfiles created by bpch2coards
'''
## Modules
# plotting module, and something to prevent using displays(can save output but not display it)

# module for hdf eos 5
import h5py 
import numpy as np
from datetime import datetime, timedelta
from glob import glob

path='/home/jesse/Desktop/Repos/OMI_regridding/Data/'
def read_trac_avg_noisop(date=datetime(2005,1,1)):
    ''' read a noisop output file'''
    fname=path+'gcrun_noisop/trac_avg.geos5_2x25_tropchem.%s000000.hdf'%date.strftime("%Y%m%d")
    print(fname)
    file=h5py.File(fname,'r')
    print(file)
    
read_trac_avg_noisop()
