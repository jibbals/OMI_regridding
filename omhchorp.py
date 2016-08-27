import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
from datetime import datetime
from glob import glob
from scipy.interpolate import RectBivariateSpline as RBS
import h5py


class omhchorp:
    '''
    Class for holding OMI level 2 ungridded dataset
    Structure containing 
        double AMF_GC(lats, lons)
        double AMF_OMI(lats, lons)
        double AMF_SSD(lats, lons)
        double ColumnAmountHCHO(lats, lons) ;
        double ColumnAmountHCHO_OMI(lats, lons) ;
        double ColumnAmount_SSD(lats, lons) ;
        int64 GridEntries(lats, lons) ;
        double Latitude(lats) ;
        double Longitude(lons) ;
        double ScatteringWeight(lats, lons, 47) ;
        double ShapeFactor_GC(72, lats, lons) ;
        double ShapeFactor_OMI(lats, lons, 47) ;
        double Sigma_GC(72, lats, lons) ;
    '''
    # date and dimensions
    date=0
    Latitude=0
    Longitude=0
    Sigma_GC=0 # [ 72, lats, lons ]
    
    # Arrays [ lats, lons ]
    AMF_GC=0
    AMF_OMI=0
    AMF_SSD=0
    ColumnAmountHCHO=0 
    ColumnAmountHCHO_OMI=0
    ColumnAmount_SSD=0
    GridEntries=0
    # [72, lats, lons]
    ShapeFactor_GC=0
    # [lats, lons, 47]
    ScatteringWeight=0
    ShapeFactor_OMI=0
