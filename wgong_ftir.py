#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:53:47 2018

@author: jesse
"""

# to read hdf4 ... need to install pyhdf
from pyhdf import SD
import numpy as np
from utilities import fio

def hdf4_to_hdf5():
    '''
    Take all the hdf4 files and convert to hdf5
    '''
    # Original hdf paths
    datadir = 'Data/campaigns/Wgong/'
    paths   = {'groundbased_ftir.h2co_uow002_wollongong_20070808t005747z_20071231t033605z_007.hdf':'ftir_2007.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20080107t004723z_20081222t045218z_007.hdf':'ftir_2008.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20090107t003314z_20091224t022055z_007.hdf':'ftir_2009.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20100104t232758z_20101229t045540z_007.hdf':'ftir_2010.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20110115t221709z_20111224t051045z_007.hdf':'ftir_2011.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20120101t010158z_20121205t231012z_007.hdf':'ftir_2012.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20130110t234951z_20131230t070333z_007.hdf':'ftir_2013.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20140104t001457z_20141230t224148z_007.hdf':'ftir_2014.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20150102t195923z_20151230t001903z_007.hdf':'ftir_2015.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20160107t062738z_20161223t063416z_007.hdf':'ftir_2016.h5',
               'groundbased_ftir.h2co_uow002_wollongong_20180603t231815z_20180831t061149z_007.hdf':'ftir_2018.h5',}

    for path,newpath in paths.items():

        # will save some data and attributes into simplified hdf5 file..
        data,attrs = {},{}

        # read it all in
        hdfile = SD.SD(datadir+path)
        for key in hdfile.datasets():
            print(key)
            data[key] = hdfile.select(key)[:]
            attrs[key] = hdfile.select(key).attributes()

        # keep the global attributes also
        fattrs=hdfile.attributes()

        # save as hdf5
        fio.save_to_hdf5(datadir+newpath, data, attrdicts=attrs, fattrs=fattrs)
        print('saved ',datadir+newpath)

        # close the hdfile I think..
        hdfile.end()


