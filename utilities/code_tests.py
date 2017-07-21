#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:03:42 2017

Code specifically to test various functions etc.

@author: jesse
"""

###############
### MODULES ###
###############

# global modules
#
import numpy as np


# local modules
#
# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

# my file reading library
#from utilities import fio
import utilities.utilities as util
sys.path.pop(0)

###############
### METHODS ###
###############

def check_regridding():
    '''
        check utilities.regrid()
    '''
    print('utilities.regrid tests running')

    low=3
    high=2*low
    highbad=2*low-1 # TESTS FAIL IN THIS CASE:
    latsl=np.linspace(1,low,low) # Start, Stop, Steps
    lonsl=np.linspace(1,low,low) # 1,2,3
    latsh=np.linspace(1,low,high)
    lonsh=np.linspace(1,low,high)
    latshb=np.linspace(1,low,highbad)
    lonshb=np.linspace(1,low,highbad)

    datal=np.zeros([low,low])
    for x in range(low):
        for y in range(low):
            datal[x,y] = x+low*y
    print(datal)
    # From low res to high res should work fine:
    datah= util.regrid(datal,latsl,lonsl,latsh,lonsh)
    datahb= util.regrid(datal,latsl,lonsl,latshb,lonshb)

    # Check first and last elements are the same:
    assert np.isclose(datal[0,0],datah[0,0]), "low to high res first elements differ"
    assert np.isclose(datal[-1,-1],datah[-1,-1]), "low to high res last elements differ"
    assert np.isclose(datal[0,0],datahb[0,0]), "low to high res(b) first elements differ"
    assert np.isclose(datal[-1,-1],datahb[-1,-1]), "low to high res(b) last elements differ"
    print("Low to high res test passes")

    # Check conversion back to orig is identical:
    datal2=util.regrid(datah, latsh,lonsh, latsl, lonsl)
    datal2b=util.regrid(datahb, latshb,lonshb, latsl, lonsl)
    assert (datal2 == datal).all(), "high to low res TEST FAILS: array changed"
    assert (datal2b == datal).all(), "high(b) to low res TEST FAILS: array changed"
    print("High to low res test passes, original grid returned OK")

    print("TODO: check subset regrid works")

    return True

if __name__=='__main__':
    print("Running tests in utilities.code_tests.py")
    check_regridding()