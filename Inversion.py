#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 09:48:11 2016

    Functions to creating the isoprene emissions estimates from my HCHO product

    Invert formaldehyde to Isoprene emissions using $\Omega_{HCHO} = S \times E_{ISOP} + B$
    Y_isop is the RMA regression between $k_{HCHO}*\Omega_{HCHO}$, and $k_{isop}*\Omega_{isop}$
    and $S = Y_{isop} / k_{HCHO}$
    B will be calculated from the remote pacific near the same latitude.

@author: jesse
"""

import numpy as np
import fio
from JesseRegression import RMA

def Background(H):
    '''
        Determine background HCHO as a function of latitude and time, based on the average over the remote pacific ocean
        Remote pacific = lon0 to lon1, lat+-5 degrees (TODO:)

        Assume H[lat,lon,time]
        Return B[lat,time]
    '''

def Yield(H, k_H, I, k_I):
    '''
        H=HCHO, k_H=loss, I=Isoprene, k_I=loss
        The returned yield will match the first two dimensions.

        Y_isop is the RMA regression between $k_{HCHO}*\Omega_{HCHO}$, and $k_{isop}*\Omega_{isop}$

        I think this should be run on each 8 days of gridded model output to get the Yield for that 8 days

    '''
    # first handle the lats and lons:
    n0=np.shape(H)[0]
    n1=np.shape(H)[1]

    #Isoprene Yield, spatially varying, not temporally varying:
    #
    Y_I=np.zeros([n0,n1])
    A=k_H*H
    B=k_I*I
    for i in range(n0):
        for j in range(n1):
            m,b,r,ci1,ci2=RMA(A[i,j,:], B[i,j,:])
            Y_I[i,j] = m

    #Return the yields
    return Y_I

def Emissions(month=datetime(2005,1,1)):
    '''
        Determine emissions of isoprene for a particular month
        1) Calculate Yield and background for a particular month
        2) Use Biogenic HCHO product to calculate E_isoprene.
    '''
    # Read model data for this month
    #

    # Read biogenic HCHO product
    #

    # Calculate Background and Yield from model data
    #

    # Calculate Emissions from these
    #

