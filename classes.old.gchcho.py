#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ??

    Class to hold GCHCHO data

@author: jesse
"""
###############
### MODULES ###
###############
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm # lognormal color bar
from matplotlib.ticker import FormatStrFormatter as fsf # formatter for labelling

import numpy as np
#from datetime import datetime
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline as RBS

# Add parent folder to path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,os.path.dirname(currentdir))

from utilities import fio
sys.path.pop(0)

###############
### GLOBALS ###
###############
__VERBOSE__=True

###############
### CLASS   ###
###############

# rename some variables to more coding friendly ones:
_names={ key:str.lower(key) for key in fio.__GCHCHO_KEYS__ }
_names['VCHCHO']='VC_HCHO'
_names['LATITUDE']='lats'
_names['LONGITUDE']='lons'
_names['NAIR']='N_Air'
_names['NHCHO']='N_HCHO'
_names['SHAPEZ']='Shape_z'
_names['SHAPESIGMA']='Shape_s'
_names['BOXHEIGHTS']='boxH'
_names['SIGMA']='sigmas'

class gchcho:
    '''
    Class for holding the GCHCHO data
    Generally data arrays will be shaped as [(levels, )lats, lons]
    Units will be in molecules, metres, hPa
    THESE FILES ARE THE 1300-1400 averaged satellite outputs from GEOS-Chem
    '''
    date = 0

    # total columns     (molecules/m2)
    #VC_HCHO
    # density profiles  (molecules/m3)
    #N_HCHO
    #N_Air

    # Shape factors ( Palmer 01 )
    # Shape_z  # (1/m) [72, 91, 144]
    # Shape_s  # [72, 91, 144]

    # dimensions
    # pmids    # (hPa)
    # pedges   # (hPa)
    # lons     # [144]
    # lats     # [91]
    # sigmas   # [ 72, 91, 144 ]
    # boxH     # (m) [ 72, 91, 144 ]
    # zmids    # (m) [ 72, 91, 144 ]

    def __init__(self,date):
        #self.date=datetime(2005,1,1)
        fdata=fio.read_gchcho(date)
        for key in fio.__GCHCHO_KEYS__:
            if __VERBOSE__:
                print("Reading key %s (%s)"%(key,_names[key]))
            setattr(self,_names[key],fdata[key])

        # Extras:
        self.date=date

        ztops = np.cumsum(self.boxH, axis=0) # height at top of each box
        zmids = ztops-self.boxH/2.0   # minus half the box height = box midpoint
        self.zmids =zmids                         # altitude midpoints
        assert np.all(zmids > 0), "zmids calculation error: %s"%str(zmids)


    def get_apriori(self, latres=0.25, lonres=0.3125):
        '''
        Read GC HCHO sigma shape factor and regrid to lat/lon res.
        temporal resolution is one month
        inputs:
            latres, lonres for resolution of GC 2x2.5 hcho columns to be regridded onto
        '''
        assert False, "Method is old and wrong currently"
        # new latitude longitude we interpolate to.
        newlats= np.arange(-90,90, latres) + latres/2.0
        newlons= np.arange(-180,180, lonres) + lonres/2.0

        # Mesh[lat,lon]
        mlons,mlats = np.meshgrid(self.lons,self.lats)
        mnewlons,mnewlats = np.meshgrid(newlons,newlats)

        ## Get sigma apriori and regrid it
        #
        newS_s = np.zeros([72,len(newlats),len(newlons)])
        newSigma = np.zeros([72,len(newlats),len(newlons)])

        # interpolate at each pressure level...
        for ii in range(72):
            newS_s[ii,:,:] = griddata( (mlats.ravel(), mlons.ravel()),
                                       self.Shape_s[ii,:,:].ravel(),
                                       (mnewlats, mnewlons),
                                       method='nearest')
            newSigma[ii,:,:]=griddata( (mlats.ravel(), mlons.ravel()),
                                     self.sigmas[ii,:,:].ravel(),
                                     (mnewlats, mnewlons),
                                     method='nearest')

        # return the normalised sigma apriori used to recalculate AMF
        return newS_s, newlats, newlons, newSigma

    def get_latlon_inds(self, lat, lon):
        # match closest lat/lon
        # ignore warning when comparing array with nans:
        with np.errstate(invalid='ignore'):
            latind=(np.abs(self.lats-lat)).argmin()
            lonind=(np.abs(self.lons-lon)).argmin()
        return(latind,lonind)

    def get_single_pmid(self, lat, lon):
        latind,lonind=self.get_latlon_inds(lat,lon)
        return(self.pmids[:,latind,lonind])

    def get_single_apriori(self, lat, lon, z=False):
        '''
        Return the apriori shape factor and sigmas closest to the inputted lat,lon
        z=True returns the S_z and zmids
        '''
        latind,lonind=self.get_latlon_inds(lat,lon)
        if z:
            shape=self.Shape_z[:,latind,lonind] # 1/m
            coord=self.zmids[:,latind,lonind]   # m
        else:
            shape=self.Shape_s[:,latind,lonind] # unitless
            coord=self.sigmas[:,latind,lonind]  # unitless

        return(shape, coord)

    def calculate_AMF(self, w, w_pmids, AMF_G, lat, lon, plotname=None, debug_levels=False):
        '''
        Return AMF calculated using normalized shape factors
        uses both S_z and S_sigma integrations

        Determines
            AMF_z = \int_0^{\infty} { w(z) S_z(z) dz }
            AMF_s = \int_0^1 { w(s) S_s(s) ds } # this one uses sigma dimension
            AMF_Chris = \Sigma_i (Shape(P_i) * \omega(P_i) * \Delta P_i) /  \Sigma_i (Shape(P_i) * \omega(P_i) )

        update:20161109
            Chris AMF calculated, along with normal calculation without AMF_G
            lowest level fix now moved to a function to make this method more readable
        update:20160905
            Fix lowest level of both GC box and OMI pixel to match the pressure
            of the least low of the two levels, and remove any level which is
            entirely below the other column's lowest level.
        update:20160829
            LEFT AND RIGHT NOW SET TO 0 ()
        OLD:
            Determine AMF_z using AMF_G * \int_0^{\infty} { w(z) S_z(z) dz }
            Determine AMF_sigma using AMF_G * \int_0^1 { w(s) S_s(s) ds }

        '''
        # column index, pressures, altitudes, sigmas:
        #
        lati, loni=self.get_latlon_inds(lat,lon)
        S_pmids=self.pmids[:,lati,loni].copy()  # Pressures (hPa)

        S_zmids=self.zmids[:,lati,loni]         # Altitudes (m)
        S_smids=self.sigmas[:,lati,loni]        # Sigmas
        h=self.boxH[:,lati,loni]                # box heights (m)
        S_pedges=self.pedges[:,lati,loni].copy()# Pressure edges(hPa)

        # The shape factors!
        S_z=self.Shape_z[:,lati,loni].copy()
        S_s=self.Shape_s[:,lati,loni].copy()

        # Interpolate the shape factors to these new pressure levels:
        # S_s=np.interp(S_pmids, S_pmids_init[::-1], S_s[::-1])
        # also do S_z? currently I'm leaving this for comparison. AMF_z will be sanity check

        # calculate sigma edges
        S_sedges = (S_pedges - S_pedges[-1]) / (S_pedges[0]-S_pedges[-1])
        dsigma = S_sedges[0:-1]-S_sedges[1:]  # change in sigma at each level

        # Default left,right values (now zero)
        lv,rv=0.,0.

        # sigma midpoints for interpolation
        w_smids = (w_pmids - S_pedges[-1])/ (S_pedges[0]-S_pedges[-1])

        # convert w(press) to w(z) and w(s), on GEOS-Chem's grid
        #
        w_zmids = np.interp(w_pmids, S_pmids[::-1], S_zmids[::-1])
        w_z     = np.interp(S_zmids, w_zmids, w,left=lv,right=rv) # w_z does not account for differences between bottom levels of GC vs OMI pixel
        w_s     = np.interp(S_smids, w_smids[::-1], w[::-1],left=lv,right=rv)
        w_s_2   = np.interp(S_smids, w_smids[::-1], w[::-1]) # compare without fixed edges!

        # Integrate w(z) * S_z(z) dz using sum(w(z) * S_z(z) * height(z))
        AMF_z = np.sum(w_z * S_z * h)
        AMF_s= np.sum(w_s * S_s * dsigma)

        # Calculations with bottom relevelled
        # match he bottom levels in the pressure midpoints dimension
        w_pmids_new,S_pmids_new,w_new,S_s_new = match_bottom_levels(w_pmids,S_pmids,w,S_s)
        S_pedges_new = S_pedges.copy()
        for i in range(1,len(S_pedges_new)-1):
            S_pedges_new[i]=(S_pmids_new[i-1]*S_pmids_new[i]) ** 0.5
        S_sedges_new = (S_pedges_new - S_pedges_new[-1]) / (S_pedges_new[0]-S_pedges_new[-1])
        dsigma_new = S_sedges_new[0:-1]-S_sedges_new[1:]
        w_smids_new = (w_pmids_new - S_pedges_new[-1])/ (S_pedges_new[0]-S_pedges_new[-1])
        S_smids_new=(S_pmids_new - S_pedges_new[-1]) / (S_pedges_new[0]-S_pedges_new[-1])
        w_s_new     = np.interp(S_smids_new, w_smids_new[::-1], w_new[::-1], left=lv, right=rv)
        AMF_s_new = np.sum(w_s_new * S_s_new * dsigma_new)

        if plotname is not None:
            integral_s_old=np.sum(w_s_2 * S_s * dsigma)
            AMF_s_old= AMF_G * integral_s_old

            f,axes=plt.subplots(2,2,figsize=(14,12))
            # omega vs pressure, interpolated and not interpolated
            plt.sca(axes[0,0])
            plt.plot(w, w_pmids, linestyle='-',marker='o', linewidth=2, label='original',color='k')
            plt.title('$\omega$')
            plt.ylabel('p(hPa)'); plt.ylim([1015, 0.01]); plt.yscale('log')
            ax=plt.twinx(); ax.set_ylabel('z(m)'); plt.sca(ax)
            plt.plot(w_z, S_zmids, linestyle='-',marker='x', label='$\omega$(z)',color='fuchsia')
            # add legend from both axes
            h1,l1 = axes[0,0].get_legend_handles_labels()
            h2,l2 = ax.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2,loc=0)

            # omega vs sigma levels
            plt.sca(axes[0,1])
            plt.plot(w_s, S_smids, '.')
            plt.title('$\omega(\sigma)$')
            plt.ylabel('$\sigma$')
            plt.yscale('log')
            plt.ylim(1.01, 0.001)
            axes[0,1].yaxis.tick_right()
            axes[0,1].yaxis.set_label_position("right")
            # add AMF value to plot
            for yy,lbl in zip([0.6, 0.7, 0.8, 0.9], ['AMF$_z$=%5.2f'%AMF_z, 'AMF$_{\sigma}$=%5.2f'%AMF_s, 'AMF$_{\sigma}$(pre-fix)=%5.2f'%AMF_s_old, 'AMF$_{\sigma relevelled}$=%5.2f'%AMF_s_new]):
                plt.text(.1,yy,lbl,transform=axes[0,1].transAxes,fontsize=16)

            # shape factor z plots
            plt.sca(axes[1,0])
            plt.plot(S_z, S_pmids, '.',label='S$_z$(p)', color='k')
            plt.ylim([1015,0.01])
            plt.title('Shape')
            plt.ylabel('p(hPa)')
            plt.yscale('log')
            plt.xlabel('m$^{-1}$')
            # overplot omega*shape factor on second y axis
            ax=plt.twinx(); ax.set_ylabel('z(m)'); plt.sca(ax)
            plt.plot(S_z*w_z, S_zmids,label='$S_z(z) * \omega(z)$',color='fuchsia')
            # legend
            h1,l1 = axes[1,0].get_legend_handles_labels()
            h2,l2 = ax.get_legend_handles_labels()
            ax.legend(h1+h2,l1+l2,loc=0)
            axes[1,0].xaxis.set_major_formatter(fsf('%2.1e'))

            # sigma shape factor plots
            plt.sca(axes[1,1]);
            plt.title('Shape$_\sigma$')
            plt.plot(S_s, S_smids, label='S$_\sigma$', color='k')
            plt.plot(S_s*w_s, S_smids, label='S$_\sigma * \omega_\sigma$', color='fuchsia')
            plt.plot(S_s_new*w_s_new, S_smids_new, label='new S$_\sigma * \omega_\sigma$', color='orange')
            plt.plot(S_s*w_s_2, S_smids, '--', label='old product', color='cyan')
            plt.legend(loc=0)
            plt.ylim([1.05,-0.05])
            plt.ylabel('$\sigma$')
            plt.xlabel('unitless')

            plt.suptitle('amf calculation factors')
            f.savefig(plotname)
            print('%s saved'%plotname)
            plt.close(f)
        return (AMF_s, AMF_z)

    def interp_to_grid(self, newlats, newlons):
        '''
        Return interpolated HCHO columns
        Inputs:
            newlats: ascending regular grid of latitudes
            newlons: '' of longitudes
        Outputs: VC_HCHO in molecs/m2
            VC_HCHO[ newlats, newlons ], newlats, newlons
        '''
        # X, Y, DATA[X,Y]
        interp=RBS(self.lons, self.lats, np.transpose(self.VC_HCHO))
        newhcho=np.transpose(interp(newlons,newlats))
        return newhcho

    def PlotVC(self, lllat=-65, urlat=65, lllon=-179, urlon=179, vmin=1e17, vmax=1e21, cm2=False):
        '''
        Basemap(...).pcolor of vertical column HCHO
        returns (map, image, colorbar)
        '''
        m=Basemap(llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon, resolution='i',projection='merc')
        mlons, mlats= np.meshgrid(self.lons,self.lats)
        VC=self.VC_HCHO
        label='Molecules m$^{-2}$'
        if cm2:
            VC = VC*1e-4
            label='Molecules cm$^{-2}$'
        cs=m.pcolor(mlons,mlats,VC,latlon=True, vmin=vmin,vmax=vmax,norm = LogNorm())

        # draw coastlines and equator
        m.drawcoastlines()
        m.drawparallels([0],labels=[0,0,0,0])

        # add title, colorbar
        plt.title('GEOS-Chem $\Omega_{HCHO}$ '+self.date.strftime('%Y%m%d'))
        cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
        cb.set_label(label)
        return (m,cs,cb)

    def PlotProfile(self, lat=-30,lon=140):
        '''
        Plot a profile
        '''

        # latitude longitude index
        xi=np.searchsorted(self.lons,130)
        yi=np.searchsorted(self.lats,-30)
        z = self.pmids[:,yi,xi]
        yrange=[1e3, 1e-1]

        # number density profile
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.N_HCHO[:,yi,xi], z)
        # set to log scales
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        # set x, y ranges
        ax1.set_ylim(yrange)
        ax1.set_xlim([1e11, 1e17])
        ax1.set_title('HCHO at lat=%d, lon=%d on %s'%(lat,lon,self.date.strftime('%Y%m%d')))
        ax1.set_xlabel('HCHO molecules/m3')
        ax1.set_ylabel('hPa')

        # normalized shape factor
        ax2.plot(self.Shape_s[:,yi,xi], z)
        ax2.set_title('apriori shape(sigma)')
        ax2.set_xlabel('S_s')
        ax2.set_ylabel('hPa')

def match_bottom_levels(p1i, p2i, arr1i, arr2i):
    '''
    Takes two arrays of pressure (from surface to some altitude)
    Returns the same two arrays with the lowest levels set to the same pressure
    This method raises the lowest levels of the pressure arrays to match each other.
    Also interpolates arr1, arr2 to the new pressures if their pedges have changed
    '''
    # update:20160905, 20161109
    # one array will have a lower bottom level than the other
    #
    p1,p2=np.array(p1i.copy()),np.array(p2i.copy())
    arr1,arr2=np.array(arr1i.copy()),np.array(arr2i.copy())
    if p1[0] > p2[0]:
        plow=p1
        phigh=p2
        alow=arr1
    elif p1[0] == p2[0]:
        return p1,p2,arr1,arr2
    else:
        plow=p2
        phigh=p1
        alow=arr2
    plow_orig=plow.copy()

    # now lower has a lower surface altitude(higher surface pressure)
    movers=np.where(plow+0.05*plow[0] > phigh[0])[0] # index of pressure edges below higher surface pressure
    above_ind = movers[-1]+1
    if above_ind >= len(plow):
        print("Whole plow array below surface of phigh array")
        print("plow:",plow.shape)
        print(plow)
        print("phigh:",phigh.shape)
        print(phigh)
        assert False, "Fix this please"


    above = plow[above_ind] # pressure edge above the ones we need to move upwards
    rmovers=movers[::-1] # highest to lowest list of pmids to relevel
    # for all but the lowest pmid, increase above lowest pmid in other pressure array
    for ii in rmovers[0:-1]:
        plow[ii]=(phigh[0]*above)**0.5
        above=plow[ii]
    # for the lowest pmid, raise to match other array's lowest pmid
    plow[0]=phigh[0]

    # now interpolate the changed array ( reversed as interp wants increasing array xp)
    alow[:]=np.interp(plow,plow_orig[::-1],alow[::-1])

    return p1,p2,arr1,arr2
