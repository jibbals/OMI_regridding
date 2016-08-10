import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm # lognormal color bar

import numpy as np
from datetime import datetime
from glob import glob
from scipy.interpolate import griddata
import h5py

class gchcho:
    '''
    Class for holding the GCHCHO data
    Generally data arrays will be shaped as [(levels, )lats, lons]
    Units will be in molecules, metres, hPa
    '''
    date=0
    # total columns     (molecules/m2)
    VC_HCHO=0   #[91, 144]
    # density profiles  (molecules/m3)
    N_HCHO =0
    N_Air =0
    # Shape factors ( Palmer 01 )
    Shape_z=0   #(1/m) [72, 91, 144]
    Shape_s=0   # [72, 91, 144]
    # dimensions
    pmids=0     #(hPa)
    pedges=0    #(hPa)
    lons=0
    lats=0
    sigmas=0
    boxH=0      #(m)
    
    def __init__(self):
        self.date=datetime(2005,1,1)
    
    def ReadFile(self, date):
        fpath=glob('gchcho/hcho_%4d%02d.he5' % ( date.year, date.month ) )[0]
        with h5py.File(fpath, 'r') as in_f:
            dsetname='GC_UCX_HCHOColumns'
            dset=in_f[dsetname]
            self.date=date
            #VC(molecs/m2), N(molecs/m3), ShapeZ(1/m), ShapeSig(na), Pmids, Sigma, Boxheights...
            self.VC_HCHO    = dset['VCHCHO'].squeeze() # molecs/m2
            self.N_HCHO     = dset['NHCHO'].squeeze() # molecs/m3 profile
            self.N_Air      = dset['NAIR'].squeeze() # molecs/m3 profile
            self.Shape_z    = dset['SHAPEZ'].squeeze() # 1/m
            self.Shape_s    = dset['SHAPESIGMA'].squeeze() # unitless
            self.pmids      = dset['PMIDS'].squeeze() # geometric pressure mids (hPa)
            self.pedges     = dset['PEDGES'].squeeze() # pressure edges hPa
            self.lons       = dset['LONGITUDE'].squeeze() # longitude and latitude midpoints
            self.lats       = dset['LATITUDE'].squeeze()
            self.sigmas     = dset['SIGMA'].squeeze() # Sigma dimension
            self.boxH       = dset['BOXHEIGHTS'].squeeze() # box heights (m)
    
    def get_apriori(self, latres=0.25, lonres=0.3125):
        '''
        Read GC HCHO sigma shape factor and regrid to lat/lon res. 
        temporal resolution is one month
        inputs:
            latres, lonres for resolution of GC 2x2.5 hcho columns to be regridded onto
        '''
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
    
    def get_single_apriori(self, lat, lon):
        '''
        Return the apriori shape factor and sigmas closest to the inputted lat,lon
        '''
        # match closest lat/lon
        latind=(np.abs(self.lats-lat)).argmin()
        lonind=(np.abs(self.lons-lon)).argmin()
        # Shape_s is unitless
        shape=self.Shape_s[:,latind,lonind]
        sigma=self.sigmas[:,latind,lonind]
        return(shape,sigma)
    
    def PlotVC(self, lllat=-65, urlat=65, lllon=-179, urlon=179, vmin=1e17, vmax=1e21):
        '''
        Basemap(...).pcolor of vertical column HCHO
        returns (map, image, colorbar)
        '''
        m=Basemap(llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon, resolution='i',projection='merc')
        mlons, mlats= np.meshgrid(self.lons,self.lats)
        
        cs=m.pcolor(mlons,mlats,self.VC_HCHO,latlon=True, vmin=vmin,vmax=vmax,norm = LogNorm())
        # draw coastlines and equator
        m.drawcoastlines()
        m.drawparallels([0],labels=[0,0,0,0])
        # add title, colorbar
        plt.title('GEOS-Chem VC_HCHO (2x2.5)')
        cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
        cb.set_label('Molecules/m2')
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
        ax1.set_title('HCHO at lat=%d, lon=%d '%(lat,lon))
        ax1.set_xlabel('HCHO molecules/m3')
        ax1.set_ylabel('hPa')
        
        # normalized shape factor 
        ax2.plot(self.Shape_s[:,yi,xi], z)
        ax2.set_title('apriori shape(sigma)')
        ax2.set_xlabel('S_s')
        ax2.set_ylabel('hPa')


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
