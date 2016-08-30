import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm # lognormal color bar
from matplotlib.ticker import FormatStrFormatter as fsf # formatter for labelling


import numpy as np
from datetime import datetime
from glob import glob
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline as RBS
import h5py

class gchcho:
    '''
    Class for holding the GCHCHO data
    Generally data arrays will be shaped as [(levels, )lats, lons]
    Units will be in molecules, metres, hPa
    '''
    date = 0
    # total columns     (molecules/m2)
    VC_HCHO = 0 #[91, 144]
    # density profiles  (molecules/m3)
    N_HCHO = 0
    N_Air = 0
    # Shape factors ( Palmer 01 )
    Shape_z = 0 #(1/m) [72, 91, 144]
    Shape_s = 0 # [72, 91, 144]
    # dimensions
    pmids = 0   # (hPa)
    pedges = 0  # (hPa)
    lons = 0    # [144]
    lats = 0    # [91]
    sigmas = 0  # [ 72, 91, 144 ]
    boxH = 0    # (m) [ 72, 91, 144 ]
    zmids = 0   # (m) [ 72, 91, 144 ]
    
    def __init__(self):
        self.date=datetime(2005,1,1)
    
    def ReadFile(self, date):
        fpath=glob('gchcho/hcho_%4d%02d.he5' % ( date.year, date.month ) )[0]
        with h5py.File(fpath, 'r') as in_f:
            dsetname='GC_UCX_HCHOColumns'
            dset=in_f[dsetname]
            self.date=date
            #VC(molecs/m2), N(molecs/m3), ShapeZ(1/m), ShapeSig(na), Pmids, Sigma, Boxheights...
            self.VC_HCHO    = dset['VCHCHO'].squeeze()      # molecs/m2
            self.N_HCHO     = dset['NHCHO'].squeeze()       # molecs/m3 profile
            self.N_Air      = dset['NAIR'].squeeze()        # molecs/m3 profile
            self.Shape_z    = dset['SHAPEZ'].squeeze()      # 1/m
            self.Shape_s    = dset['SHAPESIGMA'].squeeze()  # unitless
            self.pmids      = dset['PMIDS'].squeeze()       # geometric pressure mids (hPa)
            self.pedges     = dset['PEDGES'].squeeze()      # pressure edges hPa
            self.lons       = dset['LONGITUDE'].squeeze()   # longitude and latitude midpoints
            self.lats       = dset['LATITUDE'].squeeze()
            self.sigmas     = dset['SIGMA'].squeeze()       # Sigma dimension
            self.boxH       = dset['BOXHEIGHTS'].squeeze()  # box heights (m)
            zedges=np.cumsum(self.boxH, axis=0)     # height at top of each box
            zmids=zedges-self.boxH/2.0      # minus half the box height = box midpoint
            self.zmids      = zmids                         # altitude midpoints
    
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
    
    def get_latlon_inds(self, lat, lon):
        # match closest lat/lon
        latind=(np.abs(self.lats-lat)).argmin()
        lonind=(np.abs(self.lons-lon)).argmin()
        return(latind,lonind)
    
    def get_single_pmid(self, lat, lon):
        latind,lonind=get_latlon_inds(lat,lon)
        return(self.pmids[:,latind,lonind])
    
    def get_single_apriori(self, lat, lon, z=False):
        '''
        Return the apriori shape factor and sigmas closest to the inputted lat,lon
        z=True returns the S_z and zmids
        '''
        latind,lonind=get_latlon_inds(lat,lon)
        if z:
            shape=self.Shape_z[:,latind,lonind] # 1/m
            coord=self.zmids[:,latind,lonind]   # m
        else:
            shape=self.Shape_s[:,latind,lonind] # unitless
            coord=self.sigmas[:,latind,lonind]  # unitless
        
        return(shape, coord)
    
    def calculate_AMF(self, w, w_pmids, AMF_G, lat, lon, plotname=None):
        '''
        Return AMF calculated using normalized shape factors
        uses both S_z and S_sigma integrations
        
        Determine AMF_z using AMF_G * \int_0^{\infty} { w(z) S_z(z) dz }
        
        Determine AMF_sigma using AMF_G * \int_0^1 { w(s) S_s(s) ds }
        
        LEFT AND RIGHT NOW SET TO 0 (updated: 20160829)
        '''
        # column index, pressures, altitudes, sigmas
        #
        lati, loni=self.get_latlon_inds(lat,lon)
        S_pmids=self.pmids[:,lati,loni]     # Pressures (hPa)
        S_zmids=self.zmids[:,lati,loni]     # Altitudes (m)
        S_smids=self.sigmas[:,lati,loni]    # Sigmas
        h=self.boxH[:,lati,loni]            # box heights (m)
        S_pedges=self.pedges[:,lati,loni]   # Pressure edges(hPa)
        
        # calculate sigma edges
        S_sedges= (S_pedges - S_pedges[-1]) / (S_pedges[0]-S_pedges[-1])
        dsigma=S_sedges[0:-1]-S_sedges[1:]  # change in sigma at each level
        # The shape factors!
        S_z=self.Shape_z[:,lati,loni]
        S_s=self.Shape_s[:,lati,loni]
        
        # Default left,right values (now zero)
        lv,rv=0.,0.
        
        # convert w(press) to w(z) and w(s), on GEOS-Chem's grid
        # 
        w_zmids = np.interp(w_pmids, S_pmids[::-1], S_zmids[::-1])
        w_z     = np.interp(S_zmids, w_zmids, w,left=lv,right=rv)
        w_smids = (w_pmids - S_pedges[-1])/ (S_pedges[0]-S_pedges[-1])
        w_s     = np.interp(S_smids, w_smids[::-1], w[::-1],left=lv,right=rv)
        w_s_2   = np.interp(S_smids, w_smids[::-1], w[::-1]) # compare without fixed edges!
        # TODO: Maybe interpolate using logarithms?
        
        # Integrate w(z) * S_z(z) dz using summation
        # done over w(z) * S_z(z) * height(z)
        integral_z = np.sum(w_z * S_z * h)
        AMF_z = AMF_G * integral_z
        integral_s = np.sum(w_s * S_s * dsigma)
        AMF_s = AMF_G * integral_s
        
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
            h1, l1 = axes[0,0].get_legend_handles_labels()
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
            for yy,lbl in zip([0.7, 0.8, 0.9],['AMF$_z$=%5.2f'%AMF_z,'AMF$_{\sigma}$=%5.2f'%AMF_s,'AMF$_{\sigma}$(pre-fix)=%5.2f'%AMF_s_old]):
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
            plt.plot(S_s*w_s_2, S_smids, '--', label='product pre-fix', color='cyan')
            plt.legend(loc=0)
            plt.ylim([1.05,-0.05])
            plt.ylabel('$\sigma$')
            plt.xlabel('unitless')
            
            plt.suptitle('amf calculation factors')
            f.savefig(plotname)
            print('%s saved'%plotname)
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
        plt.title('GEOS-Chem $\Omega_{HCHO}$')
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
        ax1.set_title('HCHO at lat=%d, lon=%d '%(lat,lon))
        ax1.set_xlabel('HCHO molecules/m3')
        ax1.set_ylabel('hPa')
        
        # normalized shape factor 
        ax2.plot(self.Shape_s[:,yi,xi], z)
        ax2.set_title('apriori shape(sigma)')
        ax2.set_xlabel('S_s')
        ax2.set_ylabel('hPa')


