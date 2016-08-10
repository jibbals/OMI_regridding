# Python script reading hdf5 junk

## Modules
# module for hdf eos 5
import h5py

import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar

## File to be read:
fname="Fires/MYD14C8H.2005001.h5"


## Fields to be read:
# Count of fires in each grid box over 8 days
corrFirePix = 'CorrFirePix'
cloudCorrFirePix = 'CloudCorrFirePix'

## read in file:
with h5py.File(fname,'r') as in_f:
    ## get data arrays
    cfp     = in_f[corrFirePix].value
    ccfp    = in_f[cloudCorrFirePix].value

# from document at http://www.fao.org/fileadmin/templates/gfims/docs/MODIS_Fire_Users_Guide_2.4.pdf
# latitude = 89.75 - 0.5 * y
# longitude = -179.75 + 0.5 * x 
lats    = np.arange(90,-90,-0.5) - 0.25
lons    = np.arange(-180,180, 0.5) + 0.25

## Plotting

# Australia only ( and a little of indonesia)
m=Basemap(llcrnrlat=-45,  urcrnrlat=0,
          llcrnrlon=105, urcrnrlon=155,
          resolution='c',projection='merc')

# mesh lon/lats
mlons,mlats = np.meshgrid(lons,lats)    

#clevs = np.arange(0,30,2)
cs=m.pcolor(mlons, mlats, cfp, latlon=True, vmin=1, vmax=10, cmap=plt.cm.jet)
cs.cmap.set_under('w')

# 
m.drawcoastlines()
m.drawparallels([0],labels=[0,0,0,0])
    
#add title, colorbar
cb=m.colorbar(cs,"right",size="5%", pad="2%")
cb.set_label('Active Fires over 8 days')
plt.title('AQUA 2005001 ')


plt.show()
