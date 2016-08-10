# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:09:48 2016

Regrid from 0.5x0.5 degree resolution to 0.25x0.3125 and check results

@author: jesse
"""

## Modules
import fires_fio as ffio

import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import datetime


def ausplot(mlats,mlons,data,vmin=1,vmax=10):
    ''' function to plot australia with basemap'''
    # Australia only ( and a little of indonesia)
    m=Basemap(llcrnrlat=-45, urcrnrlat=0, llcrnrlon=105, urcrnrlon=155,
              resolution='i',projection='merc')
    
    # show 1 to 10 fires, don't show 0
    cs=m.pcolor(mlons, mlats, data, latlon=True, vmin=1, vmax=10, cmap=plt.cm.jet)
    cs.cmap.set_under('w')

    m.drawcoastlines()
    return m, cs


## day of file to be read:
day=datetime(2005,1,1)

## get normal and interpolated fire
orig, lats, lons=ffio.read_8dayfire(day)

lonres,latres = 0.3125, 0.25
regr, lats2, lons2=ffio.read_8dayfire_interpolated(day, latres=latres, lonres=lonres)

# Number checks..
assert np.max(orig) == np.max(regr), "Maximum values don't match..."
print ("Mean orig = %4.2f\nMean regridded = %4.2f"%(np.mean(orig[orig>-1]), np.mean(regr[regr>-1])))

## PLOTTING
##

# EG plot of grids...
fig0 = plt.figure(0,figsize=(10,8))
# tasmania lat lon = -42, 153
m0=Basemap(llcrnrlat=-45, urcrnrlat=-37, llcrnrlon=142, urcrnrlon=150,
          resolution='i',projection='merc')
m0.drawcoastlines()
m0.fillcontinents(color='coral',lake_color='aqua')
m0.drawmapboundary(fill_color='white')

# original grid = thick black
d=[1000,1]
m0.drawparallels(lats-0.25, linewidth=2.5, dashes=d)
m0.drawmeridians(lons-0.25, linewidth=2.5, dashes=d)
# new grid = thin brown
m0.drawparallels(lats2-latres/2.0, color='brown', dashes=d)
m0.drawmeridians(lons2-lonres/2.0, color='brown', dashes=d)
plt.title('Original grid(black) vs new grid(red)')

## Regridded Data plot comparison
# plot on two subplots
fig = plt.figure(1,figsize=(14,9))
fig.patch.set_facecolor('white')

# mesh lon/lats
mlons,mlats = np.meshgrid(lons,lats)    
mlons2,mlats2 = np.meshgrid(lons2,lats2)
inputs=[(mlats,mlons,orig),
        (mlats2,mlons2,regr)]
axtitles=['original (0.5x0.5)',
          'regridded to %1.4fx%1.4f (latxlon)' % (latres,lonres)]
for i in range(2):
    ax=plt.subplot(121+i)
    m,cs = ausplot(*inputs[i])
    ax.set_title(axtitles[i])
    # colorbar stuff
    cb = m.colorbar(cs,'bottom')
    cb.set_label('Active Fires over 8 days')

plt.suptitle('AQUA 2005001',fontsize=20)
plt.tight_layout()
plt.show()
