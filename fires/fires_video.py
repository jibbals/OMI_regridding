# -*- coding: utf-8 -*-
'''
Create an animation from the 8day average fire counts over Australia
'''

## Modules
# module for hdf eos 5
import h5py

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

## Files to be read:
#fname="Fires/MYD14C8H.2004001.h5"
files=glob.glob('Fires/*.h5')
files.sort() # sort alphabetically

## Fields to be read:
# Count of fires in each grid box over 8 days
corrFirePix = 'CorrFirePix'
cloudCorrFirePix = 'CloudCorrFirePix'

cfps=[]

for f in files:
    ## read in files:
    with h5py.File(f,'r') as in_f:
        ## get data arrays
        cfp     = in_f[corrFirePix].value
        #ccfp    = in_f[cloudCorrFirePix].value
    cfps.append(cfp)

# latitude = 89.75 - 0.5 * y
# longitude = -179.75 + 0.5 * x 
lats    = np.arange(90,-90,-0.5) - 0.25
lons    = np.arange(-180,180, 0.5) + 0.25
# mesh lon/lats
mlons,mlats = np.meshgrid(lons,lats)    

## Plotting
fig = plt.figure(figsize=(14,9))
# Australia only (and a little of indonesia)
m=Basemap(llcrnrlat=-45,  urcrnrlat=0,
          llcrnrlon=105, urcrnrlon=155,
          resolution='c',projection='merc')

clevs = np.arange(0,30,2)
cs=m.contourf(mlons, mlats, cfps[0], clevs, latlon=True, vmin=1, cmap=plt.cm.jet, extend='both')
# set oob_low to white
cs.cmap.set_under('w')

# draw coastlines and equator
m.drawcoastlines()
m.drawparallels([0],labels=[0,0,0,0])

# add title, colorbar
cb=m.colorbar(cs,"right",size="5%", pad="2%")
cb.set_label('Active Fires over 8 days')
txt=plt.title('0: '+files[0])

# animate 
def updatefig(i):
    global cs
    for c in cs.collections: c.remove()
    cs = m.contourf(mlons,mlats,cfps[i],clevs,latlon=True,vmin=1,cmap=plt.cm.jet,extend='both')
    cs.cmap.set_under('w')
    txt.set_text(str(i)+': '+files[i])
    
ani = animation.FuncAnimation(fig, updatefig, frames=len(files))

ani.save('FiresMovie.mp4')
plt.show()