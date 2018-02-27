# import the functions in other python script
import regrid_swaths as rs

# dates and maths
from datetime import datetime
import numpy as np

# for plotting
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar

d0=datetime(2005,1,1)
latres=2.0
lonres=2.0

#rs.make_gridded_swaths(d0,latres=latres,lonres=lonres)

data,attr=rs.read_regridded_swath(d0)

print(data.keys()) # shows what keys are in our data structure

print(data['VC_C'].shape) # shape of VC_C array

print(attr['VC_C']) # attributes for VC_C

HCHO=data['VC_C']
fire=data['fires']
lats=data['lats']
lons=data['lons']

###
### Plot map example:
### You can ask me anything about this stuff - if you get stuck
###

# S W N E
__AUSREGION__=[-45, 108.75, -7, 156.25]
__GLOBALREGION__=[-69, -178.75, 69, 178.75]
def plot_map(data,lats,lons, linear=True,
             region=__GLOBALREGION__,vmin=None,vmax=None):
    '''
    '''

    m=Basemap(llcrnrlat=region[0], urcrnrlat=region[2], llcrnrlon=region[1], urcrnrlon=region[3],
                  resolution='i', projection='merc')

    # Make into 2D meshed grid
    mlons,mlats=np.meshgrid(lons,lats)

    if vmin is None:
        vmin=0.1
    if vmax is None:
        vmax=np.nanmax(data)

    # can set many arguments for the map:
    cmap='rainbow'
    norm=None
    if not linear:
        norm=LogNorm()
    cs=m.pcolormesh(mlons, mlats, HCHO, latlon=True,
                    norm=norm,
                    vmin=vmin, vmax=vmax, clim=(vmin, vmax),
                    cmap=cmap)

    # draw coastlines
    m.drawcoastlines()

    return m,cs


# Create a figure:
plt.figure(figsize=(12,6))

# Plot columns:
plt.subplot(121) # one row two columns, first subplot
m,cs = plot_map(HCHO,lats,lons,linear=False,vmin=1e14,vmax=1e16)
plt.title('OMI HCHO Columns')

# add colour bar
cb=m.colorbar(cs, 'bottom', size='5%', pad='1%', extend='both')
cb.set_label('molecules/cm2')

# plot fires:
plt.subplot(122) # one row two columns, second subplot
m,cs = plot_map(fire,lats,lons, linear=False,vmin=1,vmax=1e8)
plt.title('MOD14A1 fires')

cb=m.colorbar(cs, 'bottom', size='5%', pad='1%', extend='both')
cb.set_label('fire pixels/day')

# save the figure:
plt.suptitle(d0.strftime('Plots for %Y %m %d'))
plt.savefig('test_plot.png')
plt.close()