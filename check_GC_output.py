# -*- coding: utf-8 -*-
'''
# Python script created by jesse greenslad

Check the ncfiles created by bpch2coards
Run from main project directory or else imports will not work
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# module for hdf eos 5
#import h5py 
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import matplotlib.pyplot as plt

# local imports:
import plotting as pp
from Data.GC_fio import get_tropchem_data, get_UCX_data

##################
#####GLOBALS######
##################

################
###FUNCTIONS####
################

def compare_GC_plots(date=datetime(2005,1,1)):
    ''' maps of UCX and tropchem surface HCHO'''
    ausregion=pp.__AUSREGION__ # [S W N E]
    dstr=date.strftime("%Y%m%d")
    
    # First get tropchem data:
    #
    tdat=get_tropchem_data(date=date,monthavg=True,surface=True)
    thcho=tdat['IJ-AVG-$CH2O']
    tlat=tdat['latitude']
    tlon=tdat['longitude']
    # determine min and max:
    tvmin,tvmax = np.nanmin(thcho), np.nanmax(thcho)
    print("Global tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))
    tvmin,tvmax = pp.findrange(thcho,tlat,tlon, ausregion)
    print("Aus tropchem min=%.2e, max=%.2e"%(tvmin,tvmax))
    
    # Then get UCX data:
    #
    udat=get_UCX_data(date=date,surface=True)
    uhcho=udat['IJ_AVG_S__CH2O']
    ulat=udat['lat']
    ulon=udat['lon']
    assert (np.array_equal(ulat,tlat)) and (np.array_equal(ulon,tlon)), "LATS AND LONS DIFFER"
    
    # determine min and max:
    uvmin,uvmax = np.nanmin(uhcho), np.nanmax(uhcho)
    print("Global UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    uvmin,uvmax = pp.findrange(uhcho,ulat,ulon, ausregion)
    print("Aus UCX min=%.2e, max=%.2e"%(uvmin,uvmax))
    vmin,vmax=np.min([uvmin,tvmin]),np.max([uvmax,tvmax])
    
    # Figures with 4 subplots
    f,axes=plt.subplots(2,2,figsize=(14,14))
    kwargs={'vmin':vmin,'vmax':vmax,'linear':True}
    # first is tropchem
    plt.sca(axes[0,0]) 
    m,cs,cb=pp.ausmap(thcho,tlat,tlon, **kwargs)
    plt.title('tropchem surface')
    cb.set_label('ppbv')
    
    # second is UCX
    plt.sca(axes[0,1])
    m,cs,cb=pp.ausmap(uhcho,ulat,ulon, **kwargs)
    plt.title('UCX surface')
    cb.set_label('ppbv')
    
    # Third is diffs:
    plt.sca(axes[1,0])
    m,cs,cb = pp.ausmap(uhcho-thcho, tlat, tlon, **kwargs)
    plt.title('UCX - tropchem')
    cb.set_label('ppbv')

    # Fourth is rel diff:
    plt.sca(axes[1,1])
    kwargs['vmin']=-10; kwargs['vmax']=10
    m,cs,cb = pp.ausmap((uhcho-thcho)/thcho*100, tlat, tlon, **kwargs)
    plt.title('100*(UCX - tropchem)/tropchem')
    cb.set_label('% difference')
    
    
    pname='Figs/GC/tropchem_hcho_%s.png'%dstr
    plt.suptitle('HCHO %s'%dstr)
    plt.savefig(pname)
    print("%s saved"%pname)

# If this script is run directly:
if __name__=='__main__':
    pp.InitMatplotlib()
    compare_GC_plots()


