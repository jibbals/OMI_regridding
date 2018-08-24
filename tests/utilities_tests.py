import numpy as np
from classes import omhchorp
from utilities import utilities as util
from utilities import plotting as pp
from utilities import GMAO
import matplotlib.pyplot as plt
from datetime import datetime

def test_adjacent_masking():
    z=np.zeros([5,6])
    z[0,4]=1
    z[2,2]=1
    z[4,5]=1
    za = util.set_adjacent_to_true(z).astype(int)
    print(z)
    print(za)

    assert za[0,3]+za[0,4]+za[0,5]+za[1,3]+za[1,4]+za[1,5] == 6, "top edge doesn't work as expected"
    assert za[1,1]+za[1,2]+za[1,3]+za[2,1]+za[2,3]+za[2,3]+za[3,1]+za[3,2]+za[3,3] == 9, "middle doesn't work as expected"
    assert za[3,4]+za[3,5]+za[4,4]+za[4,5] == 4, "bottom corner doesn't work as expected"
    assert np.sum(za) == 19-1, "Should be 18 filtered squares" # one overlaps


def check_resolution_binning(d0=datetime(2005,1,1),):
    '''
        See what happens when regridding to lower resolution using pixel counts
    '''
    region=pp.__AUSREGION__

    OMHCHORP=omhchorp(day0=d0, ignorePP=False)
    arr_names=['VCC_OMI','gridentries','ppentries','col_uncertainty_OMI','firemask','smokemask','anthromask']
    arrs= [getattr(OMHCHORP,s) for s in arr_names]
    arrs_i= {arr_names[i]:i for i in range(len(arr_names))}

    OMHsubsets=util.lat_lon_subset(OMHCHORP.lats,OMHCHORP.lons,region,data=arrs, has_time_dim=False)
    omilats=OMHsubsets['lats']
    omilons=OMHsubsets['lons']
    omilati=OMHsubsets['lati']
    # map subsetted arrays into another dictionary
    OMHsub = {s:OMHsubsets['data'][arrs_i[s]] for s in arr_names}


    VCC_OMI               = OMHsub['VCC_OMI']
    pixels                = OMHsub['gridentries']
    pixels_PP             = OMHsub['ppentries']
    uncert                = OMHsub['col_uncertainty_OMI']
    firefilter            = OMHsub['firemask']+OMHsub['smokemask']
    anthrofilter          = OMHsub['anthromask']

    lats_lr,lons_lr, lats_e_lr, lons_e_lr = util.lat_lon_grid(GMAO.__LATRES_GC__,GMAO.__LONRES_GC__)

    lati_lr,loni_lr = util.lat_lon_range(lats_lr,lons_lr,region)
    lats_lr,lons_lr = lats_lr[lati_lr], lons_lr[loni_lr]

    VCC_OMI_lr=util.regrid_to_lower(VCC_OMI,omilats,omilons,lats_lr,lons_lr,pixels=pixels)
    pixels_lr=util.regrid_to_lower(pixels,omilats,omilons,lats_lr,lons_lr,func=np.nansum)
    VCC_OMI_lr_2=util.regrid_to_lower(VCC_OMI,omilats,omilons,lats_lr,lons_lr) # unwweighted means

    # Map showing before and after binning: also counts...
    plt.figure(figsize=(16,16))
    plt.subplot(321)
    pp.createmap(VCC_OMI,omilats,omilons,vmin=1e14,vmax=4e16,
                 title='HighRes VCC OMI', region=region)
    plt.subplot(322)
    pp.createmap(VCC_OMI_lr,lats_lr,lons_lr,vmin=1e14,vmax=4e16,
                 title='LowRes VCC OMI',region=region)
    plt.subplot(323)
    pp.createmap(VCC_OMI_lr_2,lats_lr,lons_lr,vmin=1e14,vmax=4e16,
                 title='LowRes(unweighted) VCC OMI',region=region)
    plt.subplot(324)
    pp.createmap((VCC_OMI_lr - VCC_OMI_lr_2)/VCC_OMI_lr_2, lats_lr,lons_lr,
                 vmin=-0.5, vmax=.5, title='(LR-LR(unweighted)) / LR(unweighted)',
                 linear=True, region=region, cmapname='bwr')
    plt.subplot(325)
    pp.createmap(pixels,omilats,omilons,vmin=0,vmax=7,title='HighRes pixels',
                 linear=True, region=region)
    plt.subplot(326)
    pname='test_lower_resolution.png'
    pp.createmap(pixels_lr,lats_lr,lons_lr,vmin=0,vmax=200,title='LowRes pixels',
                 linear=True, pname=pname, region=region)
