'''
    File to test various calculations
    Tests for coding are moved to utilities/tests.py
'''
## Modules
import matplotlib
matplotlib.use('Agg') # don't actually display any plots, just create them

# my file reading and writing module
from utilities import fio
import reprocess
from utilities import plotting as pp
from utilities.JesseRegression import RMA
from utilities import utilities as util
from classes.omhchorp import omhchorp as omrp
from utilities.utilities import match_bottom_levels
from classes.GC_class import GC_tavg, GC_sat

# Tests are pulled in from tests/blah.py
from tests import check_files, test_filters, RSC_tests, test_E_new

import numpy as np
from numpy.ma import MaskedArray as ma
from scipy import stats
from copy import deepcopy as dcopy

from datetime import datetime#, timedelta

from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
from matplotlib.ticker import FormatStrFormatter # tick formats



#import matplotlib.patches as mpatches
import seaborn # kdeplots

import random # random number generation
import timeit
# EG using timer:
#start_time=timeit.default_timer()
#runprocess()
#elapsed = timeit.default_timer() - start_time
#print ("TIMEIT: Took %6.2f seconds to runprocess()"%elapsed)

##############################
######### GLOBALS ############
##############################

# STRINGS FOR THINGS
Ohcho='$\Omega_{HCHO}$' # VC of HCHO
Ovc='\Omega'
Og='$\Omega_{G}$'       # GEOS-Chem VC
Ogc='$\Omega_{GC}$'     # GC VCC
Op='$\Omega_{P}$'       # Palmer VC
Opc='$\Omega_{PC}$'     # Palmer VCC
Oo='$\Omega_{O}$'       # OMI VC
Ooc='$\Omega_{OC}$'     # OMI VCC

##############################
########## FUNCTIONS #########
##############################

def mmm(arr):
    print("%1.5e, %1.5e, %1.5e"%(np.nanmin(arr),np.nanmean(arr),np.nanmax(arr)))

def check_array(array, nonzero=False):
    '''
    print basic stuff about an array
    '''
    arrayt=array
    if nonzero:
        arrayt=array[np.nonzero(array)]
    print ('mean :%f'%np.nanmean(arrayt))
    print ('min :%f'%np.nanmin(arrayt))
    print ('max :%f'%np.nanmax(arrayt))
    print ('count :%f'%np.sum(np.isfinite(arrayt)))
    print ('shape :'+str(np.shape(arrayt)))




#############################################################################
######################       TESTS                  #########################
#############################################################################


def check_HEMCO_restarts():
    '''
        Check how different the hemco restarts are between UCX and tropchem
    '''
    fpat='Data/GC_Output/%s/restarts/HEMCO_restart.200501010000.nc'
    fucx=fpat%'UCX_geos5_2x25'
    ftrp=fpat%'geos5_2x25_tropchem'
    ucx=fio.read_netcdf(fucx)
    trp=fio.read_netcdf(ftrp)
    print("Name, run, shape, mean")
    for name in ucx.keys():#['PARDF_DAVG','PARDR_DAVG']:
        ucx_mean=np.nanmean(ucx[name])
        print (name, "ucx", ucx[name].shape, ucx_mean)
        if name in trp.keys():
            trp_mean=np.nanmean(trp[name])
            print(name, 'trop', trp[name].shape ,trp_mean)
            print(['NOT EQUAL','EQUAL'][np.isclose(trp_mean,ucx_mean)])

    return None



def compare_GC_OMI_new(date=datetime(2005,1,1),aus=True):
    '''
    '''
    vmin=1e14; vmax=1e17
    lllat=-75; urlat=75; lllon=-170; urlon=170
    austr=''
    dstr=date.strftime("%Y%m%d")
    if aus:
        lllat=-45; urlat=-8; lllon=108; urlon=156
        austr='AUS_'

    f=plt.figure(figsize=[13,13])
    # Figure of OMIcc, VCC, GC, diffs
    plt.subplot(223)
    gc=fio.read_gchcho(date)
    m,cs,cb=gc.PlotVC(lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=vmin, vmax=vmax, cm2=True)
    #pname="Figs/map_%sGC_%s.png"%(austr,dstr)
    plt.title(Ogc)

    om=omrp(date)
    lats=om.latitude
    lons=om.longitude
    VCC_OMI=om.VC_OMI_RSC
    VCC=om.VCC
    # plot the regridded OMI and reprocessed maps:
    for arr,arrstr,i in zip([VCC_OMI,VCC],[Oomi,Ogc],range(2)):
        plt.subplot(221+i)
        pp.createmap(arr,lats,lons, vmin=vmin, vmax=vmax, latlon=True,
              lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, colorbar=False)
        plt.title(arrstr)

    # relative differences
    #plt.subplot(224)
    #diffs=100.0*(VCC - VCC_OMI) / VCC_OMI
    #pp.linearmap(diffs,lats,lons,vmin=vmin,vmax=vmax,latlon=True,
    #          lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    #plt.title("%s vs %s relative difference"%(Ogc,Oomi))

    pname="Figs/map_GOM_%s%s.png"%(austr,dstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close()


def compare_products(month=datetime(2005,1,1), positiveonly=False,
                     region=pp.__AUSREGION__):
    '''
    look at remapped hcho (1day, 1month)
    Plot VCs, both OMI and Reprocessed, and the RMA corellations
    TODO: Update to look only at corrected, including RandalMartin calculated
    '''



    #ax=plt.subplot(223)
    #m,cs,cb = pp.linearmap(100.0*(omi-vcc)/omi, lats, lons,
    #                       vmin=-120, vmax=120,
    #                       lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #plt.title('100(%s-%s)/%s'%(somi,sgc,somi))
    #plt.subplot(212)
    #if positiveonly:
    #    omi[omi<0]=np.NaN
    #    vcc[vcc<0]=np.NaN
    #pp.plot_corellation(omi, vcc, oceanmask=oceanmask,verbose=True)
    #plt.title('RMA corellation')
    #plt.xlabel(somi)
    #plt.ylabel(sgc)

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    plt.suptitle(yyyymmdd, fontsize=30)
    plt.tight_layout()
    outfig="Figs/%sproducts%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")


def check_products(month=datetime(2005,1,1), region=pp.__AUSREGION__):
    '''
    Test a day or 8-day reprocessed HCHO map
    Plot VCs, both OMI and Reprocessed, and the RMA corellations
    '''

    lllat, lllon, urlat, urlon= region

    # Grab reprocessed OMI data
    om=omrp(month,util.last_day(month),keylist=['VCC_OMI','VCC_GC','VCC_PP'])

    #oceanmask=om.oceanmask  # true for ocean squares
    subsets=util.lat_lon_subset(om.lats,om.lons,region,[om.VCC_OMI, om.VCC_GC, om.VCC_PP], has_time_dim=True)
    lats,lons=subsets['lats'],subsets['lons']

    # Plot
    # 1-day: VCC_omi, VCC_GC, VCC_PP
    # 1-month: '', '', ''
    f = plt.figure(num=0,figsize=(16,14))
    titles=['OMI','GC','PP']
    vmin,vmax=1e15,1e16
    cmapname='YlOrBr'
    for i,VC in enumerate(subsets['data']):
        for j in [0,1]:
            VCj=[VC[0],np.nanmean(VC,axis=0)][j]
            plt.subplot(2,3,1+i+3*j)
            bmap,cs,cb = pp.createmap(VCj,lats,lons, region=region,
                                   vmin=vmin,vmax=vmax,
                                   title=[titles[i],''][j], cmapname=cmapname,
                                   colorbar=False)

    yms = month.strftime('%Y%m')
    plt.suptitle('$\Omega_{HCHO}$ for %s'%yms,fontsize=35)
    pp.add_colourbar(f,cs,label='molec/cm2',fontsize=24)

    outfig=month.strftime("Figs/VCC_check_%Y%m.png")
    #plt.tight_layout()
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def Test_Uncertainty(month=datetime(2005,1,1), region=pp.__AUSREGION__):
    '''
        Effect on provided uncertainty with 8 day averaging
        Also calculate uncertainty as in DeSmedt2012
        2 rows, 3 Columns:
             |  1 day  |   1 month    |  1 month filtered
        HCHO |
        Unc. |

    '''
    yms=month.strftime("%Y%m")
    lllat, lllon, urlat, urlon=region

    # Grab reprocessed OMI data: OMI columns, masks, uncertainties, entries
    keylist=['VCC_OMI','firemask','anthromask','smokemask',
             'col_uncertainty_OMI','gridentries']
    om = omrp(month,util.last_day(month),keylist=keylist)
    lats,lons=om.lats,om.lons

    # 1 day, 1 month, 1 month filtered
    vccday=om.VCC_OMI[0]
    vccmon=np.nanmean(om.VCC_OMI,axis=0)
    vccmasked = np.copy(om.VCC_OMI)
    counts = om.gridentries
    countsmasked = np.copy(om.gridentries)
    mask = (om.firemask+om.anthromask+om.smokemask).astype(np.bool)
    vccmasked[mask] = np.NaN
    countsmasked[mask] = 0
    vccmasked=np.nanmean(vccmasked,axis=0)
    # same but for uncertainty
    unc=om.col_uncertainty_OMI
    # divide avg uncertainty by sqrt count
    with np.errstate(divide='ignore', invalid='ignore'):
        uncday=unc[0] / np.sqrt(counts[0])
        uncmon=np.nanmean(unc,axis=0)/ np.nansum(counts,axis=0)
        uncmasked=np.nanmean(unc,axis=0)/np.nansum(countsmasked,axis=0)
        # Make sure div by zero are nans and not infinites
        for arr in [uncday,uncmon,uncmasked]:
            arr[~np.isfinite(arr)] = np.NaN

    HCHO    = [vccday, vccmon, vccmasked]
    UNCERTS  = [uncday, uncmon, uncmasked]

    f = plt.figure(num=0,figsize=(16,14))
    titles= [ '$\Omega_{HCHO}$ '+s for s in ['Day', 'Month', 'Month masked'] ]
    vmin,vmax=[1e15,1e14],[1e16,1e15]
    cmapnames='YlOrBr','Reds'
    for i in range(3):
        for j in range(2):
            arr=[HCHO[i],UNCERTS[i]][j]
            plt.subplot(2,3,1+i+3*j)
            bmap,cs,cb = pp.createmap(arr,lats,lons, region=region,
                                      vmin=vmin[j],vmax=vmax[j],
                                      title=[titles[i],'Uncertainty'][j],
                                      cmapname=cmapnames[j],
                                      clabel='molec/cm2')
            ta=plt.gca().transAxes
            plt.text(0.05,.05, 'mean=%.1e'%np.nanmean(arr),transform=ta)

    yms = month.strftime('%Y%m')
    plt.suptitle('OMI HCHO and uncertainty for %s'%yms,fontsize=35)
    #pp.add_colourbar(f,cs,label='molec/cm2',fontsize=24)
    outfig="Figs/Uncertainty_OMI_%s.png"%yms
    plt.tight_layout()
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def plot_swaths(region=pp.__AUSREGION__):
    '''
    plot swaths
    '''
    d0=datetime(2005,1,1)
    for dn in [datetime(2005,1,1),datetime(2005,1,8),datetime(2005,1,31)]:
        om=omrp(d0,dn)
        pname='Figs/swath_%s.png'%dn.strftime("%Y%m%d")
        cmargs={'pname':pname}
        om.plot_map(day0=d0,dayn=dn,**cmargs)

def look_at_swaths(date=datetime(2005,1,1), lllat=-80, lllon=-179, urlat=80, urlon=179, pltname=""):
    '''
        Plot one swath -> one day -> 8 days .
    '''

    # Grab OMI data
    omi_8=fio.read_omhcho_8days(date)
    omi_1=fio.read_omhcho_day(date)
    omi_s=fio.read_omhcho(fio.determine_filepath(date))

    counts= omi_8['counts']
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lonmids=omrp['longitude']
    latmids=omrp['latitude']
    lons,lats = np.meshgrid(lonmids,latmids)

    # Plot
    # SC, VC_omi, AMF_omi
    # VCC, VC_gc, AMF_GC
    # VC_OMI-GC, VC_GC-GC, GC_map
    # cuncs, AMF-correlation, AMF_GCz

    f, axes = plt.subplots(4,3,num=0,figsize=(16,20))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0,0])
    m,cs,cb = pp.createmap(omrp['SC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = pp.createmap(omrp['VC_OMI'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[0,2])
    m,cs,cb = pp.linearmap(omrp['AMF_OMI'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = pp.createmap(omrp['VCC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = pp.createmap(omrp['VC_GC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC GC')
    plt.sca(axes[1,2])
    m,cs,cb = pp.linearmap(omrp['AMF_GC'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GC}$')

    # Plot third row the GEOS-Chem map divergences

    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    OMIdiff=100*(omrp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omrp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[2,0])
    vmin,vmax = -150, 150
    m,cs,cb = pp.linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #m,cs,cb = pp.createmap(gc.VC_HCHO*1e-4, glats,glons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[2,1])
    m,cs,cb = pp.linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[2,2])
    m,cs,cb = pp.createmap(fineHCHO, lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')

    # plot fourth row: uncertainties and AMF_GCz
    #
    plt.sca(axes[3,0])
    m,cs,cb = pp.createmap(omrp['col_uncertainty_OMI'], lats, lons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('col uncertainty (VC$_{OMI} \pm 1 \sigma$)')
    plt.sca(axes[3,1])
    vc_gc,vc_omi=omrp['VC_GC'],omrp['VC_OMI']
    plt.scatter(vc_gc,vc_omi)
    plt.xlabel('VC$_{GC}$')
    plt.ylabel('VC$_{OMI}$')
    scatlims=[1e12,2e17]
    plt.yscale('log'); plt.ylim(scatlims)
    plt.xscale('log'); plt.xlim(scatlims)
    plt.plot(scatlims,scatlims,'k--',label='1-1') # plot the 1-1 line for comparison
    slp,intrcpt,r,CI1,CI2=RMA(vc_gc,vc_omi) # get regression
    plt.plot(scatlims, slp*np.array(scatlims)+intrcpt,color='red',
            label='slope=%4.2f, r=%4.2f'%(slp,r))
    plt.legend(title='lines',loc=0)
    plt.title("VC$_{GC}$ vs VC$_{OMI}$")
    plt.sca(axes[3,2])
    m,cs,cb = pp.linearmap(omrp['AMF_GCz'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GCz}$')

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%scorrected%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def Check_AMF_relevelling():
    '''
    Create a few example arrays and relevel the pressures and see what happens
    '''
    p0=np.array([1000,800,600,400,100,10])
    a0=np.array([10,9,8,7,6,5])
    p1=np.array([1100, 1050, 950, 500, 200, 80, 5])
    a1=np.array([9,8.5,8,7,6,5,4])
    p2=np.array([810, 650, 450, 250, 20, 1])
    a2=np.array([11,8,7,5,4,3])
    p3=np.array([1250, 1100, 900, 400, 100])
    a3=np.array([11,8,9,4,2.3])
    xlims=[2,12]
    ylims=[1300,.5]

    p00,p10,a00,a10=match_bottom_levels(p0.copy(),p1.copy(),a0.copy(),a1.copy())
    #print((p1,p10,a1,a10))
    assert p1[0]!=p10[0] and a1[0]!=a10[0], 'Err: No change in lowest level!'
    assert (a0 == a00).all() and (p0 == p00).all(), 'Err: Higher levels changed!'
    p1c=p1.copy() # make sure arguments aren't edited?
    p00,p10,a00,a10=match_bottom_levels(p0,p1c,a0,a1)
    assert (p1c==p1).all(), 'Err: Argument changed!'
    p11,p21,a11,a21=match_bottom_levels(p1,p2,a1,a2)
    p12,p32,a12,a32=match_bottom_levels(p1,p3,a1,a3)

    # look at 0vs1, 1vs2, 1vs3
    f,axes=plt.subplots(1,3,sharey=True,sharex=True,squeeze=True)

    # loop over comparisons
    for i, arrs in enumerate(((a0,p0,a1,p1,a10,p10),(a1,p1,a2,p2,a11,p11),(a1,p1,a3,p3,a32,p32))):
        aa,pa,ab,pb,an,pn = arrs
        plt.sca(axes[i])
        plt.plot(aa,pa, 'mo--',label='aa')
        plt.plot(ab,pb, 'b^--',label='ab')
        plt.plot(an,pn, 'r*',label='new ab')
        plt.plot(xlims,[pn[0],pn[0]],'k--', alpha=0.4) # horizontal line for examination

    plt.yscale('log')
    plt.ylim(ylims); plt.xlim([2,12])
    plt.legend(loc='best')

    plt.suptitle('Surface relevelling check')
    pname='Figs/SurfaceRelevelCheck.png'
    plt.savefig(pname)
    print(pname+' Saved')

def Check_OMI_AMF():
    '''
    See if the AMF_G is used to form the AMF in the OMI dataset
    '''
    # get one day of pixels
    om=fio.read_omhcho('omhcho/OMI-Aura_L2-OMHCHO_2005m0101t0020-o02472_v003-2014m0618t141821.he5',maxlat=60)
    #{'HCHO':hcho,'lats':lats,'lons':lons,'AMF':amf,'AMFG':amfg,
    #'omega':w,'apriori':apri,'plevels':plevs, 'cloudfrac':clouds,
    #'rad_ref_col':ref_c, 'RSC_OMI':rsc_omi,
    #'qualityflag':qf, 'xtrackflag':xqf, 'sza':sza,
    #'coluncertainty':cunc, 'convergenceflag':fcf, 'fittingRMS':frms}

    amf=om['AMF']
    goods=~np.isnan(amf)
    amf=np.array(amf[goods])
    s=np.array(om['apriori'][:,goods])
    w=np.array(om['omega'][:,goods])
    amfg=np.array(om['AMFG'][goods])
    p=np.array(om['plevels'][:,goods])

    dp=np.zeros([47,len(amf)])
    dp[-1,:]=p[-1,:]
    dp[0:46,:]=p[1:47,:]-p[0:46,:]
    dp[-1,:]=0.01 # top level pressure edge defined as 0.01hPa
    chris=np.sum(s*dp*w,axis=0) / np.sum(s*dp,axis=0)
    plt.scatter(chris,amf)
    plt.title('AMF calculation check on one OMI swath')
    plt.xlabel('Chris AMF')
    plt.ylabel('OMI AMF')
    plt.plot([0,3.5],[0,3.5],'--',label='1-1 line')
    plt.text(-0.5,3.4,"Chris AMF = $\Sigma_i (Shape(P_i) * \omega(P_i) * \Delta P_i) /  \Sigma_i (Shape(P_i) * \omega(P_i) )$")
    plt.text(-0.5,1.5,"mean AMF_G = %4.2f"%np.mean(amfg))
    plt.legend(loc=0)
    pname='Figs/Chris_AMF_Check.png'
    plt.savefig(pname)
    print('saved %s'%pname)

def Check_AMF():
    '''
    Check the various AMF's against eachother
    '''
    # get one day of pixels
    gps=reprocess.get_good_pixel_list(datetime(2005,1,1))
    amf=gps['AMF_OMI']
    amf_rm=gps['AMF_RM']
    amf_gc=gps['AMF_GC']
    lims=[0,4]
    plt.figure(0,figsize=[26,10])
    plt.subplot(131)
    pp.plot_corellation(amf,amf_gc,lims=lims,logscale=False)
    plt.xlabel('OMI AMF'); plt.ylabel('GC AMF')
    plt.subplot(132)
    pp.plot_corellation(amf,amf_rm,lims=lims,logscale=False)
    plt.xlabel('OMI AMF'); plt.ylabel('RM AMF')
    plt.subplot(133)
    pp.plot_corellation(amf_gc,amf_rm,lims=lims,logscale=False)
    plt.xlabel('GC AMF'); plt.ylabel('RM AMF')
    pname='Figs/Compare_AMFs.png'
    plt.savefig(pname)
    print('saved %s'%pname)




def Summary_Single_Profile():
    '''
    Get a single good pixel and show all the crap it needs to go through
    '''
    # plot left to right:
    # OMI Apriori, GC Apriori, Omega, AMF, AMF_new, VC_OMI, VCC
    day=datetime(2005,1,1)

    # Read OMHCHO data ( using reprocess get_good_pixels function )
    pixels=reprocess.get_good_pixel_list(day, getExtras=True)

    #N = len(pixels['lat']) # how many pixels do we have
    #i=random.sample(range(N),1)[0]
    #print(i)
    i=153375
    lat,lon=pixels['lat'][i],pixels['lon'][i]
    omega=pixels['omega'][:,i]
    AMF_G=pixels['AMF_G'][i]
    AMF_OMI=pixels['AMF_OMI'][i]
    w_pmids=pixels['omega_pmids'][:,i]
    apri=pixels['apriori'][:,i]
    SC=pixels['SC'][i]
    AMF_GC=pixels['AMF_GC'][i]
    VC_GC=SC/AMF_GC
    VC_OMI=SC/AMF_OMI

    # read gchcho
    #lat,lon=0,0
    gchcho = fio.read_gchcho(day)
    gcapri, gc_smids = gchcho.get_single_apriori(lat,lon)
    gc_pmids    = gchcho.get_single_pmid(lat,lon)

    # rerun the AMF calculation and plot the shampoo
    #innerplot='Figs/AMF_test/AMF_test_innerplot%d.png'%i
    AMFS,AMFZ = gchcho.calculate_AMF(omega, w_pmids, AMF_G, lat, lon)

    print( "AMF_s=", AMFS )
    print( "AMF_z=", AMFZ )
    print( "AMF_o=", AMF_OMI )

    # Also make a plot of the regression new vs old AMFs
    f=plt.figure(figsize=(9,12))

    plt.plot(apri,w_pmids,'-k_',linewidth=2,markersize=20)
    plt.xlabel('Molecules cm$^{-2}$')
    plt.ylabel('hPa')
    ax=plt.twiny()
    plt.plot(gcapri, gc_pmids,'-r_',linewidth=2,markersize=20)
    plt.plot(omega, w_pmids, '-',linewidth=2,color='fuchsia')
    plt.title('Old vs new Apriori (%4.1fN, %4.1fE)'%(lat,lon),y=1.06)
    ax.tick_params(axis='x', colors='red')
    plt.ylim([1015, .04])
    plt.yscale('log')

    # omhchorp final vert column
    om=omrp(day,oneday=True)
    omlat,omlon=om.latitude, om.longitude
    omlati,omloni= (np.abs(omlat-lat)).argmin(),(np.abs(omlon-lon)).argmin()
    vcc=om.VCC[omlati,omloni] # corrected averaged grid square value
    ta=plt.gca().transAxes
    fs=30

    plt.text(.3,.95, 'AMF$_{OMI}$=%5.2f'%AMF_OMI,transform=ta, fontsize=fs)
    plt.text(.3,.86, 'AMF$_{GC}$=%5.2f'%AMFS,transform=ta, fontsize=fs)
    plt.text(.3,.77, '$\Omega_{OMI}$=%4.2e'%VC_OMI,transform=ta, fontsize=fs)
    plt.text(.3,.68, '$\Omega_{GC}$=%4.2e'%VC_GC, transform=ta, fontsize=fs)
    plt.text(.3,.60, '$\Omega_{GCC}$=%4.2e'%vcc, transform=ta, fontsize=fs)
    plt.text(.3,.47, 'OMI a priori',color='k',transform=ta, fontsize=27)
    plt.text(.3,.41, 'GC a priori',color='r',transform=ta, fontsize=27)
    plt.text(.3,.35, '$Scattering Weights$',color='fuchsia',transform=ta, fontsize=27)

    fname='Figs/SummarySinglePixel.png'
    f.savefig(fname)
    print("Saved "+fname)
    plt.close(f)

def check_timeline():
    '''
    '''
    print("Check Timeline test to be implemented")

def test_reprocess_corrected(date=datetime(2005,1,1), oneday=True, lllat=-80, lllon=-179, urlat=80, urlon=179,pltname=""):
    '''
    Test a day or 8-day reprocessed HCHO map
    Plot VCs, both OMI and Reprocessed,
        as well as AMFs and comparison against GEOS-Chem model.
    '''

    # Grab one day of reprocessed OMI data
    omhchorp=fio.read_omhchorp(date,oneday=oneday)

    counts = omhchorp['gridentries']
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lonmids=omhchorp['longitude']
    latmids=omhchorp['latitude']
    lons,lats = np.meshgrid(lonmids,latmids)

    # Plot
    # SC, VC_omi, AMF_omi
    # VCC, VC_gc, AMF_GC
    # VC_OMI-GC, VC_GC-GC, GC_map
    # cuncs, AMF-correlation, AMF_GCz

    f, axes = plt.subplots(4,3,num=0,figsize=(16,20))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0,0])
    m,cs,cb = pp.createmap(omhchorp['SC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = pp.createmap(omhchorp['VC_OMI'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[0,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_OMI'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = pp.createmap(omhchorp['VCC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = pp.createmap(omhchorp['VC_GC'],lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('VC GC')
    plt.sca(axes[1,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_GC'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GC}$')

    # Plot third row the GEOS-Chem map divergences

    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    OMIdiff=100*(omhchorp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omhchorp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[2,0])
    vmin,vmax = -150, 150
    m,cs,cb = pp.linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    #m,cs,cb = pp.createmap(gc.VC_HCHO*1e-4, glats,glons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[2,1])
    m,cs,cb = pp.linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[2,2])
    m,cs,cb = pp.createmap(fineHCHO, lats,lons,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')

    # plot fourth row: uncertainties and AMF_GCz
    #
    plt.sca(axes[3,0])
    m,cs,cb = pp.createmap(omhchorp['col_uncertainty_OMI'], lats, lons, lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('col uncertainty (VC$_{OMI} \pm 1 \sigma$)')
    plt.sca(axes[3,1])
    vc_gc,vc_omi=omhchorp['VC_GC'],omhchorp['VC_OMI']
    plt.scatter(vc_gc,vc_omi)
    plt.xlabel('VC$_{GC}$')
    plt.ylabel('VC$_{OMI}$')
    scatlims=[1e12,2e17]
    plt.yscale('log'); plt.ylim(scatlims)
    plt.xscale('log'); plt.xlim(scatlims)
    plt.plot(scatlims,scatlims,'k--',label='1-1') # plot the 1-1 line for comparison
    slp,intrcpt,r,CI1,CI2=RMA(vc_gc,vc_omi) # get regression
    plt.plot(scatlims, slp*np.array(scatlims)+intrcpt,color='red',
            label='slope=%4.2f, r=%4.2f'%(slp,r))
    plt.legend(title='lines',loc=0)
    plt.title("VC$_{GC}$ vs VC$_{OMI}$")
    plt.sca(axes[3,2])
    m,cs,cb = pp.linearmap(omhchorp['AMF_GCz'],lats,lons,vmin=0.6,vmax=6.0,lllat=lllat,lllon=lllon,urlat=urlat,urlon=urlon)
    plt.title('AMF$_{GCz}$')

    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="Figs/%scorrected%s%s.png"%(onedaystr, yyyymmdd, pltname)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

def analyse_VCC_pp(day=datetime(2005,3,1), oneday=False, region=pp.__AUSREGION__):
    '''
    Look closely at AMFs over Australia, specifically over land
    and see how our values compare against the model and OMI swaths.and Paul Palmers code
    '''
    # useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,dayn=util.last_day(day))
    lats,lons=om.lats,om.lons

    # AMF Subsets
    subsets=util.lat_lon_subset(lats,lons,region,data=[om.AMF_OMI,om.AMF_GC,om.AMF_PP,om.VC_OMI_RSC,om.VCC,om.VCC_PP],has_time_dim=True)
    lats,lons=subsets['lats'],subsets['lons']
    for i,istr in enumerate(['AMF (OMI)', 'AMF (GC) ', 'AMF (PP) ']):
        dat=subsets['data'][i]
        print("%s mean : %7.4f, std: %7.4f"%(istr, np.nanmean(dat),np.nanstd(dat)))

    #unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)
    oceanmask=maskoceans(mlons,mlats,mlons,inlands=0).mask
    oceanmask3d=np.repeat(oceanmask[np.newaxis,:,:],om.n_times,axis=0)

    # the datasets with nans and land or ocean masked
    OMP_l = [] # OMI, My, Paul palmer
    OMP_o = [] # OMI, My, Paul palmer
    OMP_str = ['OMI_RSC','VCC', 'VCC_PP']
    OMP_col = ['k','r','m']
    land_data=[]
    ocean_data=[]
    for arr in subsets['data'][3:]:

        alsonans=np.isnan(arr)
        OMP_l.append(ma(arr, mask=oceanmask3d)) # as a masked array
        OMP_o.append(ma(arr,  mask=~oceanmask3d))
        land_data.append(arr[~oceanmask3d * ~alsonans ]) # as a list of data
        ocean_data.append(arr[oceanmask3d * ~alsonans])
    #ocean_data=np.transpose(ocean_data)
    #land_data=np.transpose(land_data)

    # Print the land and ocean averages for each product
    print("%s land averages:"%str(region))
    print("%25s   land,   ocean"%'')
    for arrl, arro, arrstr in zip(OMP_l,OMP_o,OMP_str):
        print("%21s: %5.3e,  %5.3e "%(arrstr, np.nanmean(arrl),np.nanmean(arro)))

    # Plot the histogram of VC entries land and sea
    f=plt.figure(figsize=(14,10))
    #land_data=np.transpose([OMP_l[ii][landinds] for ii in range(3)])
    #ocean_data=np.transpose([OMP_o[ii][oceaninds] for ii in range(3)])

    olabel=['ocean '+thing for thing in OMP_str]
    llabel=['land ' +thing for thing in OMP_str]
    print(np.shape(land_data),np.shape(land_data[0]))
    plt.hist(ocean_data, bins=np.logspace(13, 17, 20), color=['k','darkblue','lightblue'], label=olabel)
    plt.hist(land_data, bins=np.logspace(13, 17, 20), color=['grey','orange','yellow'], label=llabel)
    plt.xscale("log")
    plt.yscale('log',nonposy='clip') # logarithmic y scale handling zero
    plt.title('Vertical column distributions ($\Omega_{HCHO}$)',fontsize=26)
    plt.ylabel('frequency'); plt.xlabel('molec cm$^{-2}$')
    plt.legend(loc='center left')
    ta=plt.gca().transAxes
    plt.text(0.05,.95, 'land count=%d'%np.sum(~oceanmask3d),transform=ta)
    plt.text(.05,.90, 'ocean count=%d'%np.sum(oceanmask3d),transform=ta)
    for ii in range (3):
        plt.text(.05,.85-0.05*ii, '%s mean(land)=%5.3e'%(OMP_str[ii],np.nanmean(OMP_l[ii])), transform=ta)

    timestr=['_month','_day'][oneday]
    pname='Figs/hist%s_%s.png'%(timestr,ymdstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

    # show corellations for OMI_RSC- VCC
    f=plt.figure(figsize=(18,7))

    nans_l=np.isnan(OMP_l[0])
    nans_o=np.isnan(OMP_o[0])
    for ii in range(3):
        nans_l=nans_l+np.isnan(OMP_l[ii]) + OMP_l[ii].mask # all the nans we won't fit
        nans_o=nans_o+np.isnan(OMP_o[ii]) + OMP_o[ii].mask
    print( "nans_l")
    print(np.shape(nans_l))
    print("%d nans in land data "%np.sum(nans_l))
    for ii in range(3):
        plt.subplot(131+ii)
        # pp.plot_corellation()
        plt.scatter(OMP_o[ii], OMP_o[ii-1], color='blue', label="Ocean", alpha=0.4)
        plt.scatter(OMP_l[ii], OMP_l[ii-1], color='gold', label="Land")
        m,x0,r,ci1,ci2 = RMA(OMP_l[ii][~nans_l], OMP_l[ii-1][~nans_l])
        X=np.array([1e10,5e16])
        plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%.5f, r=%.5f'%(m,r))
        plt.plot( X, X, 'k--', label='1-1' )
        plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
        plt.ylabel(OMP_str[ii-1]); plt.xlabel(OMP_str[ii])

    # save plot
    pname='Figs/correlations%s_%s.png'%(timestr,ymdstr)
    f.suptitle("Product comparison for %s"%ymdstr,fontsize=28)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def plot_VCC_rsc_gc_pp_fireaffects(d0=datetime(2005,3,1),dn=None,region=[-45, 99.5, -11, 160.0]):
    '''
        Plot columns with different amf bases
        also different fire filtering strengths
              |  VCC   |  VCC_gc   |  VCC_pp
        fire0 |
        fire1 |
        fire2 |
        fire4 |
        fire8 |
    '''

    # start by reading all the VCC stuff
    # useful strings
    ymdstr=d0.strftime('%Y%m%d')
    if dn is not None:
        ymdstr=ymdstr+'-%s'%dn.strftime('%Y%m%d')
    pname='Figs/VCC_fires_%s.png'%ymdstr
    pname2='Figs/VCC_entries_%s.png'%ymdstr
    vmin,vmax=4e15,9e15 # min,max for colourbar
    linear=True # linear colour scale?
    vmin2,vmax2=0,40

    # time stuff:
    start_time=timeit.default_timer()

    # read in omhchorp
    om=omrp(d0,dayn=dn, keylist=['VC_OMI_RSC','VCC','VCC_PP','gridentries','ppentries'])
    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to read omhchorp"%elapsed)

    #print(vars(om).keys()) # [3, 720, 1152] data arrays returned, along with lats/lons etc.
    oceanmask=om.oceanmask # lets just look at land squares

    # Regionmask is the area which isnt within the region subset
    regionmask=~om.inds_subset(lat0=region[0],lat1=region[2],lon0=region[1],lon1=region[3])

    # Plot rows,cols,size:
    f,axes=plt.subplots(5,3,figsize=[18,18])
    # second plot just for entries
    f2, axes2=plt.subplots(5,3,figsize=[18,18])

    # first line is maps of VCC, VC_GC, VCC_PP
    titles=["OMI VCC","S(z) updated","S(z)+$\omega$(z) updated"]

    for i,arr in enumerate([om.VC_OMI_RSC, om.VCC, om.VCC_PP]):
        # entries for normal or ppamf
        entries=np.copy([om.gridentries,om.ppentries][i==2])
        entries=entries.astype(np.float) # so I can Nan the ocean/non-aus areas

        if len(np.shape(arr))==3:
            arr=np.nanmean(arr,axis=0) # average over time
            entries=np.nansum(entries,axis=0) # how many entries
        arr[oceanmask]=np.NaN # nanify the ocean
        arr[regionmask]=np.NaN # nanify outside the region
        entries[oceanmask]=np.NaN
        entries[regionmask]=np.NaN

        plt.sca(axes[0,i]) # first row ith column
        m,cs,cb= pp.createmap(arr,om.lats,om.lons,
                              region=region,
                              linear=linear,vmin=vmin,vmax=vmax,
                              cmapname='rainbow',colorbar=False)
        plt.title(titles[i])

        # add a little thing showing entries and mean and max
        txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries),np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr), 'max = %.2e'%np.nanmax(arr)]
        for txt, yloc in zip(txt,[0.01,0.07,0.13]):
            plt.text(0.01, yloc, txt,
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=plt.gca().transAxes,
                 color='k', fontsize=10)

        # also plot entries
        plt.sca(axes2[0,i])
        pp.createmap(entries,om.lats,om.lons,region=region,
                     cmapname='jet', colorbar=False,
                     linear=True,vmin=vmin2,vmax=vmax2)
        plt.title(titles[i])

    # Now loop over the same plots after NaNing our different fire masks
    for j, N in enumerate([1,2,4,8]):

        # read in fire mask for 1,2,4,8 days prior masking
        # firemask is 3dimensional: [days,lats,lons]
        fstart=timeit.default_timer()
        firemask=om.make_fire_mask(d0,dN=dn,days_masked=N)
        felapsed = timeit.default_timer() - fstart
        print ("TIMEIT: Took %6.2f seconds to make_fire_mask(%d days)"%(felapsed,N))

        for i, arr in enumerate([om.VC_OMI_RSC, om.VCC, om.VCC_PP]):
            plt.sca(axes[j+1,i]) # jth row ith column

            # pixelcounts for normal or ppamf
            entries=np.copy([om.gridentries,om.ppentries][i==2]).astype(np.float)

            # Nanify fire squares
            arr[firemask]=np.NaN

            # masking to our specific region:
            if len(np.shape(arr))==3:
                for k in range(np.shape(arr)[0]):
                    entries[k][regionmask]=np.NaN

            firepix=int(np.nansum(entries[firemask])) # how many fire pixels
            entries[firemask]=0
            # flatten arrays for plotting and entries
            if len(np.shape(arr))==3:
                arr=np.nanmean(arr,axis=0) # average over time
                entries=np.nansum(entries,axis=0) # how many entries

            # Nanify ocean squares and area outside region
            arr[regionmask]=np.NaN
            arr[oceanmask]=np.NaN
            entries[regionmask]=np.NaN
            entries[oceanmask]=np.NaN

            plt.sca(axes[j+1,i]) # first row ith column
            pp.createmap(arr,om.lats,om.lons,region=region,
                         linear=linear,vmin=vmin,vmax=vmax,
                         cmapname='rainbow', colorbar=False)

            # add a little thing showing entries and mean and max
            txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries), np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr),
                 'max = %.2e'%np.nanmax(arr), 'firepix=%d'%firepix]
            for txt, yloc in zip(txt,[0.01,0.07,0.13,0.19]):
                plt.text(0.01, yloc, txt,
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=plt.gca().transAxes,
                     color='k', fontsize=10)

            # also plot entries
            plt.sca(axes2[j+1,i])
            m2,cs2,cb2= pp.createmap(entries,om.lats,om.lons,region=region,
                                     cmapname='jet',linear=True, colorbar=False,
                                     vmin=vmin2,vmax=vmax2)

    # Add row labels
    rows = ['%d days'%fdays for fdays in [0,1,2,4,8]]
    rows[0]='fire filter\n'+rows[0]
    for ax, ax2, row in zip(axes[:,0], axes2[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='small')

    ticks=[np.logspace(np.log10(vmin),np.log10(vmax),5),np.linspace(vmin,vmax,5)][linear]
    pp.add_colourbar(f,cs,ticks=ticks,label='molec/cm$^2$')
    pp.add_colourbar(f2,cs2,ticks=np.linspace(vmin2,vmax2,5),label='pixels')

    f.savefig(pname)
    plt.close(f)
    print("Saved ",pname)
    f2.savefig(pname2)
    plt.close(f2)
    print("Saved ",pname2)

    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to run plot_VCC_rsc_gc_pp()"%elapsed)

    #createmap(data, lats, lons, make_edges=False, GC_shift=True,
    #          vmin=None, vmax=None, latlon=True,
    #          region=__GLOBALREGION__, aus=False, linear=False,
    #          clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
    #          cbarorient='bottom',
    #          pname=None,title=None,suptitle=None, smoothed=False,
    #          cmapname=None, fillcontinents=None):

def plot_VCC_rsc_gc_pp(month=datetime(2005,3,1),region=[-45, 99.5, -11, 160.0]):
    '''
        Plot columns with different amf bases
        Differences and Correlations
              |  VCC   |  VCC_gc   |  VCC_pp
       abs dif|
       rel dif|
       distrs |

    '''

    # start by reading all the VCC stuff
    # useful strings
    ymstr=month.strftime('%Y%m')
    d0=datetime(month.year,month.month,1)
    dN=util.last_day(month)
    pname='Figs/VCC_gc_pp_%s.png'%ymstr
    linear=True # linear colour scale?

    start_time=timeit.default_timer()
    # read in omhchorp
    om=omrp(d0,dayn=dN, keylist=['VCC_OMI','VCC_GC','VCC_PP','gridentries','ppentries'])
    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to read omhchorp"%elapsed)

    start_time2=timeit.default_timer()
    # Subset the data to our region
    subsets=util.lat_lon_subset(om.lats,om.lons,region,[om.VCC_OMI,om.VCC_GC,om.VCC_PP,om.gridentries,om.ppentries],has_time_dim=True)
    lats,lons=subsets['lats'],subsets['lons']
    VCC_OM,VCC_GC,VCC_PP,pix,pix_pp = subsets['data']

    elapsed = timeit.default_timer() - start_time2
    print("TIMEIT: Took %6.2f seconds to subset the VCC arrays"%elapsed)

    oceanmask=util.oceanmask(lats,lons)

    # firemask is 3dimensional: [days,lats,lons]
    fstart=timeit.default_timer()
    firemask,fdates,flats,flons=fio.make_fire_mask(d0,dN=dN,region=region) # use defaults
    smokemask,sdates,slats,slons=fio.make_smoke_mask(d0,dN=dN,region=region)
    anthromask,adates,alats,alons=fio.make_anthro_mask(d0,dN,region=region)
    fullmask=firemask+smokemask+anthromask
    felapsed = timeit.default_timer() - fstart
    print ("TIMEIT: Took %6.2f seconds to get fire,smoke, and anthro masks"%(felapsed))

    # Plot rows,cols,size:
    f=plt.figure(figsize=[18,18])

    # first line is maps of VCC_OMI, VCC_GC, VCC_PP
    titles=[[Ooc,Ogc+" (S(z) updated)",Opc+" (S(z)+$\omega$(z) updated)"],
            [Ogc+'-'+Ooc,Opc+"-"+Ogc,Ooc+"-"+Opc],
            ['','','']
            ]
    maps = [[VCC_OM, VCC_GC, VCC_PP], # orig
            [VCC_GC-VCC_OM, VCC_PP-VCC_GC, VCC_OM-VCC_PP], # abs diff
            [100*(VCC_GC-VCC_OM)/VCC_OM, 100*(VCC_PP-VCC_GC)/VCC_GC, 100*(VCC_OM-VCC_PP)/VCC_PP] # rel diff
            ]
    vmins,vmaxs=[4e15,None,-120],[9e15,None,120] # min,max for colourbars
    cmapnames=['plasma','plasma','seismic']
    cbarlabels=['molec/cm2','molec/cm2','%']
    area_list=[]
    ts_list=[]
    for j in range(3): #col
        for i in range(3): #row
            plt.subplot(5,3,i*3+j+1)

            arr = maps[i][j]
            arr[fullmask]=np.NaN # Remove fire,smoke,anthro...
            arr[:,oceanmask]=np.NaN # nanify the ocean
            if i==0:
                ts_list.append(np.nanmean(arr,axis=(1,2)))
            arr=np.nanmean(arr,axis=0) # average over time
            if i == 0:
                area_list.append(arr) # save each map for distribution plot
            m,cs,cb= pp.createmap(arr,lats,lons,linear=True,
                                  region=region,
                                  vmin=vmins[i],vmax=vmaxs[i],
                                  cmapname=cmapnames[i],
                                  colorbar=j==1, # colorbar in middle column only
                                  clabel=cbarlabels[i])
            plt.title(titles[i][j])

            # add a little thing showing entries and mean and max
            # entries for normal or ppamf
            entries=np.copy([om.gridentries,om.ppentries][j==2])
            entries=entries.astype(np.float) # so I can Nan the ocean/non-aus areas
            entries=np.nansum(entries,axis=0) # how many entries
            entries[oceanmask]=np.NaN

            txt=['N($\mu$)=%d(%.1f)'%(np.nansum(entries),np.nanmean(entries)), '$\mu$ = %.2e'%np.nanmean(arr), 'max = %.2e'%np.nanmax(arr)]
            for txt, yloc in zip(txt,[0.01,0.07,0.13]):
                plt.text(0.01, yloc, txt,
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=plt.gca().transAxes,
                     color='k', fontsize=10)

    # Add time series for each VCC
    plt.subplot(5,1,4)
    labels=[Ooc,Ogc,Opc]
    plt.plot(ts_list,label=labels)

    # Finally add density plots for each map
    plt.subplot(5,1,5)

    area_list= np.transpose([vec for vec in area_list]) # list of np arrays to array of vectors..
    plt.hist(area_list,bins=np.linspace(vmins[0], vmaxs[0], 20), label=labels)

    #
    #ticks=[np.logspace(np.log10(vmin),np.log10(vmax),5),np.linspace(vmin,vmax,5)][linear]
    #pp.add_colourbar(f,cs,ticks=ticks,label='molec/cm$^2$')
    #pp.add_colourbar(f2,cs2,ticks=np.linspace(vmin2,vmax2,5),label='pixels')

    f.savefig(pname)
    plt.close(f)
    print("Saved ",pname)


    elapsed = timeit.default_timer() - start_time
    print("TIMEIT: Took %6.2f seconds to run plot_VCC_rsc_gc_pp()"%elapsed)

    #createmap(data, lats, lons, make_edges=False, GC_shift=True,
    #          vmin=None, vmax=None, latlon=True,
    #          region=__GLOBALREGION__, aus=False, linear=False,
    #          clabel=None, colorbar=True, cbarfmt=None, cbarxtickrot=None,
    #          cbarorient='bottom',
    #          pname=None,title=None,suptitle=None, smoothed=False,
    #          cmapname=None, fillcontinents=None):

def CompareMaps(day=datetime(2005,1,1), oneday=False, ausonly=True, PP=False):
    '''
    '''
    #useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,oneday=oneday)

    lats=om.latitude
    lons=om.longitude
    unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)

    # the datasets with nans and land or ocean masked
    OMP = [om.VC_OMI_RSC, om.VCC] # OMI, My, Paul palmer
    OMP_str = ['OMI_RSC','VCC']
    if PP:
        OMP.append(om.VCC_PP)
        OMP_str.append('VCC_PP')

    # Plot the maps:
    #
    f=plt.figure(figsize=(16,13))
    lllat=-65; urlat=65; lllon=-175; urlon=175
    if ausonly:
        lllat=-50; urlat=-5; lllon=100; urlon=170

    for ii in range([2,3][PP]):
        arr=OMP[ii]
        # Maps
        spnum=[221,231][PP]
        plt.subplot(spnum+ii)
        m,cs,cb = pp.createmap(OMP[ii],mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
        plt.title(OMP_str[ii],fontsize=20)

        # Difference maps
        diff=(OMP[ii]-OMP[ii-1]) * 100.0 / OMP[ii-1]
        spnum=[223,234][PP]
        plt.subplot(spnum+ii)
        m,cs,cb = pp.linearmap(diff,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=-200,vmax=200)
        plt.title('(%s - %s)*100/%s'%(OMP_str[ii],OMP_str[ii-1],OMP_str[ii-1]),fontsize=20)
    plt.suptitle("Maps of HCHO on %s"%ymdstr)

    ausstr=['','_AUS'][ausonly]
    eightstr=['_8day',''][oneday]
    pname='Figs/maps%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)


def test_calculation_corellation(day=datetime(2005,1,1), oneday=False, aus_only=False):
    '''
    Look closely at AMFs over Australia, specifically over land
    and see how our values compare against the model and OMI swaths.
    '''
    # useful strings
    ymdstr=day.strftime('%Y%m%d')

    # read in omhchorp
    om=omrp(day,oneday=oneday)
    VCC=om.VCC_GC
    VCC_PP=om.VCC_PP
    VC_OMI_RSC=om.VC_OMI_RSC
    VC_OMI=om.VC_OMI

    lats=om.latitude
    lons=om.longitude
    unc = om.col_uncertainty_OMI
    mlons,mlats=np.meshgrid(lons,lats)

    if aus_only:
        # filter to just Australia rectangle [.,.,.,.]
        landinds=om.inds_aus(maskocean=True)
    else:
        landinds=om.inds_subset(maskocean=True)
    oceaninds=om.inds_subset(maskocean=False,maskland=True)

    # the datasets with nans and land or ocean masked
    vcomi_l     = ma(VC_OMI,mask=~landinds)
    vcomic_l      = ma(VC_OMI_RSC,mask=~landinds)
    vcc_l       = ma(VCC,mask=~landinds)
    vcc_pp_l    = ma(VCC_PP, mask=~landinds)
    vcomi_o     = ma(VC_OMI,mask=~oceaninds)
    vcomic_o      = ma(VC_OMI_RSC,mask=~oceaninds)
    vcc_o       = ma(VCC,mask=~oceaninds)
    vcc_pp_o    = ma(VCC_PP, mask=~oceaninds)
    landunc     = ma(unc,mask=~landinds)

    # Print the land and ocean averages for each product
    print("%s land averages (oceans are global):"%(['Global','Australian'][aus_only]))
    print("%25s   land,   ocean"%'')
    for arrl,arro,arrstr in zip([vcomi_l, vcomic_l, vcc_l, vcc_pp_l],[vcomi_o,vcomic_o,vcc_o, vcc_pp_o],['OMI','OMI_RSC','OMI_GCC', 'VCC_PP']):
        print("%21s: %5.3e,  %5.3e "%(arrstr, np.nanmean(arrl),np.nanmean(arro)))

    f=plt.figure(figsize=(14,10))
    # Plot the histogram of VC entries land and sea
    land_data=np.transpose([VC_OMI_RSC[landinds],VCC[landinds]])
    ocean_data=np.transpose([VC_OMI_RSC[oceaninds],VCC[oceaninds]])
    olabel=['ocean '+thing for thing in [Ooc,'VCC']]
    llabel=['land ' +thing for thing in [Ooc,'VCC']]
    plt.hist(ocean_data, bins=np.logspace(13, 17, 20), color=['darkblue','lightblue'], label=olabel)
    plt.hist(land_data, bins=np.logspace(13, 17, 20), color=['orange','yellow'], label=llabel)
    plt.xscale("log")
    plt.yscale('log',nonposy='clip') # logarithmic y scale handling zero
    plt.title('Vertical column distributions ($\Omega_{HCHO}$)',fontsize=26)
    plt.ylabel('frequency'); plt.xlabel('molec cm$^{-2}$')
    plt.legend(loc='center left')
    ta=plt.gca().transAxes
    plt.text(0.05,.95, 'land count=%d'%np.sum(landinds),transform=ta)
    plt.text(.05,.90, 'ocean count=%d'%np.sum(oceaninds),transform=ta)
    plt.text(.05,.86, '%s mean(land)=%5.3e'%('VCC',np.nanmean(vcc_l)),transform=ta)
    plt.text(.05,.82, '%s mean(land)=%5.3e'%(Ooc,np.nanmean(vcomic_l)),transform=ta)
    ausstr=['','_AUS'][aus_only]
    eightstr=['_8day',''][oneday]
    pname='Figs/land_VC_hist%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    plt.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

    # Plot the maps:
    #

    f=plt.figure(figsize=(16,13))
    plt.subplot(231)
    lllat=-65; urlat=65; lllon=-175; urlon=175
    if aus_only:
        lllat=-50; urlat=-5; lllon=100; urlon=170

    # OMI_RSC map
    m,cs,cb = pp.createmap(vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title(Ooc,fontsize=20)

    # VCC map
    plt.subplot(232)
    m,cs,cb = pp.createmap(vcc_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    plt.title('VCC',fontsize=20)

    # (VCC- OMI_RSC)/OMI_RSC*100 map
    plt.subplot(233)
    m,cs,cb = pp.linearmap((vcc_l-vcomic_l)*100/vcomic_l,mlats,mlons,lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon, vmin=-200,vmax=200)
    plt.title('(%s - %s)*100/%s'%('VCC',Ooc,Ooc),fontsize=20)

    # show corellations for OMI_RSC- VCC
    nans=np.isnan(vcc_l) + np.isnan(vcomic_l) + vcomic_l.mask # all the nans we won't fit
    plt.subplot(223)
    plt.scatter(vcomic_o, vcc_o, color='blue', label="Ocean", alpha=0.4)
    plt.scatter(vcomic_l, vcc_l, color='gold', label="Land")
    m,x0,r,ci1,ci2 = RMA(VC_OMI_RSC[~nans], VCC[~nans])
    X=np.array([1e10,5e16])
    plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%.5f, r=%.5f'%(m,r))
    plt.plot( X, X, 'k--', label='1-1' )
    plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
    plt.ylabel('VCC'); plt.xlabel(Ooc)

    # show corellations for OMI - VCC
    plt.subplot(224)
    nans=np.isnan(vcc_l) + np.isnan(vcomi_l) + vcc_l.mask # all the nans we won't fit
    plt.scatter(vcomi_o, vcc_o, color='blue', label="Ocean", alpha=0.4)
    plt.scatter(vcomi_l, vcc_l, color='gold', label="Land")
    m,x0,r,ci1,ci2 = RMA(vcomi_l[~nans],vcc_l[~nans])
    X=np.array([1e10,5e16])
    plt.plot( X, m*X+x0,color='fuchsia',
            label='Land: m=%4.2f, r=%.2f'%(m,r))
    plt.plot( X, X, 'k--', label='1-1' )
    plt.legend(loc=2,scatterpoints=1, fontsize=10,frameon=False)
    plt.ylabel('VCC'); plt.xlabel(Oomi)

    # save plot
    pname='Figs/correlations%s%s_%s.png'%(eightstr,ausstr,ymdstr)
    f.suptitle("Product comparison for %s"%ymdstr,fontsize=28)
    f.savefig(pname)
    print("%s saved"%pname)
    plt.close(f)

def test_amf_calculation(scount=50):
    '''
    Grab input argument(=50) columns and the apriori and omega values and check the AMF calculation on them
    Plot old, new AMF and integrand at each column
    Also check the AMF calculated using non sigma normalised shape factor
    '''
    day=datetime(2005,1,1)
    # Read OMHCHO data ( using reprocess get_good_pixels function )
    pixels=reprocess.get_good_pixel_list(day, getExtras=True)
    N = len(pixels['lat']) # how many pixels do we have

    # read gchcho
    gchcho = fio.read_gchcho(day)

    # check the random columns
    AMF_old=[]
    AMF_z, AMF_s=[],[]

    # find 50 GOOD columns(non nans)
    for i, jj in enumerate(random.sample(range(N), scount)):
        lat,lon=pixels['lat'][jj],pixels['lon'][jj]
        omega=pixels['omega'][:,jj]
        AMF_G=pixels['AMF_G'][jj]
        AMF_old.append(pixels['AMF_OMI'][jj])
        w_pmids=pixels['omega_pmids'][:,jj]
        # rerun the AMF calculation and plot the shampoo
        innerplot='Figs/AMF_test/AMF_test_innerplot%d.png'%i
        AMFS,AMFZ = gchcho.calculate_AMF(omega, w_pmids, AMF_G, lat, lon, plotname=innerplot,debug_levels=True)
        AMF_s.append(AMFS)
        AMF_z.append(AMFZ)
    #print( "AMF_s=", AMF_s[:] )
    #print( "AMF_z=", AMF_z[:] )
    #print( "AMF_o=", AMF_old[:] )

    # Also make a plot of the regression new vs old AMFs
    f=plt.figure(figsize=(12,12))
    amfs=np.array(pixels['AMF_GC'])
    amfo=np.array(pixels['AMF_OMI'])
    plt.scatter(amfs, amfo, color='k', label='pixel AMFs')
    # line of best fit
    isnans= np.isnan(amfs) + np.isnan(amfo)
    slp,intrcpt,r,p,sterr=stats.linregress(amfs[~isnans],amfo[~isnans])
    # straight line on log-log plot is 10^(slope*log(X) + Yintercept)
    Xarr=np.array([1,6])
    plt.plot(Xarr,slp*Xarr+intrcpt, color='red', label='slope=%.5f, r=%.5f'%(slp,r))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('Figs/AMF_test/AMF_test_corr.png')
    plt.close(f)

    # Post regridding, check the AMFs within 50 degrees of the equator, and color by land/sea
    # Achieve this using maskoceans
    #
    f=plt.figure(figsize=(10,10))
    omhchorp=fio.read_omhchorp(day,oneday=True, keylist=['AMF_GC','AMF_OMI','latitude','longitude'])
    lats=omhchorp['latitude']
    lons=omhchorp['longitude']
    mlons,mlats=np.meshgrid(lons,lats)  # [lats x lons] ?
    amf=omhchorp['AMF_GC']      # [lats x lons]
    amfo=omhchorp['AMF_OMI']    # [lats x lons]
    ocean=maskoceans(mlons,mlats,amf,inlands=False).mask
    landamfo=amfo.copy()
    landmlats=mlats.copy()
    landmlons=mlons.copy()
    landamf=amf.copy()
    oceanamfo=amfo.copy()
    oceanmlats=mlats.copy()
    oceanmlons=mlons.copy()
    oceanamf=amf.copy()
    for arrr in [landamf, landamfo, landmlats, landmlons]:
        arrr[ocean]=np.NaN
    for arrr in [oceanamf, oceanamfo, oceanmlats, oceanmlons]:
        arrr[~ocean]=np.NaN
    m,cs,cb=pp.linearmap(landmlats,landmlons,landamfo)
    f.savefig('Figs/oceancheck.png')
    plt.close(f)
    # Check slopes and regressions of ocean/non ocean AMFs
    f=plt.figure(figsize=(14,14))
    plt.scatter(amf[ocean], amfo[ocean], color='cyan', alpha=0.5)
    plt.scatter(amf[~ocean],amfo[~ocean], color='fuchsia', alpha=0.5)
    slopeo,intercepto,ro,p,sterr = stats.linregress(amf[ocean], amfo[ocean])
    slopel,interceptl,rl,p,sterr = stats.linregress(amf[~ocean],amfo[~ocean])
    plt.plot(Xarr, slopel*Xarr+interceptl,color='fuchsia',
            label='Land: slope=%.5f, r=%.5f'%(slopel,rl))
    plt.plot(Xarr, slopeo*Xarr+intercepto, color='cyan',
            label='Ocean: slope=%.5f, r=%.5f'%(slopeo,ro))
    plt.xlabel('AMF_GC')
    plt.ylabel('AMF_OMI')
    plt.legend(loc=0)
    plt.title('AMF correlation')
    f.savefig('Figs/AMF_test/AMF_test_corr_masked.png')

    #amfland=maskoceans(mlons,mlats,amf,inlands=False)
    #amfoland=maskoceans(mlons,mlats,amfo,inlands=False)


def compare_cloudy_map():
    '''
    Check the result of running the 8 day reprocess after filtering cloudy grid
        cells.
    '''
    date=datetime(2005,1,1,0)
    # read the dataset WITHOUT cloud filtering:
    data=fio.read_omhchorp(date)
    counts = data['gridentries']
    lonmids=data['longitude']
    latmids=data['latitude']
    # read the dataset WITH cloud filtering
    # ( at that point I stored regridded vc seperately )
    cloudyfile="omhchorp/cloudy_omhcho_8p0.25x0.31_20050101.he5"
    cloudydata=fio.read_omhchorp(date,filename=cloudyfile)
    cloudycounts=cloudydata['GridEntries']

    # look at differences before and after cloud filter at 0.4 threshhold
    print("%4e entries before cloud filter, %4e entries after cloud filter"%(np.nansum(cloudycounts),np.nansum(counts)))
    print("Cloud filtered stats:")
    print(" MIN,\tMEAN,\tMAX ")
    print("OLD AMF: ")
    mmm(data['AMF_OMI'])
    print("NEW AMF: ")
    mmm(data['AMF_GC'])
    print("OLD_VC:  ")
    mmm(data['VC_OMI'] )
    print("NEW_VC:  ")
    mmm(data['VC_GC'] )
    print("Unfiltered stats:")
    print("OLD AMF: ")
    mmm(cloudydata['AMF_OMI'])
    print("NEW AMF: ")
    mmm(cloudydata['AMF_GC'])
    print("OLD_VC:  ")
    mmm(cloudydata['VC_OMI'] )
    print("NEW_VC:  ")
    mmm(cloudydata['VC_GC'] )

    # Plot OMI, oldrp, newrp VC map
    f, axes = plt.subplots(2,3,num=0,figsize=(16,14))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    #plt.sca(axes[0,0])
    i=0
    for d in [cloudydata, data]:
        plt.sca(axes[i,0])
        m,cs,cb = pp.createmap(d['VC_OMI'],latmids,lonmids)
        plt.title('VC Regridded')
        plt.sca(axes[i,1])
        m,cs,cb = pp.createmap(d['VC_GC'],latmids,lonmids)
        plt.title('VC Reprocessed')
        plt.sca(axes[i,2])
        m,cs,cb = pp.linearmap(d['AMF_GC'],latmids,lonmids,vmin=1.0,vmax=6.0)
        plt.title('AMF reprocessed')
        i=1

    # save plots
    plt.tight_layout()
    plt.savefig("Figs/cloud_filter_effects.png")
    plt.close()





def test_gchcho():
    '''
    Function tests gchcho output file created by create_column.pro
    Saves a plot of HCHO density,
    apriori sigma shape before/after interpolation,
    and total column hcho map
    '''
    ## Files to be read based on date
    day0 = datetime(2005,1,1)
    ymstr= day0.strftime('%Y %b')
    ymdstr=day0.strftime('%Y%m%d')
    print("Testing GCHCHO stuff on "+ymstr)

    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)

    plt.figure(figsize=(14,12))
    m,cs,cb=gchcho.PlotVC()
    plt.savefig('Figs/GCHCHO_Vertical_Columns%s.png'%ymdstr)
    plt.clf()

    plt.figure(figsize=(12,12))
    gchcho.PlotProfile()
    plt.savefig('Figs/GCHCHO_EGProfile%s.png'%ymdstr)
    plt.clf()

def test_hchorp_apriori():
    '''
    Check the apriori saved into reprocessed file looks as it should.
    (Compares omhchorp against gchcho, and omhchorg)
    Currently looks ok(11/4/16 monday)
    UPDATE: 20160906
        omhchorp no longer contains the aprioris and omegas - testing that stuff is done in amf tests.
        now this test just looks at the gchcho vs omhcho aprioris
    '''
    ## Files to be read based on date
    day0 = datetime(2005,1,1)
    ymstr= day0.strftime('%Y %b')
    #yyyymm= day0.strftime('%Y%m')
    print("Testing GCHCHO apriori stuff on "+ymstr)

    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)

    # Read omhcho day of swaths:
    omhcho = fio.read_omhcho_day(day0) # [ plevs, ~24k rows, 60 cols ]
    omishape = omhcho['apriori']
    omilats, omilons = omhcho['lats'],omhcho['lons']
    omiplevs=omhcho['plevels']

    # Sample of normalized columns
    scount=15 # expect around 7 columns which aren't on nan pixels
    ii=random.sample(range(len(omilats[:,0])), scount)
    jj=random.sample(range(len(omilats[0,:])), scount)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,9))
    for i in range(scount):
        # indices of random lat/lon pair
        rowi,coli=ii[i],jj[i]
        lat=omilats[rowi,coli]
        if np.isnan(lat):
            continue
        lon=omilons[rowi,coli]
        omiS=omishape[:,rowi,coli]
        omiP=omiplevs[:,rowi,coli]
        ax1.plot(omiS,omiP, label="%-7.2f,%-7.2f"%(lat,lon))
        gcShape,gcSig=gchcho.get_single_apriori(lat,lon,z=False)
        ax2.plot(gcShape,gcSig)
    ax1.set_xlabel('Molecules/cm2')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title('OMI apriori')
    ax1.set_ylim([1050, 0.05])
    ax1.set_yscale('log')
    ax2.set_ylim([1.04,-0.04])
    ax2.set_ylabel('Sigma')
    ax2.set_title('GEOS-Chem S$_\sigma$')
    ax2.set_xlabel('unitless')
    ax1.legend(title='    lat,  lon ',loc=0)
    pltname='Figs/Shape_Factor_Examples.png'
    plt.savefig(pltname); print("%s saved!"%pltname)

def check_high_amfs(day=datetime(2005,1,1)):
    '''
    Get good pixels, look at AMFs over 10, examine other properties of pixel
    '''
    print("Running check_high_amfs")

    start_time=timeit.default_timer()
    pix=reprocess.get_good_pixel_list(day, getExtras=True)
    elapsed = timeit.default_timer() - start_time
    print ("Took %6.2f seconds to read %d entries"%(elapsed, len(pix['AMF_OMI'])))

    cloud=np.array(pix['cloudfrac'])    # [~1e6]
    lat=np.array(pix['lat'])            #
    # inds2 = high amf, cloud < 0.4
    inds2= (np.array(pix['AMF_GC']) > 10.0) * (cloud < 0.4)
    # inds1 = non high amf, cloud < 0.4
    inds1= (~inds2) * (cloud < 0.4)
    omegas=pix['omega']         # [47, ~1e6]
    aprioris=pix['apriori']     #
    sigmas=pix['sigma']         #

    assert np.sum(np.abs(pix['qualityflag']))==0, 'qualityflags non zero!'
    assert np.sum(np.abs(pix['xtrackflag']))==0, 'xtrackflags non zero!'

    # normal AMFS
    omegas1=omegas[:,inds1]
    n1=len(omegas1[0,:])
    sigmas1=pix['sigma'][:,inds1]
    # cloud seems fine, check relation to latitude
    cloud1=cloud[inds1]
    lat1=lat[inds1]
    # high AMFS
    omegas2=omegas[:,inds2]
    n2=len(omegas2[0,:])
    sigmas2=pix['sigma'][:,inds2]
    cloud2=cloud[inds2]
    lat2=lat[inds2]
    f = plt.figure(figsize=(15,9))
    a1 = f.add_subplot(1, 3, 1)
    a2 = f.add_subplot(1, 3, 2, sharex = a1, sharey=a1)
    a3 = f.add_subplot(2, 3, 3)
    a4 = f.add_subplot(2, 3, 6, sharex = a3)
    plt.sca(a1)
    # plot random set of 100 normal omegas
    for i in random.sample(range(n1),100):
        plt.plot(omegas1[:,i],sigmas1[:,i],alpha=0.4,color='orange')
    # plot mean omega
    X,Y = np.mean(omegas1, axis=1), np.mean(sigmas1,axis=1)
    Xl,Xr = X-2*np.std(omegas1,axis=1),X+2*np.std(omegas1,axis=1)
    plt.plot(X,Y, linewidth=3, color='k')
    plt.fill_betweenx(Y, Xl, Xr, alpha=0.5, color='cyan')
    plt.text(4.75,.9, 'mean cloud =%6.2f%%'%(np.mean(cloud1)*100.))
    plt.text(4.75,.86, 'count = %d'%n1)
    plt.text(6, 0.12, 'Mean profile',fontweight='bold')
    plt.text(6, 0.16, '$\pm$2 std devs', color='cyan')
    plt.text(6, 0.20, '100 random $\omega$ profiles', color='orange')
    plt.title('AMF <= 10.0')
    plt.ylabel('Altitude ($\sigma$)')
    plt.sca(a2)
    #plot 100 highamf omegas
    for i in random.sample(range(n2),100):
        plt.plot(omegas2[:,i],sigmas2[:,i],alpha=0.4,color='orange')
    # plot mean omega
    X,Y = np.mean(omegas2, axis=1), np.mean(sigmas2,axis=1)
    Xl,Xr = X-2*np.std(omegas2,axis=1),X+2*np.std(omegas2,axis=1)
    plt.plot(X,Y, linewidth=3, color='k')
    plt.fill_betweenx(Y, Xl, Xr, alpha=0.5, color='cyan')
    plt.text(5.75,.9, 'mean cloud =%6.2f%%'%(np.mean(cloud2)*100.))
    plt.text(5.75,.86, 'count = %d'%n2)
    plt.title('AMF > 10.0')
    plt.ylim([1.05,-0.05])
    plt.xlim([-0.1, np.max(omegas2)])
    # plot cloud frac histogram
    plt.sca(a3)
    plt.hist(lat1)
    plt.title('histogram: AMF < 10')
    plt.sca(a4)
    plt.hist(lat2, bins=np.arange(-90,91,5))
    plt.title('histogram: AMF > 10')
    plt.setp(a3, xticks=np.arange(-90,91,30)) # set xticks for histograms
    plt.xlabel('Latitude')
    plt.suptitle('Scattering weights ($\omega$)')
    outpic='Figs/high_AMF_omegas.png'
    plt.savefig(outpic)
    print('saved %s'%outpic)



def check_flags_and_entries(day=datetime(2005,1,1), oneday=True):
    '''
    Count entries and how many entries were filtered out
    Plot histograms of some fields for one or 8 day averaged omhchorp data
    '''

    print("Running check_flags_and_entries")
    # Read in one day average
    omhchorp=fio.read_omhchorp(day,oneday=oneday)
    yyyymmdd=day.strftime("%Y%m%d")
    rsc=omhchorp['RSC']
    rsc_gc=omhchorp['RSC_GC'] # RSC_GC is in molecs/cm2 as of 11/08/2016
    rsc_region=omhchorp['RSC_region']
    print(("RSC region: ", rsc_region))
    VC_GC=omhchorp['VC_GC']
    VC_OMI=omhchorp['VC_OMI']
    VCC=omhchorp['VCC']
    entries=omhchorp['gridentries']

    # show histograms for positive and negative part of data
    # pass in two axes
    def show_distribution(axes, bins, data, log=True):
        notnans=~np.isnan(data)
        fdata=data[notnans]
        plus=fdata[fdata>0]
        minus=-1*fdata[fdata<0]
        plt.sca(axes[0])
        plt.hist(minus,bins=bins)
        if log: plt.xscale("log")
        axes[0].invert_xaxis()
        plt.sca(axes[1])
        plt.hist(plus,bins=bins)
        if log: plt.xscale("log")

    f, axes = plt.subplots(2,2)
    # Logarithmic space from 1e12 - 1e19, 50 steps
    logbins=np.logspace(12,19,50)

    show_distribution(axes[0,:], logbins, VC_GC)
    axes[0,0].set_title("Negative GC_VC_HCHO")
    axes[0,1].set_title('Positive GC_VC_HCHO')

    show_distribution(axes[1,:], logbins, VC_OMI)
    axes[1,0].set_title("Negative OMI_VC_HCHO")
    axes[1,1].set_title('Positive OMI_VC_HCHO')

    plt.suptitle("VC distributions for %d day average on %s"%([8,1][oneday],yyyymmdd), fontsize=25)
    plt.savefig("Figs/distributions%d_%s"%([8,1][oneday], yyyymmdd))
    plt.close()

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running tests.py")
    pp.InitMatplotlib()

    date=datetime(2005,1,1)
    d0=date
    de=datetime(2007,12,31)
    #Summary_Single_Profile()

    #####################
    ### E_new tests
    #####################

    # Plot E_new and stdev of E_new (all 3 types)
    #test_E_new.Summary_E_new() # Last run

    # check VCC output
    #test_E_new.VCC_check()

    #compare difference between VCCs
    #test_E_new.VCC_comparison()

    #####################
    ### Files tests
    #####################
    # make sure units are as expected...
    #check_files.write_GC_units() # last run: 25/5/18
    # are fires read in and interpolated OK?
    #check_files.test_fires_fio() #


    #####################
    ### RSC TESTS
    #####################
    ## original omi VCC vs same with new RSC
    #RSC_tests.new_vs_old(date)  # last run 1/6/18

    ## Plots showing how it works
    #RSC_tests.summary(date) # Last run 31/5/18 # last run 18/5/18 - needs work
    #RSC_tests.check_RSC(date) # Last run 4/6/18

    ## Look at different ways of making the RSC (different AMFs)
    #RSC_tests.intercomparison(date) # last run 4/6/18


    #####################
    ### FILTERS TESTS
    #####################
    ## Determine affect of NO filter on OMNO2d to see if it's working right:
    #test_filters.check_no2_filter(year=datetime(2005,1,1))# Run 23/5/18
    ## Look at whether fire filter affects hcho vs temp correlation
    #[test_filters.HCHO_vs_temp_vs_fire(d0=datetime(2005,1,1),d1=datetime(2005,2,28),subset=v) for v in [0,1,2]] # Run over three subsets

    # Test how many pixels are lost to filtering...
    ## Also show affect on regions            # Run 23/7/18 (TODO: Update to add table of data)
    #test_filters.test_mask_effects(datetime(2005,1,1),datetime(2006,1,1))
    ## How mnay pixels are filtered out?     # run 23/7/18
    test_filters.show_mask_filtering()
    ## look at hcho vs fire
    #test_filters.HCHO_vs_temp_vs_fire()

    ######
    ### Tests to be sorted into files
    ######
    #Test_Uncertainty()              # last run 15/5/18
    #check_products()               # last run 15/5/18

    # GEOS Chem trop vs ucx restarts
    #check_HEMCO_restarts()

    #plot_VCC_rsc_gc_pp()
    #analyse_VCC_pp(oneday=False)


    #for dates in [ [datetime(2005,1,1),datetime(2005,1,8)],
    #               [datetime(2005,2,1),datetime(2005,2,28)],
    #               [datetime(2006,4,1),datetime(2006,4,14)]]:
    #    smoke_vs_fire(dates[0],dates[1])

    #plot_VCC_rsc_gc_pp_fireaffects(d0=datetime(2005,3,1),dn=datetime(2005,3,31))
    #plot_swaths()
    # AMF tests and correlations
    #Check_OMI_AMF()
    #Check_AMF()
    #Check_AMF_relevelling()
    #test_amf_calculation(scount=5)
    #for aus_only in [True, False]:
    #    test_calculation_corellation(day=datetime(2005,1,1), oneday=False, aus_only=aus_only)

    #compare_GC_OMI_new()

    #check_flags_and_entries() # check how many entries are filtered etc...
    # check some days (or one or no days)
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [0, 8, 16, 24, 32, 112] ]
    #dates=[ datetime(2005,1,1) + timedelta(days=d) for d in [112] ]
    #dates=[ datetime(2005,1,1) ]

    #dates=[ ]

    #CompareMaps(day=dates[0],oneday=False,ausonly=False)
    #for day in dates:
        #test_reprocess_corrected(date=day, oneday=oneday)
        #test_reprocess_corrected(date=day, oneday=oneday, lllat=-50,lllon=100,urlat=-10,urlon=170, pltname="zoomed")
        #compare_products(date=day,oneday=True)
        #compare_products(date=day,oneday=True,positiveonly=True,pltname="positive")

        #compare_products(date=day,oneday=True, lllat=-50,lllon=100,urlat=-10,urlon=170, pltname="zoomed")
    # to be updated:
    #check_timeline()
    #reprocessed_amf_correlations()

    # other tests
    #check_high_amfs()
    #test_hchorp_apriori()
    #test_gchcho()

    # Check that cloud filter is doing as expected using old output without the cloud filter
    #compare_cloudy_map()

    # check the ref sector correction is not weird.
    #check_RSC(track_corrections=True)
