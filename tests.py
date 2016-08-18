'''
File to test various parts of fio.py and reprocess.py
'''
## Modules
# these 2 lines make plots not show up ( can save them as output faster )
# use needs to be called before anythin tries to import matplotlib modules
import matplotlib
matplotlib.use('Agg')

# my file reading and writing module
import fio
import reprocess

import numpy as np
from scipy.interpolate import interp1d

from datetime import datetime

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # for lognormal colour bar
#import matplotlib.patches as mpatches

import timeit
import random

##############################
########## FUNCTIONS #########
##############################

def regularbounds(x,fix=False):
    newx=np.zeros(len(x)+1)
    xres=x[1]-x[0]
    newx[0:-1]=np.array(x) - xres/2.0
    newx[-1]=newx[-2]+xres
    # If the ends are outside 90N/S or 180E/W then bring them back
    if fix:
        if newx[-1] >= 90: newx[-1]=89.99
        if newx[0] <= -90: newx[0]=-89.99
        if newx[-1] >= 180: newx[-1]=179.99
        if newx[0] <= -180: newx[0]=-179.99
    return newx

def createmap(data,lats,lons, vmin=5e13, vmax=1e17, 
              llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-179, urcrnrlon=179):
    m=Basemap(llcrnrlat=llcrnrlat,  urcrnrlat=urcrnrlat,
          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
          resolution='i',projection='merc')
    if len(lats.shape) == 1:
        latsnew=regularbounds(lats)
        lonsnew=regularbounds(lons)
    else:
        latsnew,lonsnew=(lats,lons)
    cs=m.pcolormesh(lonsnew, latsnew, data, latlon=True, 
                vmin=vmin,vmax=vmax,norm = LogNorm(), clim=(vmin,vmax))
    cs.cmap.set_under('white')
    cs.cmap.set_over('pink')
    cs.set_clim(vmin,vmax)
    
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    cb.set_label('Molecules/cm2')
    
    return m, cs, cb
def globalmap(data,lats,lons):
    return createmap(data,lats,lons,-80,80,-179,179)
def ausmap(data,lats,lons):
    return createmap(data,lats,lons,-52,-5,100,160)
def linearmap(data,lats,lons,vmin=None,vmax=None,
              llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-179, urcrnrlon=179):
    
    m=Basemap(llcrnrlat=llcrnrlat,  urcrnrlat=urcrnrlat,
          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
          resolution='l',projection='merc')
    
    if len(lats.shape) == 1:
        latsnew=regularbounds(lats)
        lonsnew=regularbounds(lons)
        lonsnew,latsnew=np.meshgrid(lonsnew,latsnew)
    else:
        latsnew,lonsnew=(lats,lons)
    cs=m.pcolormesh(lonsnew, latsnew, data, latlon=True, vmin=vmin, vmax=vmax)
    if vmin is not None:
        cs.cmap.set_under('white')
        cs.cmap.set_over('pink')
        cs.set_clim(vmin,vmax)
    cb=m.colorbar(cs,"bottom",size="5%", pad="2%")
    m.drawcoastlines()
    m.drawparallels([0],labels=[0,0,0,0]) # draw equator, no label
    return m, cs, cb

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

def test_reprocess_corrected(run_reprocess=False, oneday=True):
    '''
    Run and time reprocess method
    Plot some of the outputs
    '''
    date=datetime(2005,1,1)
    
    # Run the reprocessing of several days
    if run_reprocess:
        start_time = timeit.default_timer()
        print ( "reprocess.Reprocess_N_days being called now" )
        reprocess.Reprocess_N_days(date, plot_tracks=True) # defaults to 8 days, 8 processors
        elapsed = timeit.default_timer() - start_time
        print ("Took " + str(elapsed/60.0)+ " minutes to reprocess eight days")
    
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
    # VCC, VC_gc, AMF_gc
    # GEOS-Chem month average map
    #
    f = plt.figure(num=0,figsize=(16,16))
    axes=[f.add_subplot(3,3,j) for j in range(1,10)]
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0])
    m,cs,cb = createmap(omhchorp['SC'],lats,lons)
    plt.title('SC')
    plt.sca(axes[1])
    m,cs,cb = createmap(omhchorp['VC_OMI'],lats,lons)
    plt.title('VC OMI ($\Omega_{OMI}$)')
    plt.sca(axes[2])
    m,cs,cb = linearmap(omhchorp['AMF_OMI'],lats,lons,vmin=1.0,vmax=6.0)
    plt.title('AMF OMI')
    plt.sca(axes[3])
    m,cs,cb = createmap(omhchorp['VCC'],lats,lons)
    plt.title('VCC')
    plt.sca(axes[4])
    m,cs,cb = createmap(omhchorp['VC_GC'],lats,lons)
    plt.title('VC GC')
    plt.sca(axes[5])
    m,cs,cb = linearmap(omhchorp['AMF_GC'],lats,lons,vmin=1.0,vmax=6.0)
    plt.title('AMF_GC')
    
    # Plot finally the GEOS-Chem map divergences
    
    gc=fio.read_gchcho(date) # read the GC data
    fineHCHO=gc.interp_to_grid(latmids,lonmids) * 1e-4 # molecules/m2 -> molecules/cm2
    print(fineHCHO.shape)
    print(lats.shape)
    OMIdiff=100*(omhchorp['VC_OMI'] - fineHCHO) / fineHCHO
    GCdiff=100*(omhchorp['VCC'] - fineHCHO) / fineHCHO
    plt.sca(axes[6])
    vmin,vmax = -150, 150
    m,cs,cb = linearmap(OMIdiff, lats,lons,vmin=vmin,vmax=vmax)
    #m,cs,cb = createmap(gc.VC_HCHO*1e-4, glats,glons, llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-179, urcrnrlon=179)
    plt.title('100($\Omega_{OMI}$-GEOSChem)/GEOSChem')
    plt.sca(axes[7])
    m,cs,cb = linearmap(GCdiff, lats,lons,vmin=vmin,vmax=vmax)
    plt.title('100(VCC-GEOSChem)/GEOSChem')
    plt.sca(axes[8])
    m,cs,cb = createmap(fineHCHO, lats,lons)
    plt.title('GEOS-Chem $\Omega_{HCHO}$')
    
    # save plots
    yyyymmdd = date.strftime("%Y%m%d")
    f.suptitle(yyyymmdd, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    onedaystr= [ 'eight_day_','one_day_' ][oneday]
    outfig="pictures/%scorrected%s.png"%(onedaystr, yyyymmdd)
    plt.savefig(outfig)
    plt.close()
    print(outfig+" Saved.")

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
        m,cs,cb = createmap(d['VC_OMI'],latmids,lonmids)
        plt.title('VC Regridded')
        plt.sca(axes[i,1])
        m,cs,cb = createmap(d['VC_GC'],latmids,lonmids)
        plt.title('VC Reprocessed')
        plt.sca(axes[i,2])
        m,cs,cb = linearmap(d['AMF_GC'],latmids,lonmids,vmin=1.0,vmax=6.0)
        plt.title('AMF reprocessed')
        i=1
        
    # save plots
    plt.tight_layout()
    plt.savefig("pictures/cloud_filter_effects.png")
    plt.close()
    
def test_fires_fio():
    '''
    Test 8 day average fire interpolation and fio
    '''
    day = datetime(2005,1,1)
    ## get normal and interpolated fire
    orig, lats, lons=fio.read_8dayfire(day)
    
    lonres,latres = 0.3125, 0.25
    regr, lats2, lons2=fio.read_8dayfire_interpolated(day, latres=latres, lonres=lonres)

    check_array(orig)
    check_array(regr)
    
    # Number checks..
    assert np.max(orig) == np.max(regr), "Maximum values don't match..."
    print ("Mean orig = %4.2f\nMean regridded = %4.2f"%(np.mean(orig[orig>-1]), np.mean(regr[regr>-1])))
    
    ## PLOTTING
    ##
    
    # EG plot of grids...
    plt.figure(0,figsize=(10,8))
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
    plt.savefig('pictures/AQUAgrids.png')
    plt.close()
    ## Regridded Data plot comparison
    # plot on two subplots
    fig=plt.figure(1, figsize=(14,9))
    fig.patch.set_facecolor('white')
    
    # mesh lon/lats
    mlons,mlats = np.meshgrid(lons,lats)    
    mlons2,mlats2 = np.meshgrid(lons2,lats2)
    axtitles=['original (0.5x0.5)',
              'regridded to %1.4fx%1.4f (latxlon)' % (latres,lonres)]
    
    ax1=plt.subplot(121)
    m1,cs1, cb1 = linearmap(orig,mlats,mlons, vmin=1, vmax=10,
        llcrnrlat=-57, urcrnrlat=1, llcrnrlon=110, urcrnrlon=170)
    ax2=plt.subplot(122)
    m2,cs2, cb2 = linearmap(regr,mlats2,mlons2, vmin=1, vmax=10,
        llcrnrlat=-57, urcrnrlat=1, llcrnrlon=110, urcrnrlon=170)
    
    for i in range(2):
        [ax1,ax2][i].set_title(axtitles[i])
        [cb1, cb2][i].set_label('Fire Count (8 day)')
    
    plt.suptitle('AQUA 2005001',fontsize=20)
    plt.tight_layout()
    plt.savefig('pictures/AQUA2005001.png')
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
    print("Testing GCHCHO stuff on "+ymstr)
    
    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)
    
    # check class structure
    print(vars(gchcho).keys())
    
    print ("Total of a sample of normalized columns")
    ii=random.sample(range(0,91),10)
    jj=random.sample(range(144),10)
    S_s=gchcho.Shape_s
    for i in range(10):
        print ("sum(S_s[:,%d,%d]): %f"%(ii[i],jj[i],np.sum(S_s[:,ii[i],jj[i]])))
        
    plt.figure(figsize=(14,12))
    m,cs,cb=gchcho.PlotVC()
    plt.savefig('pictures/GC_Vertical_Columns.png')
    plt.clf()
    
    plt.figure(figsize=(12,12))
    gchcho.PlotProfile()
    plt.savefig('pictures/GC_Profile.png')
    plt.clf()
    
    # Check that structure's get_apriori function works:
    loncheck, latcheck = 130, -30
    lats, lons = gchcho.lats, gchcho.lons
    xi=np.searchsorted(lons,loncheck)
    yi=np.searchsorted(lats,latcheck)
    z = gchcho.pmids[:,yi,xi]
    new_S, new_lats, new_lons = gchcho.get_apriori(latres=0.25,lonres=0.3125)
    new_xi=np.searchsorted(new_lons,loncheck)
    new_yi=np.searchsorted(new_lats,latcheck)
    print ("sum(new_S[:,%d,%d]): %f"%(new_yi,new_xi, np.sum(new_S[:,new_yi,new_xi])))
    # Old Shape Factor
    plt.figure(figsize=(12,12))
    
    plt.plot(S_s[:,yi,xi], z, label='on 2x2.5 grid', linewidth=3, color='black')
    # set to y log scale
    plt.yscale('log')
    plt.ylim([1e3, 1e-1])
    plt.title('Apriori Shape at lat=%d, lon=%d '%(latcheck,loncheck))
    plt.xlabel('S_s')
    plt.ylabel('hPa')
    
    # new grid shape factor
    plt.plot(new_S[:,new_yi,new_xi], z, label='on .25x.3125 grid', linewidth=2, color='pink')
    plt.legend()
    plt.savefig('pictures/GC_apriori_interpolation.png')
    plt.clf()
    
    return()

def test_hchorp_apriori():
    '''
    Check the apriori saved into reprocessed file looks as it should.
    (Compares omhchorp against gchcho, and omhchorg)
    Currently looks ok(11/4/16 monday)
    '''
    ## Files to be read based on date
    day0 = datetime(2005,1,1)
    ymstr= day0.strftime('%Y %b')
    #yyyymm= day0.strftime('%Y%m')
    print("Testing GCHCHO apriori stuff on "+ymstr)
    
    ## grab shape factors and aprioris from reprocessed file ( one day average for now )
    #
    keylist= ['latitude','longitude','ShapeFactor_GC','ShapeFactor_OMI','Sigma_GC', 'Sigma_OMI']
    omhchorp = fio.read_omhchorp(day0, oneday=True, keylist=keylist)
    lats, lons = omhchorp['latitude'], omhchorp['longitude']
    mlons, mlats = np.meshgrid(lons,lats)
    
    ## CHECK Shape factors from OMI interpolated to the new grid OK
    #
    S_omi = omhchorp['ShapeFactor_OMI'] # [ 720, 1152, 47 ]
    sigma_omi = omhchorp['Sigma_OMI']   # [ 720, 1152, 47 ]
    S_gc  = omhchorp['ShapeFactor_GC']  # [ 72, 720, 1152 ]
    sigma_gc = omhchorp['Sigma_GC']     # [ 72, 720, 1152 ]
    
    # OMHCHORG: (data, lats, lons, count, amf, omega, shape)
    # rg shape = [ 720, 1152, 47 ]
    latres, lonres = 0.25, 0.3125
    rgdata, rglats, rglons, rgcount, rgamf, rgomega, rgshape = fio.read_omhchorg(day0, latres=latres,lonres=lonres)
    
    # grab stuff from gchcho file
    gchcho= fio.read_gchcho(day0)
    sgc_pre, lats_pre, lons_pre, sigma_pre = gchcho.get_apriori() # [ 72, 720, 1152 ]
    
    print ("shape factors for OMI, OMI pre, GEOS-Chem, GEOS-Chem pre:")
    print ([arr.shape for arr in [S_omi, rgshape, S_gc, sgc_pre]])
    
    # Sample of normalized columns
    scount=10
    ii=random.sample(range(-90,91), scount) # lat samples
    jj=random.sample(range(-180,181),scount) # lon samples
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(13,9))
    for i in range(scount):
        # indices of random lat/lon pair
        lati=np.searchsorted(lats,ii[i])
        loni=np.searchsorted(lons,jj[i])
        
        # plot OMI in black
        # GEOS-Chem in purple
        vomi=sigma_omi[lati,loni,:]
        somi=S_omi[lati,loni,:]
        somirg=rgshape[lati,loni,:]
        vgc =sigma_gc[:,lati,loni]
        sgc =S_gc[:,lati,loni]
        sgc_prei = sgc_pre[:,lati,loni]
        ax2.plot(somi, vomi, color='black',  linewidth=2)
        ax1.plot(somirg, vomi, color='brown', linewidth=2)
        ax4.plot(sgc, vgc,  color='purple', linewidth=2)
        ax3.plot(sgc_prei, vgc, color='pink', linewidth=2)
    ax1.set_ylim(1,0)
    ax1.set_ylabel('Sigma')
    ax1.set_xlabel('Molecules/cm2')
    ax1.set_title('OMI S_s (Pre)')
    ax2.set_title('OMI S_s (Post)')
    ax3.set_xlabel('unitless')
    ax3.set_title('GC S_s (Pre)')
    ax4.set_title('GC S_s (Post)')
    #black_patch = mpatches.Patch(color='black', label='S_s[x,y,:]')
    #chart_patch = mpatches.Patch(color='purple', label='S_s[xnew,ynew,:]')
    #plt.legend(handles=[black_patch, chart_patch])
    
    plt.savefig('pictures/Shape_Factor_Examples.png')
    print("Shape_Factor_Examples.png saved!")

def check_reprocessed(date=datetime(2005,1,1)):
    '''
    Read reprocessed file and look at data comparisons
    ''' 
    # First read the reprocessed data:
    data=fio.read_omhchorp(date)
    amf_new = data['AMF_GC']
    amf_old = data['AMF_OMI']
    VC_new = data['VC_GC']
    #VC_old = data['VC_OMI']
    counts = data['gridentries']
    print (type(amf_new))
    lonmids=data['longitude']
    latmids=data['latitude']
    lonedges= regularbounds(lonmids)
    latedges= regularbounds(latmids)
    
    (omg_hcho, omg_lats, omg_lons, omg_amf, omg_counts) = fio.read_omhchog_8day(date)
    
    # remove divide by zero crap
    amf_new[counts<1] = np.nan
    clons,clats=np.meshgrid(lonmids,latmids)
    clats[counts<1] = np.nan
    clons[counts<1] = np.nan
    
    # min mean max of array 
    
    # look at data entry counts...  
    print ("grid squares with/without new AMF:")
    print (np.sum(counts > 0), np.sum(counts < 1))
    print ("min, mean, max of counts:")
    print(mmm(counts))
    
    print ("min mean max old amfs:")
    mmm(amf_old)
    print ("min mean max of OMI gridded amfs")
    #omg_amf[omg_amf < 0] = np.nan
    mmm(omg_amf)
    print ("min mean max new amfs:")
    mmm(amf_new)
    
    print("min mean max new AMFS(sub 60 lats)")
    mmm(fio.filter_high_latitudes(amf_new))
    
    print ("min mean max old VCs:TBD")
    #mmm(VC_old)
    
    print ("min mean max new VCs:")
    mmm(VC_new)
    
    print("min mean max new VCs(sub 60 lat):")
    mmm(fio.filter_high_latitudes(VC_new))
    ## plot old vs new AMF:
    #
    plt.figure(figsize=(18,18))
    
    amf_l=0.1
    amf_h=10
    
    # old amf
    plt.subplot(221)
    plt.title('OMI AMF(regridded)')
    m,cs,cb = linearmap(amf_old, latmids, lonmids, vmin=amf_l, vmax=amf_h)
    cb.set_label('Air Mass Factor')
    
    # new amfs (counts < 1 removed)
    plt.subplot(222)
    plt.title('new AMF')
    m,cs,cb = linearmap(amf_new, latmids, lonmids, vmin=amf_l, vmax=amf_h)
    
    # pct difference
    plt.subplot(223)
    plt.title('(new-old)*100/old')
    m,cs,cb = linearmap((amf_new-amf_old)*100/amf_old, clats, clons, vmin=-100, vmax=100)
    cb.set_label('Pct Changed')
    
    # AMF from OMHCHOG
    plt.subplot(224)
    plt.title('OMHCHOG AMF(orig gridded product)')
    m,cs,cb = linearmap(omg_amf, omg_lats, omg_lons, vmin=amf_l, vmax=amf_h)
    cb.set_label('Air Mass Factor .25x.25')
    
    
    # title and save the figure
    plt.suptitle('AMF updated with GC aprioris 2005 01 01', fontsize=22)
    figname1='Reprocessed_AMF_comparison.png'
    plt.savefig(figname1)
    print("figure saved:" + figname1)
    # clear the figure
    plt.clf()
    mlons,mlats=np.meshgrid(lonedges,latedges)
    
    plt.figure(figsize=(18,18))
    ## Old vs New Vertical Columns
    #
    # old VCs
    plt.subplot(221)
    plt.title('Regridded VC')
    #m,cs,cb = createmap(VC_old, latmids, lonmids)
    cb.set_label('Molecules/cm2')
    
    # new VCs
    plt.subplot(222)
    plt.title('Reprocessed VC')
    m,cs,cb = createmap(VC_new, mlats, mlons)
    cb.set_label('Molecules/cm2')
    
    # diff:
    plt.subplot(223)
    plt.title('OMHCHOG (L2 gridded)')
    m,cs,cb = createmap(omg_hcho, omg_lats, omg_lons)
    cb.set_label('New-Old')
    
    plt.subplot(224)
    plt.title('Pct Diff')
    #m,cs,cb = linearmap(100.0*(VC_new-VC_old)/VC_old, mlats, mlons, vmin=-100,vmax=100)
    cb.set_label('(New-Old) * 100 / Old')
    
    # save figure
    plt.suptitle('Reprocessed Vertical Columns 2005 01 01',fontsize=22)
    figname2='Reprocessed_VC_comparison.png'
    plt.savefig(figname2)
    print("figure saved: "+figname2)
    plt.close()

def test_reprocess():
    '''
    read reprocessed data and check for problems...
    '''
    # run it, save it, play with 1-day mean
    meandict = fio.omhcho_1_day_reprocess(save=True)
    check_array(meandict['VC_GC'])
    # check the data created...
    test_hchorp_apriori()
    check_reprocessed()  

def test_amf_calculation():
    '''
    Grab 5 columns and the apriori and omega values and check the AMF calculation on them
    Plot old, new AMF and integrand at each column
    '''
    day=datetime(2005,1,1)
    # Read OMHCHO data ( 1 day of swathes averaged )
    omhchorg= fio.read_omhchorg(day,oneday=True)
    
    omega=omhchorg['ScatteringWeight'] 
    S_omi =omhchorg['ShapeFactor'] # [720, 1152, 47]
    amf_omi = omhchorg['AMF']  # [720, 1152, 47]
    amfg_omi = omhchorg['AMF_G']  # [720, 1152, 47]
    plevs_omi = omhchorg['PressureLevels'] # [720, 1152, 47]
    Sigma_omi = np.zeros(plevs_omi.shape)
    om_toa  = plevs_omi[:,:,-1]
    om_diff = plevs_omi[:,:,0]-om_toa
    for ss in range(47):
        Sigma_omi[:,:,ss] = (plevs_omi[:,:,ss] - om_toa)/om_diff
    # read one swathe of omhcho and check pressure levels
    omhchopath=fio.determine_filepath(day)[0]
    omhcho = fio.read_omhcho(omhchopath) # [47, 1496, 60]
    
    # read gchcho 
    gchcho = fio.read_gchcho(day)
    lats = gchcho.lats
    lons = gchcho.lons
    S_gc = gchcho.Shape_s
    Sigma_gc = gchcho.sigmas
    
    # Look at 5 random columns
    scount=5
    ii=random.sample(range(-90,90), 50) # lat sample
    jj=random.sample(range(-180,180),50) # lon sample
    omegas=[]
    Shapes=[]
    Product=[]
    Product_new=[]
    AMF_G=[]
    AMF_old=[]
    AMF_new=[]
    goodcols=[]
    
    f, axes = plt.subplots(1, scount, sharey=True, figsize=(17,10))
    axes[0].set_ylim([1,0])
    axes[0].set_ylabel('Sigma')
    plt.setp(axes, xlim=[0,6], xticks=range(7))
    # find 5 GOOD columns(non nans)
    for j in range(50):
        latj=np.searchsorted(lats,ii[j])
        lonj=np.searchsorted(lons,jj[j])
        if np.isnan(omega[latj,lonj,:]).all():
            continue
        else:
            goodcols.append((latj,lonj))
        if len(goodcols) == scount:
            break
    
    # at each lat/lon grab column omega and shape and AMFs
    for (lati,loni),i in zip(goodcols,range(scount)):
        
        AMF_old.append(amf_omi[lati,loni])
        AMF_G.append(amfg_omi[lati,loni])
        omegas.append(omega[lati,loni,:])
        Shapes.append(S_gc[:,lati,loni])
        wcoords=Sigma_omi[lati,loni,:]
        winterp=interp1d(omegas[i],wcoords,'linear',bounds_error=False)#, fill_value=0.0)
        scoords=Sigma_gc[:,lati,loni]
        sinterp=interp1d(Shapes[i],scoords,'linear',bounds_error=False)#, fill_value=0.0)
        
        
        # numpy linear interpolation
        # [::-1] simply reverses the arrays
        def winterp_new(x):
            return np.interp(x, wcoords[::-1],  omegas[i][::-1])
        
        def sinterp_new(x):
            return np.interp(x, scoords[::-1], Shapes[i][::-1])
        
        Product.append(winterp(scoords)*sinterp(scoords))
        Product_new.append(winterp_new(scoords)*sinterp_new(scoords))
        # use all these to create new AMF
        innerplot='pictures/AMF_test_innerplot%d.png'%i
        AMF_new.append(reprocess.calculate_amf_sigma(AMF_G[i], 
            omegas[i], Shapes[i], wcoords, scoords,
            plotname=innerplot, plt=plt))
        
        # Plot the 5 random columns
        ax=axes[i]
        # title=[lat, lon]
        ax.set_title("[%4d, %4d]"%(lats[lati],lons[loni]))
        for k in range(3):
            x=[ omegas[i], Shapes[i], Product_new[i] ][k]
            y=[wcoords, scoords, scoords][k]
            style =['--','-','.'][k]
            leg = ['omega','apriori','product(new)'][k]
            ax.plot(x,y, style,label=leg, linewidth=2)
            
        # Also plot the interpolated things
        for k in np.arange(2,4):
            x=[winterp(scoords), sinterp(scoords), winterp_new(scoords), sinterp_new(scoords)][k]
            y=scoords
            c=['maroon','darkgreen'][k%2]
            d=['.','x'][int(k/2)]
            leg=['w interp','S interp','w interp(new)','S interp(new)'][k]
            ax.plot(x,y, d,color=c,linewidth=1,label=leg)
        
        # text of AMF amounts
        y=0
        for label in [ 'AMF_G=%1.2f'%AMF_G[i], 'AMF=%1.2f'%AMF_old[i],'AMF_new=%2.2f'%AMF_new[i]]:
            ax.annotate(label, xy=(1, y), xycoords='axes fraction', fontsize=14,
                horizontalalignment='right', verticalalignment='bottom')
            y=y+0.05
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('pictures/AMF_test.png')
    print('pictures/AMF_test.png Saved')
    
    plt.close()
    plt.figure(3,(5,5))

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
    outpic='pictures/high_AMF_omegas.png'
    plt.savefig(outpic)
    print('saved %s'%outpic)

def check_RSC(day=datetime(2005,1,1), track_corrections=False):
    '''
    Grab the RSC from both GEOS-Chem and OMI for a particular day
    Plot and compare the RSC region
    Plot the calculated corrections
    '''
    print("Running check_RSC")
    # Read in one day average
    omhchorp1=fio.read_omhchorp(day,oneday=True)
    yyyymmdd=day.strftime("%Y%m%d")
    rsc=omhchorp1['RSC']
    ref_lat_bins=omhchorp1['RSC_latitude']
    rsc_gc=omhchorp1['RSC_GC'] # RSC_GC is in molecs/cm2 as of 11/08/2016
    
    ## plot each track with a slightly different colour
    #
    if track_corrections:
        f1=plt.figure(0,figsize=(8,10))
        colors=[plt.cm.jet(i) for i in np.linspace(0, 1, 60)]
        for track in range(60):
            plt.plot(rsc[:,track], ref_lat_bins, '.', color=colors[track])
            #plt.plot(rsc_function(ref_lat_bins,track), ref_lat_bins, color=colors[track])
        plt.ylabel('latitude')
        plt.xlabel('molecules/cm2')
        plt.title('Reference sector correction for %s'%yyyymmdd)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=plt.Normalize(vmin=0,vmax=60))
        sm._A=[]
        cb=plt.colorbar(sm)
        cb.set_label('track')
        outfig1='pictures/track_corrections%s.png'%yyyymmdd
        plt.savefig(outfig1)
        print(outfig1+' saved')
        plt.close(f1)
    
    def RSC_map(lons,lats,data,labels=[0,0,0,0]):
        ''' Draw standard map around RSC region '''
        vmin, vmax= 5e14, 1e16
        lat0, lat1, lon0, lon1=-75, 75, -170, -130
        m=Basemap(lon0,lat0,lon1,lat1, resolution='i', projection='merc')
        cs=m.pcolormesh(lons, lats, data, latlon=True, 
              vmin=vmin,vmax=vmax,norm = LogNorm(), clim=(vmin,vmax))
        cs.cmap.set_under('white')
        cs.cmap.set_over('pink')
        cs.set_clim(vmin,vmax)
        m.drawcoastlines()
        m.drawmeridians([ -160, -140],labels=labels)
        return m, cs
    
    ## plot the reference region, 
    #
    # we need 501x3 bounding edges for our 500x2 2D data (which is just the 500x1 data twice)
    rsc_lat_edges= list(ref_lat_bins-0.18)
    rsc_lat_edges.append(90.)
    rsc_lat_edges = np.array(rsc_lat_edges)
    lons_gc, lats_gc = np.meshgrid( [-160., -150, -140.], rsc_lat_edges )
    
    # Make the GC map:
    f2=plt.figure(1, figsize=(14,10))
    plt.subplot(141)
    rsc_gc_new = np.transpose(np.array([ rsc_gc, rsc_gc] ))
    m,cs = RSC_map(lons_gc,lats_gc,rsc_gc_new, labels=[0,0,0,1])
    m.drawparallels([-45,0,45],labels=[1,0,0,0])
    cb=m.colorbar(cs,"right",size="8%", pad="3%")
    cb.set_label('Molecules/cm2')
    plt.title('RSC_GC')
    
    # add the OMI reference sector for comparison
    sc_omi=omhchorp1['SC'] # [ 720, 1152 ] molecs/cm2
    lats_rp=omhchorp1['latitude']
    lons_rp=omhchorp1['longitude']
    rsc_lons= (lons_rp > -160) * (lons_rp < -140)
    newlons=regularbounds(lons_rp[rsc_lons])
    newlats=regularbounds(lats_rp)
    newlons,newlats = np.meshgrid(newlons,newlats)
    # new map with OMI SC data
    plt.subplot(142)
    plt.title("SC_OMI")
    m,cs=RSC_map(newlons, newlats, sc_omi[:,rsc_lons])
    m.drawmeridians([ -160, -140],labels=[0,0,0,0])
    
    ## Another plot using OMI_VC (old reprocessed data)
    #
    vc_omi=omhchorp1['VC_OMI']
    plt.subplot(143)
    plt.title('VC_OMI')
    m,cs=RSC_map(newlons,newlats,vc_omi[:,rsc_lons])

    ## One more with VC_GC over the ref sector
    #
    vc_gc=omhchorp1['VC_GC'] # [ 720, 1152 ] molecs/cm2
    # new map with OMI SC data
    plt.subplot(144)
    plt.title("VC_GC")
    m,cs=RSC_map(newlons,newlats,vc_gc[:,rsc_lons])
    
    
    f2.suptitle('GEOS_Chem VC vs OMI SC over RSC on %s'%yyyymmdd)
    outfig2='pictures/RSC_GC_%s.png'%yyyymmdd
    plt.savefig(outfig2)
    print(outfig2+' saved')
    plt.close(f2)

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
    plt.savefig("pictures/distributions%d_%s"%([8,1][oneday], yyyymmdd))
    plt.close()

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running tests.py")
    #test_fires_fio()
    #test_amf_calculation() # Check the AMF stuff
    #check_flags_and_entries() # check how many entries are filtered etc...
    for oneday in [True, False]:
        test_reprocess_corrected(run_reprocess=False, oneday=oneday)
    #check_high_amfs()
    
    #test_hchorp_apriori()
    #test_gchcho()
    #test_gchcho()
    #test_reprocess()
    #check_reprocessed()
    
    # Check that cloud filter is doing as expected using old output without the cloud filter
    #compare_cloudy_map()
    
    # Plot SC, VC_omi, VC_gc, AMF_omi, AMF_gc from
    # one or eight day average reprocessed netcdf output
    #check_RSC(track_corrections=True)
