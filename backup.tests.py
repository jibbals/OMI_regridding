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

def createmap(data,lats,lons,
              llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-179, urcrnrlon=179):
    m=Basemap(llcrnrlat=llcrnrlat,  urcrnrlat=urcrnrlat,
          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
          resolution='i',projection='merc')
    if len(lats.shape) == 1:
        latsnew=regularbounds(lats)
        lonsnew=regularbounds(lons)
    else:
        latsnew,lonsnew=(lats,lons)
    vmin=5e13
    vmax=1e17
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

def test_reprocess_corrected(run_reprocess=True):
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
    omi=fio.read_omhchorp(date,oneday=True)
    # Grab monthly averaged GEOS-Chem data
    # gc=fio.read_gchcho(date)
    
    counts = omi['gridentries']
    print( "at most %d entries "%np.nanmax(counts) )
    print( "%d entries in total"%np.nansum(counts) )
    lonmids=omi['longitude']
    latmids=omi['latitude']
    lons,lats = np.meshgrid(lonmids,latmids)
    
    # Plot 
    # SC, VC_omi, AMF_omi
    # VCC, VC_gc, AMF_gc
    # 
    f, axes = plt.subplots(2,3,num=0,figsize=(16,14))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    plt.sca(axes[0,0])
    m,cs,cb = createmap(omi['SC'],lats,lons)
    plt.title('SC')
    plt.sca(axes[0,1])
    m,cs,cb = createmap(omi['VC_OMI'],lats,lons)
    plt.title('VC OMI')
    plt.sca(axes[0,2])
    m,cs,cb = linearmap(omi['AMF_OMI'],lats,lons,vmin=1.0,vmax=6.0)
    plt.title('AMF OMI')
    plt.sca(axes[1,0])
    m,cs,cb = createmap(omi['VCC'],lats,lons)
    plt.title('VCC')
    plt.sca(axes[1,1])
    m,cs,cb = createmap(omi['VC_GC'],lats,lons)
    plt.title('VCC')
    plt.sca(axes[1,2])
    m,cs,cb = linearmap(omi['AMF_GC'],lats,lons,vmin=1.0,vmax=6.0)
    plt.title('AMF_GC')
    
    # save plots
    plt.tight_layout()
    outfig="pictures/one_day_corrected%s.png"%date.strftime("%Y%m%d")
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
    counts = data['GridEntries']
    lonmids=data['Longitude']
    latmids=data['Latitude']
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
    mmm(data['ColumnAmountHCHO_OMI'] )
    print("NEW_VC:  ")
    mmm(data['ColumnAmountHCHO'] )
    print("Unfiltered stats:")
    print("OLD AMF: ")
    mmm(cloudydata['AMF_OMI'])
    print("NEW AMF: ")
    mmm(cloudydata['AMF_GC'])
    print("OLD_VC:  ")
    mmm(cloudydata['ColumnAmountHCHO_OMI'] )
    print("NEW_VC:  ")
    mmm(cloudydata['ColumnAmountHCHO'] )
    
    # Plot OMI, oldrp, newrp VC map
    f, axes = plt.subplots(2,3,num=0,figsize=(16,14))
    # Plot OMI, old, new AMF map
    # set currently active axis from [2,3] axes array
    #plt.sca(axes[0,0])
    i=0
    for d in [cloudydata, data]:
        plt.sca(axes[i,0])
        m,cs,cb = createmap(d['ColumnAmountHCHO_OMI'],latmids,lonmids)
        plt.title('VC Regridded')
        plt.sca(axes[i,1])
        m,cs,cb = createmap(d['ColumnAmountHCHO'],latmids,lonmids)
        plt.title('VC Reprocessed')
        plt.sca(axes[i,2])
        m,cs,cb = linearmap(d['AMF_GC'],latmids,lonmids,vmin=1.0,vmax=6.0)
        plt.title('AMF reprocessed')
        i=1
        
    # save plots
    plt.tight_layout()
    plt.savefig("cloud_filter_effects.png")
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
    ii=random.sample(np.arange(91),10)
    jj=random.sample(np.arange(144),10)
    S_s=gchcho.Shape_s
    for i in range(10):
        print ("sum(S_s[:,%d,%d]): %f"%(ii[i],jj[i],np.sum(S_s[:,ii[i],jj[i]])))
        
    plt.figure(figsize=(14,12))
    m,cs,cb=gchcho.PlotVC()
    plt.savefig('GC_Vertical_Columns.png')
    plt.clf()
    
    plt.figure(figsize=(12,12))
    gchcho.PlotProfile()
    plt.savefig('GC_Profile.png')
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
    plt.savefig('GC_apriori_interpolation.png')
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
    keylist= ['Latitude','Longitude','ShapeFactor_GC','ShapeFactor_OMI','Sigma_GC', 'Sigma_OMI']
    omhchorp = fio.read_omhchorp(day0, oneday=True, keylist=keylist)
    lats, lons = omhchorp['Latitude'], omhchorp['Longitude']
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
    ii=random.sample(np.arange(181)-90, scount) # lat samples
    jj=random.sample(np.arange(361)-180,scount) # lon samples
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
    
    plt.savefig('Shape_Factor_Examples.png')
    print("Shape_Factor_Examples.png saved!")

def check_reprocessed(date=datetime(2005,1,1)):
    '''
    Read reprocessed file and look at data comparisons
    ''' 
    # First read the reprocessed data:
    data=fio.read_omhchorp(date)
    amf_new = data['AMF_GC']
    amf_old = data['AMF_OMI']
    VC_new = data['ColumnAmountHCHO']
    #VC_old = data['ColumnAmountHCHO_OMI']
    counts = data['GridEntries']
    print (type(amf_new))
    lonmids=data['Longitude']
    latmids=data['Latitude']
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
    check_array(meandict['ColumnAmountHCHO'])
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
    ii=random.sample(np.arange(180)-90, 50) # lat sample
    jj=random.sample(np.arange(360)-180,50) # lon sample
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
        AMF_new.append(fio.calculate_amf_sigma(AMF_G[i], omegas[i], Shapes[i], wcoords, scoords))
        
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
            d=['.','x'][k/2]
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
    plt.savefig('AMF_test.png')
    print('AMF_test.png Saved')
    
    plt.close()
    plt.figure(3,(5,5))
    # ALSO plot the OMI apriori
    plt.plot(S_omi,wcoords,color='k',label='OMI apriori')
    plt.tight_layout()
    plt.title('OMI Apriori')
    plt.savefig('OMI_apriori.png')
    
##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    #test_fires_fio()
    #test_amf_calculation()
    #test_hchorp_apriori()
    #test_gchcho()
    #test_gchcho()
    #test_reprocess()
    #check_reprocessed()
    #compare_cloudy_map()
    test_reprocess_corrected(False);
