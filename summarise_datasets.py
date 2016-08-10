'''
File to check summaries of omhcho, omhchog, modis fires, gchcho, etc.
'''
## Modules
# these 2 lines make plots not show up ( can save them as output faster )
# use needs to be called before anythin tries to import matplotlib modules
#import matplotlib
#matplotlib.use('Agg')

# my file reading and writing module
import fio
import numpy as np
from datetime import datetime

##############################
########## FUNCTIONS #########
##############################

def mmm(arr):
    print("min:%2.5e, mean:%2.5e, max:%2.5e"%(np.nanmin(arr),np.nanmean(arr),np.nanmax(arr)))
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

def test_fires():
    '''
    '''
    day = datetime(2005,1,1)
    ## get normal and interpolated fire
    orig, lats, lons=fio.read_8dayfire(day)
    
    check_array(orig)
    

def test_omhcho():
    '''
    how many entries/day, good entries/day, etc.
    '''
    day=datetime(2005,1,1)
    paths=fio.determine_filepath(day)
    print(day)
    print(paths)
    omhcho=fio.read_omhcho(paths[0])
    h=omhcho['HCHO']
    entries=np.cumprod(np.shape(h))[-1]
    print('%d entries in first swath'% entries)
    print("%d good entries in first swath"%np.sum(~np.isnan(omhcho['HCHO'])))
    

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running summaries_datasets.py")
    test_omhcho()
