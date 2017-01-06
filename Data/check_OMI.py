'''
File to check summaries of omhcho, omhchog, modis fires, gchcho, etc.
'''
## Modules
# these 2 lines make plots not show up ( can save them as output faster )
# use needs to be called before anythin tries to import matplotlib modules
#import matplotlib
#matplotlib.use('Agg')

# import some stuff from parent directory:
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import my stuff:
import fio
import plotting


# maths and plotting tihngs
import numpy as np
from mpl_
from datetime import datetime

##############################
########## FUNCTIONS #########
##############################

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
    
def plot_swaths(day):
    '''  Plot a swath/day/8day picture '''
    

##############################
########## IF CALLED #########
##############################
if __name__ == '__main__':
    print("Running summaries_datasets.py")
    test_omhcho()
