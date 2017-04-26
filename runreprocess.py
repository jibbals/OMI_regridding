#!/apps/python3/3.5.2/bin/python3

# get command line options:
import sys, getopt
# module I created to reprocess OMI swath data
import reprocess
# datetime to reprocess
from datetime import datetime, timedelta

def main(argv):
    startday=0
    # Read input arguments
    #
    try:
        opts, args = getopt.getopt(argv,"hs:",["start="])
    except getopt.GetoptError:
        print('python3 runreprocess.py -s <start date offset(multiple of 8, plus 1)>')
        sys.exit(2)
    
    # handle each argument
    for opt, arg in opts:
        if opt == '-h':
            print('example: runreprocess.py -s 17')
            sys.exit()
        elif opt in ("-s","--start"):
            startday=int(arg)
        else:
            print('unrecognised option: '+opt)
            sys.exit(2)
    
    # Check there were enough arguments
    #
    if len(opts) < 1:
        print('example: runreprocess.py -s 17')
        sys.exit()
    
    # Check sensible startday (1,9,17,25, ...)
    if startday % 8 != 1:
        print("start day needs to be N * 8 + 1 for some N")
        sys.exit(2)
    start=datetime(2005,1,1)+timedelta(days=startday-1)
    
    ####################################
    ## CALL REPROCESSING METHODS HERE ##
    ####################################
    ymd=start.strftime('%Y%m%d')
    print('Reprocessing from %s'%ymd)
    reprocess.Reprocess_N_days(start, latres=0.25, lonres=0.3125, days=8, processes=8, remove_clouds=True, remove_fires=True)
    print("Combining the 8 days from %s"%ymd)
    reprocess.create_omhchorp_8(start,latres=0.25,lonres=0.3125)

if __name__ == "__main__":
    main(sys.argv[1:])

