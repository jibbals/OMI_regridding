#!/bin/bash

##
## This needs to be run on NCI
## Take UCX satellite_output and create shapefactors 
## Creates he5 files in Data/gchcho
## 
## History:
##  15/09/2017: jwg first version created tested on UCX satellite output

if [ $# -lt 1 ]
then
    echo "  EG: $0 2006"
    echo "    will make shapefactors for all of 2006"
    exit 0
fi

YEAR=$1
# script
script=/short/m19/jwg574/OMI_regridding/run/shapefactor.sh

# Run the script in IDL using input arguments for each month
#
for i in `seq 1 12`;
do
    printf -v mm "%02d" $i
    logfile=/short/m19/jwg574/OMI_regridding/logs/shapefactors${YEAR}${mm}.log
    # Run IDL script for this month
    qsub -v YEAR=${YEAR},MONTH=${i} -o $logfile $script
    echo "$YEAR $mm sent to qsub, log to $logfile"

done



