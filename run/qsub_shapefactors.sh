#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N ShapeFactor
#PBS -l walltime=00:20:00
#PBS -l mem=5000MB
#PBS -l cput=00:30:00
#PBS -l ncpus=2
#PBS -j oe

##
## This needs to be run on NCI
## Take UCX satellite_output and create shapefactors 
## Creates he5 files in Data/gchcho
## 
## History:
##  15/09/2017: jwg first version created tested on UCX satellite output

if [ -z ${PBS_O_LOGNAME} ] || [ -z ${YEAR} ]
then
    echo "  EG: qsub -v YEAR=2006 -o logs/shapefactors2006 $0"
    echo "    will make shapefactors for all of 2006"
    exit 0
fi

# setup virtual display window for idl
Xvfb :99 &
export DISPLAY=:99

# Run the script in IDL using input arguments for each month
#
for i in `seq 1 12`;
do

    # Run IDL script for this month
    /short/m19/jwg574/OMI_regridding/run/make_shapefactors , ${YEAR}, $i

done



