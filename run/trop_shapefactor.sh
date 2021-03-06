#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N tropShapeFactor
#PBS -l walltime=00:08:00
#PBS -l mem=5000MB
#PBS -l cput=00:08:00
#PBS -l ncpus=1
#PBS -l software=idl
#PBS -j oe

##
## This needs to be run on NCI
## Take tropchem satellite_output and create shapefactors 
## Creates he5 files in Data/gchcho
## 
## History:
##  15/09/2017: jwg first version created tested on UCX satellite output
##  20/9/17: jwg updated so can be run directly or with QSUB
##  26/10/17: turned into trop_shapefactor.sh

if [ -z ${YEAR} ] || [ -z ${MONTH} ]; then
    if [ $# -lt 2 ]; then 
        echo "  EG: $0 2005 1"
        echo "    will make Data/gchcho/shapefactors For jan 2005, using tropchem satellite_output"
        echo "  EG2: qsub -v YEAR=2005,MONTH=1 $0"
        exit 0
    fi
    # If called directly set year and month here
    YEAR=$1
    MONTH=$2
else 
    # setup virtual display window for idl if we run in compute node
    #Xvfb :99 &
    #export DISPLAY=:99
    echo "DISPLAY: $DISPLAY"
fi


# Add leading zero to month argument
printf -v mm "%02d" $MONTH 
yyyy=$YEAR

# working dir
wdir=/home/574/jwg574/rundirs/geos5_2x25_tropchem/satellite_output
# check the data exists
if [ ! -f ${wdir}/ts_satellite_omi.${yyyy}${mm}01.bpch ]
then
    echo "${wdir}/ts_satellite_omi.${yyyy}${mm}01.bpch doesn't exist!!!"
    exit 1
fi

# output dir
odir=/home/574/jwg574/OMI_regridding/Data/gchcho
# script to make output
sname=shapefactor_from_trop_satellite_output.pro
funcname=shapefactor_from_trop_satellite_output
script=/home/574/jwg574/OMI_regridding/run/${sname}

# Copy the creation script to the satellite output dir:
cp $script ${wdir}/${sname}

# Go to dir:
pushd $wdir

# Run the script in IDL using input arguments for month and year
# 
idl <<END
$funcname, $yyyy, $MONTH
END

# Remove the script
#rm ${wdir}/${sname}

# move files into data directory
mv ${wdir}/trop_shapefactor_${yyyy}${mm}.he5 ${odir}/

# return to original dir for no reason
popd


