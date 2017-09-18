#!/bin/bash

##
## This needs to be run on NCI
## Take UCX satellite_output and create shapefactors 
## Creates he5 files in Data/gchcho
## 
## History:
##  15/09/2017: jwg first version created tested on UCX satellite output

if [ $# -lt 2 ]
then
    echo "  EG: $0 2005 1"
    echo "    will make Data/gchcho/shapefactors For jan 2005, using ucx satellite_output"
    exit 0
fi

printf -v mm "%02d" $2 # Add leading zero to month argument
yyyy=$1

# working dir
wdir=/home/574/jwg574/rundirs/UCX_geos5_2x25/satellite_output
# check the data exists
if [ ! -f ${wdir}/ts_satellite.${yyyy}${mm}01 ]
then
    echo "${wdir}/ts_satellite.${yyyy}${mm}01 doesn't exist!!!"
    exit 1
fi

# output dir
odir=/home/574/jwg574/OMI_regridding/Data/gchcho
# script to make output
sname=shapefactor_from_ucx_satellite_output.pro
script=/home/574/jwg574/OMI_regridding/run/${sname}

# Copy the creation script to the satellite output dir:
cp $script ${wdir}/${sname}

# Go to dir:
pushd $wdir

# Run the script in IDL using input arguments for month and year
# 
idl <<END
shapefactor_from_ucx_satellite_output, $yyyy, $2
END

# Remove the script
#rm ${wdir}/${sname}

# move files into data directory
mv ${wdir}/ucx_shapefactor_${yyyy}${mm}.he5 ${odir}/

# return to original dir for no reason
popd


