#!/bin/bash

##
## This needs to be run on NCI
## Go to desirerd directory and save the satellite output into monthly averaged 
## he5 files
## 

if [ $# -lt 3 ]
then
    echo "  EG: $0 2005 1 halfisoprene"
    echo "    (For jan 2005 conversion of halfisop run)"
    echo "    third argument can be tropchem, UCX, halfisoprene, or noisoprene for second argument"
    exit 0
fi

#printf -v mm "%02d" $1 # Add leading zero to month argument
mm=$2
yyyy=$1
runstr=$3

wdir=/home/574/jwg574/rundirs/geos5_2x25_tropchem
script=/home/574/jwg574/OMI_regridding/run/create_monthly_shapefactor.pro

# First we determine where we will go
if runstr==tropchem
then
    wdir=${wdir}/satellite_output
else
    wdir=${wdir}_${runstr}/satellite_output
fi


# Copy the creation script to the satellite output dir:
cp $script ${wdir}/create_monthly_shapefactor.pro
# Go to dir:
pushd $wdir

# Run the script in IDL using input arguments for month and year
# 
idl <<END
create_monthly_shapefactor, $yyyy, $mm
END


# Remove the script
rm ./create_monthly_shapefactor.pro
# return to original dir for no reason
popd


