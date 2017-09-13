#!/bin/bash

##
## This needs to be run on NCI
## 

if [ $# -lt 2 ]
then
    echo "  EG: $0 200501 halfisoprene"
    echo "    (For jan 2005 conversion of halfisop run)"
    echo "    use halfisoprene|noisoprene for second argument"
    exit 0
fi

date=$1
runstr=$2

# Go to tropchem dir:
pushd /home/574/jwg574/rundirs/geos5_2x25_tropchem_${runstr}/trac_avg
fname=trac_avg.geos5_2x25_tropchem.${date}010000

# poop out the days as netcdf
idl <<END
bpch2coards, "$fname", "trac_avg_${date}.nc"
END


# return to original dir for no reason
popd


