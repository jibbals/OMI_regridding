#!/bin/bash

##
## This needs to be run on NCI
## 

if [ $# -lt 1 ]
then
    echo "  EG: $0 200501"
    echo "    (For January 2005 conversion)"
    exit 0
fi

date=$1

# Go to UCX dir:
pushd /short/m19/jwg574/rundirs/geos5_2x25_tropchem_halfisoprene/trac_avg
fname=trac_avg.geos5_2x25_tropchem.${date}010000

# poop out the days as netcdf
idl <<END
bpch2coards, "$fname", "trac_avg_halfisop_${date}.nc"
END



