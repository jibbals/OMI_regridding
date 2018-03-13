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
pushd /short/m19/jwg574/rundirs/UCX_geos5_2x25/trac_avg
fname=trac_avg_geos5_2x25_UCX_updated.${date}010000

# poop out the days as netcdf
idl <<END
bpch2coards, "$fname", "trac_avg_UCX_${date}.nc"
END

# Run script for each month in the year
#for i in `seq 1 12`
#do
#    f=trac_avg_geos5_2x25_UCX_updated.${year}*
#    echo $f
#    # make netcdf
#done

# combine the months into a year?

#outname=trac_avg_${date}.nc
## if we input a second argument, save ncfile to that instead
#if [ $# -eq 2 ]
#then
#    outname=$2
#fi
#ncrcat tavg_temp_*.nc ${outname}


## delete intermediate poop
#rm tavg_temp*

# return to original dir for no reason
popd


