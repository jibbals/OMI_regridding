#!/bin/bash

##
## This needs to be run on NCI
## 

if [ $# -lt 1 ]
then
    echo "  EG: $0 200501"
    echo "    (For jan 2005 conversion)"
    exit 0
fi

date=$1

# Go to tropchem dir:
pushd /home/574/jwg574/rundirs/geos5_2x25_tropchem/trac_avg
fname=trac_avg.geos5_2x25_tropchem.${date}010000

# poop out the days as netcdf
idl <<END
bpch2coards, "$fname", "tavg_temp_%DATE%.nc"
END

# combine those poops
for f in tavg_temp*.nc
do 
    echo $f
    # Add DXYP to all the files...
    ncks -A -v DXYP__DXYP tavg_temp_${date}01.nc $f
done

outname=trac_avg_${date}.nc
# if we input a second argument, save ncfile to that instead
if [ $# -eq 2 ]
then
    outname=$2
fi
ncrcat tavg_temp_*.nc ${outname}


# delete intermediate poop
rm tavg_temp*

# return to original dir for no reason
popd


