#!/bin/bash

##
## This needs to be run FROM WITHIN THE trac_avg FOLDER
## 

if [ $# -lt 1 ]
then
    echo "  EG: $0 200501"
    echo "    (For jan 2005 conversion)"
    echo "    Make sure you're running this from trac_avg folder.."
    exit 0
fi

date=$1
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
ncrcat tavg_temp_*.nc trac_avg_${date}.nc


# delete intermediate poop
rm tavg_temp*

