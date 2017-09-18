#!/bin/bash

#-----------------------
# Run this script with a date input to check what data is available
# Eventually this script will also be able to start other scripts 
#
# History: 
#   Created 18/9/17 by jwg366
#       Just checks files and their creation dates for given input.
#-----------------------

# First check inputs
if [ $# -lt 3 ]; then
    echo "EG: $0 2005 3 13"
    echo "  to check the data for 13th March, 2005"
    exit 0
fi

# Here's a function which tests a filename for existence:
# If the file exits it prints the last modified date
function checkfile {
    # Check argument file exists ($1 is argument to this function)
    if [ ! -f $1 ]; then
        echo "No file!   |  $1" | tee -a $torun
        flag=1 # global flag that file is missing
    else
        # Print out date of last modification and filename
        local mod_date=$(stat -c %y "$1" | cut -d' ' -f1)
        local fname=$(basename $1)
        #echo "$mod_date |  $1"
        echo "$mod_date |  $fname" | tee -a $torun
    fi
}  


# make inputs zero padded and nicer:
printf -v mm "%02d" $2
printf -v dd "%02d" $3
printf -v yy "%04d" $1
ymd=${yy}${mm}${dd}

# Create a file with list of things we'll need to run
torun=torun_${ymd}.txt

# Table of files for stdout
echo "Examining data for $ymd" > $torun
echo "Modified   |  Filename" | tee -a $torun
echo "------------------- (AMF Reprocessing)" | tee -a $torun

flag=0
# locations of ucx, tropchem data:
ucxdir=/short/m19/jwg574/rundirs/UCX_geos5_2x25
tropdir=/short/m19/jwg574/rundirs/geos5_2x25_tropchem
datadir=/short/m19/jwg574/OMI_regridding/Data
rundir=/short/m19/jwg574/OMI_regridding/run

# first check the swath files
swathfiles=( ${datadir}/omhcho/${yy}${mm}/OMI-Aura_L2-OMHCHO_${yy}m${mm}${dd}*.he5 )
swathfile=${swathfiles[0]}
checkfile $swathfile

# then look at satellite_output for UCX
satfile=${ucxdir}/satellite_output/ts_satellite.${ymd}
checkfile $satfile

# Check the converted GC-satellite file (.nc)
satfile_nc=${datadir}/gchcho/ucx_shapefactor_${yy}${mm}.he5
checkfile $satfile_nc


echo "------------------- (Emissions)" | tee -a $torun

# file reprocessed using the above 
newswathfile=${datadir}/omhchorp/omhcho_1p*${ymd}.he5
checkfile $newswathfile

# geos chem tropchem output
tavgfile=${tropdir}/trac_avg/trac_avg.geos5_2x25_tropchem.${ymd}0000
checkfile $tavgfile

# and converted to nc (monthly)
tavgfile_nc=${tropdir}/trac_avg/trac_avg_${yy}${mm}.nc
checkfile $tavgfile_nc

# creating the enew file:
emissionfile=${datadir}/Isop/E_new/emissions_${yy}${mm}.h5
checkfile $emissionfile

# TODO: Full list of scripts to run to get emissions on whatever date
# 
echo '------------------' | tee -a $torun

# Send code needed for full creation to here..
echo "Scripts to run for $ymd emissions file creation:" >> $torun
echo "1) make sure satellite swath files to NCI" >> $torun
echo "2) GEOS-Chem satellite output for UCX is needed for shapefactor creation" >> $torun
echo "3) run/make_shapefactors.sh $yy $2" >> $torun
echo "       runs run/shapefactor_from_ucx_satellite_output.pro on GC satellite data" >> $torun
echo "4) qsub -v dstr=$ymd run/reprocess_oneday.sh" >> $torun
echo "       runs reprocess.omhchorp_1() for $ymd" >> $torun
echo "5) Check there are tracer average files from tropchem output (used for yield)" >> $torun
echo "      $tavgfile" >> $torun
echo "6) run/trac_avg_to_nc_tropchem.sh ${yy}${mm}" >> $torun
echo "       runs bpch2coards on the geoschem trac_avg output" >> $torun
echo "7) qsub -o logs/log.inversion${yy}${mm} -v MONTH=${yy}${mm} run/inversion_month.sh" >> $torun
echo "       runs Inversion.store_emissions() on that month of data" >> $torun

echo "Code to run for this date sent to $torun"
    

# TODO: Fires and Anthro filters files list
#
