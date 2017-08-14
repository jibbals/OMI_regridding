#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N TracAvgNETCDF
#PBS -l walltime=01:00:00
#PBS -l mem=20000MB
#PBS -l cput=4:00:00
#PBS -l wd
#PBS -l ncpus=4
#PBS -j oe

##
## Convert trac_avg to netcdf 
##

#---------------------------------
# send to queue with qsub
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ]; then
    echo "EG usage: qsub $0"
    exit 0
fi


# Set up virtual display window so IDL/NCI doesn't poo itself
Xvfb :99 &
export DISPLAY=:99

# run the script
bash /home/574/jwg574/OMI_regridding/run/trac_avg_to_netcdf.sh 200501 trac_avg_200501_test.nc



