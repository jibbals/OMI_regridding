#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N pp_trop_amf
#PBS -l walltime=9:00:00
#PBS -l mem=2000MB
#PBS -l cput=18:00:00
#PBS -l jobfs=10MB
#PBS -l wd
#PBS -l ncpus=4
#PBS -j oe

# -P is project group to charge resource time to
# -q is which queue to use
# -N is name
# -l walltime is wallclock time limit
# -l wd means run job in working directory where it was submitted from
# -l jobfs is io intensive job time, default is 100MB.
# -j oe  output and error stream sent to output channel

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${YEAR} ] || [ -z ${MONTH} ] || [ -z ${DAY} ]; then
    echo "EG usage: qsub -v DAY=1,MONTH=1,YEAR=2005 ${0}"
    echo "    To run PP amf lidort + thingy for one day"
    exit 0
fi

# unlimit stacksize for openmp applications
ulimit -s unlimited
# unlimit coredumpsize so we can see core dumps
ulimit -c unlimited
# unlimit memorylocked 
ulimit -l unlimited

# Zero pad the day and month:
yyyy=${YEAR}
printf -v mm "%02d" ${MONTH}
printf -v dd "%02d" ${DAY}

# parameters
PP_amf_dir="/home/574/jwg574/AMF_omi"
regrid_data_dir="/home/574/jwg574/OMI_regridding/Data"
csv_dir="${regrid_data_dir}/omhcho_csv"
gchem_dir="${regrid_data_dir}/GC_Output/geos5_2x25_tropchem/satellite_output"
outfile="${regrid_data_dir}/pp_amf/tropchem/amf_${yyyy}${mm}${dd}.csv"

satin="${csv_dir}/${yyyy}-${mm}-${dd}_for_AMF.csv"
nd51in="${gchem_dir}/ts_satellite_omi.${yyyy}${mm}${dd}.bpch"

# cd to the AMF folder and run the utility
# We cd there to have jv_spec.dat and similar things in the working directory
pushd $PP_amf_dir 
echo " ${PP_amf_dir}/amf.run $satin $nd51in $outfile $yyyy $mm $dd"

${PP_amf_dir}/amf.run $satin $nd51in $outfile $yyyy $mm $dd
#./amf.run testfiles/2014-04-01_for_AMF.csv testfiles/ts_satellite_omi.20140401.bpch testrun_ouptut.csv 2014 4 1
popd

