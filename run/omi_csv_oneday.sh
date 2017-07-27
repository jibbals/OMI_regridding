#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N makeAMFcsv
#PBS -l walltime=00:40:00
#PBS -l mem=5000MB
#PBS -l cput=00:10:00
#PBS -l wd
#PBS -l ncpus=8
#PBS -j oe


#
#   This script takes omi swath dataset and poops out the _amf version
#   into Data/omhcho_amf
#

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${YEAR} ] || [ -z ${MONTH} ] || [ -z ${DAY} ]; then
    echo "EG usage: qsub -v DAY=1,MONTH=1,YEAR=2005 ${0}"
    exit 0
fi


# run the things
#
python3 <<END
import amf_calculation
from datetime import datetime
amf_calculation.pixel_list_to_csv(date=datetime(${YEAR},${MONTH},${DAY}))
END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

