#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N ReprocessOneDay
#PBS -l walltime=02:00:00
#PBS -l mem=20000MB
#PBS -l cput=16:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

##
## This script reprocesses one day of swaths and GC output 
## into omhcho_1 data
##
#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${dstr} ]; then
    echo "EG usage: qsub -v dstr=\"20050205\" run/reprocess_oneday.sh"
    echo " To reprocess and get omhchorp_1 for 5/feb/2005"
    exit 0
fi

# run the day
python3 << END
from datetime import datetime
import reprocess
date=datetime.strptime(${dstr},"%Y%m%d")
assert (date > datetime(2004,12,31)) and (date < datetime(2013,4,1))

reprocess.create_omhchorp_1(date,verbose=True)

END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

