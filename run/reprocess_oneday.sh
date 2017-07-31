#!/bin/bash
#PBS -P m19
#PBS -q express
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
if [ -z ${PBS_O_LOGNAME} ]; then
    echo "EG usage: qsub run_one_day.sh"
    exit 0
fi

# run the day
python3 << END
from datetime import datetime
import reprocess
reprocess.create_omhchorp_1(datetime(2005,1,9),verbose=True)
END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

