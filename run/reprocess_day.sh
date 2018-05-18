#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N RP_Day
#PBS -l walltime=01:40:00
#PBS -l mem=10000MB
#PBS -l cput=01:40:00
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
    if [ $# -lt 3 ]; then
        dstr="20050203"
        echo "EG usage 1: $0 2005 2 5"
        echo "  To reprocess omhchorp for feb 5 2005"
        echo "EG usage 2: qsub -v dstr=$dstr -o logs/reprocess$dstr.log  $0"
        echo " To reprocess and get omhchorp for 3/feb/2005"
        exit 0
    else
        printf -v dstr "%4d%02d%02d" $1 $2 $3
        echo $dstr
    fi
        
fi

# run the day
python3 << END
from datetime import datetime
import reprocess
date=datetime.strptime(str(${dstr}),"%Y%m%d")
assert (date > datetime(2004,12,31)) and (date < datetime(2013,4,1))

reprocess.create_omhchorp(date)

END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

