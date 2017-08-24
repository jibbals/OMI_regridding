#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N Inversion
#PBS -l walltime=03:00:00
#PBS -l mem=10000MB
#PBS -l cput=05:00:00
#PBS -l wd
#PBS -l ncpus=2
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${END} ]; then
    echo "EG usage: qsub -o logs/log.inversion -v END=20050501 run/inversion.sh"
    echo "   to save E_new from 20050101 to END "
    exit 0
fi

# run python code snippet:
python3 <<END
import Inversion
from datetime import datetime as dt

# get start to finish dates:
d0=dt(2005,1,1)
d0s=d0.strftime('%Y%m%d')
d1=dt.strptime(str(${END}),'%Y%m%d')
d1s=d1.strftime('%Y%m%d')
ndays=(d1-d0).days

print("Beginning inversion (%d days) from %s to %s"%(ndays,d0s,d1s))
store_emissions(day0=d0,dayn=d1)
print("Finished inversion")

END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

