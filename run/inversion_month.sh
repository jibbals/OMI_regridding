#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N Inversion
#PBS -l walltime=00:15:00
#PBS -l mem=10000MB
#PBS -l cput=00:30:00
#PBS -l wd
#PBS -l ncpus=2
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${MONTH} ]; then
    echo "EG usage: qsub -o logs/log.inversion -v MONTH=200505 run/inversion_month.sh"
    echo "   to save E_new fOR 200501 "
    exit 0
fi

ymdstr="${MONTH}01"
echo ${ymdstr}

# run python code snippet:
python3 <<ENDPython
import Inversion
import utilities.utilities as util
from datetime import datetime as dt

# get start to finish dates:
d0=dt.strptime(str(${ymdstr}),'%Y%m%d')
d0s=${ymdstr}
d1=util.last_day(d0)
d1s=d1.strftime('%Y%m%d')

# run with some dialog
ndays=(d1-d0).days
print("Beginning inversion (%d days) from %s to %s"%(ndays,d0s,d1s))
Inversion.store_emissions(day0=d0,dayn=d1)
print("Finished inversion")
ENDPython

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

