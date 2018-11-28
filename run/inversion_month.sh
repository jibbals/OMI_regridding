#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N Inversion
#PBS -l walltime=00:30:00
#PBS -l mem=15000MB
#PBS -l cput=00:40:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${MONTH} ]; then
    if [ $# -lt 1 ]; then
        echo "EG 1: $0 200505"
        echo "EG 2: qsub -o logs/inversion_200505 -v MONTH=200505 run/inversion_month.sh"
        echo "   to save E_new fOR 200505 "
        exit 0
    else
        echo "qsub -o logs/inversion_${1} -v MONTH=${1} run/inversion_month.sh"
        read -r -p "run that command? (N will run directly) [y/N] " response
        response=${response,,}    # tolower
        if [[ "$response" =~ ^(yes|y)$ ]]; then
            qsub -N invers${1} -o logs/inversion_${1} -v MONTH=${1} run/inversion_month.sh
            exit 0
        fi
        MONTH=$1
    fi
    
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
Inversion.store_emissions_month(month=d0)
print("Finished inversion")
ENDPython

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

