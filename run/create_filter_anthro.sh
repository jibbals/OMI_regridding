#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N anthrofilter
#PBS -l walltime=2:00:00
#PBS -l mem=10000MB
#PBS -l cput=20:00:00
#PBS -l wd
#PBS -l ncpus=14
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

if [ -z ${PBS_O_LOGNAME} ] || [ -z ${year} ]; then
    if [ $# -lt 1 ]; then
        echo "#example to create Data/OMNO2d/anthromask_2005.h5"
        echo "qsub -o logs/anthro_2005 -v year=2005 $0"
        
    else
        echo "qsub -o logs/anthromask_$1 -v year=$1 $0"
        qsub -o logs/anthromask_$1 -N anthro$1 -v year=$1 $0
    fi
    exit 0
fi

python3 <<ENDpy

from utilities import fio
from datetime import datetime

fio.make_anthro_mask_file(datetime($year,1,1),max_procs=14)

ENDpy

echo "Finished job ${PBS_JOBID}"



