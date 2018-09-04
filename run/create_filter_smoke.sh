#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N smokefilter
#PBS -l walltime=4:00:00
#PBS -l mem=10000MB
#PBS -l cput=4:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

if [ -z ${PBS_O_LOGNAME} ] || [ -z ${year} ]; then
    echo "example:"
    echo "qsub -o logs/smokemask_2005 -v year=2005 $0"
    echo "#    creates Data/OMAERUv/smokemask_2005.h5"
    exit 0
fi

python3 <<ENDpy

from utilities import fio
from datetime import datetime

fio.make_smoke_mask_file(datetime($year,1,1))

ENDpy

echo "Finished job ${PBS_JOBID}"



