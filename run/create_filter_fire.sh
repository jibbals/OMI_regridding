#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N firefilter
#PBS -l walltime=4:00:00
#PBS -l mem=50000MB
#PBS -l cput=24:00:00
#PBS -l wd
#PBS -l ncpus=14
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

if [ -z ${PBS_O_LOGNAME} ] || [ -z ${year} ]; then
    echo "example:"
    echo "qsub -o logs/firemask_2005 -v year=2005 create_filter_fire.sh"
    echo "    create Data/MOD14A1/firemask_2005.h5"
    exit 0
fi

python3 <<ENDpy

from utilities import fio
from datetime import datetime

fio.make_fire_mask_file(datetime($year,1,1),max_procs=14)

ENDpy

echo "Finished job ${PBS_JOBID}"



