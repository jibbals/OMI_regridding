#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N smearfilter
#PBS -l walltime=1:00:00
#PBS -l mem=10000MB
#PBS -l cput=1:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

if [ -z ${PBS_O_LOGNAME} ] || [ -z ${year} ]; then
    if [ $# -lt 1 ]; then
        echo "#example to create Data/smearmask_2005.h5"
        echo "$0 2005"
        echo "OR"
        echo "qsub -o logs/smearmask_2005 -v year=2005 $0"
        
    else
        echo "qsub -o logs/smearmask_$1 -v year=$1 $0"
        qsub -o logs/smearmask_$1 -N smear$1 -v year=$1 $0
    fi
    exit 0
fi

python3 <<ENDpy

from utilities import masks
from datetime import datetime

masks.make_smear_mask_file($year)

ENDpy

echo "Finished job ${PBS_JOBID}"



