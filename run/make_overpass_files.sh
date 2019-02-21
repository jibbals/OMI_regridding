#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N make_overpass
#PBS -l walltime=06:00:00
#PBS -l mem=50000MB
#PBS -l cput=06:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

# If no job id then run this script with qsub
if [ -z ${PBS_JOBID} ]; then
    echo "submitting:  qsub -v run=tropchem $0"
    echo "submitting:  qsub -v run=new_emissions $0"
    qsub -N TCoverpass -o logs/qsub_make_overpass_tc.log -v run=tropchem $0
    qsub -N NEoverpass -o logs/qsub_make_overpass_ne.log -v run=new_emissions $0
    exit 0
fi

# if it was sent to qsub without an argument just do tropchem run
if [ -z run ]; then
    run=tropchem
fi

# run the tests script, send stdout and stderr to log.tests

python3 <<END
from utilities import GC_fio
GC_fio.make_overpass_output_files(run="$run")
END


echo "Finished job ${PBS_JOBID}"



