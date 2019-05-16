#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N scriptHR
#PBS -l walltime=02:00:00
#PBS -l mem=200000MB
#PBS -l cput=02:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

if [ $# -lt 1 ]; then
    if [ -z ${fname} ]; then 
        echo "EG: $0 tests.py"
        echo "    will run tests.py on qsub normal queue"
        exit 0
    fi
else
    echo "submitting:  qsub -v fname=$1 $0"
    echo "  will run: python3 ${1} &> logs/${1%.*}.log"
    qsub -N $1 -o logs/qsub_${1%.*}.log -v fname=$1 $0
    exit 0
fi

# run the tests script, send stdout and stderr to log.tests

echo "Running: python3 ${fname} &> logs/${fname%.*}"
python3 ${fname} &> logs/${fname%.*}.log

echo "Finished job ${PBS_JOBID}"



