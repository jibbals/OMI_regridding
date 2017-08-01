#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N Tests
#PBS -l walltime=00:30:00
#PBS -l mem=25000MB
#PBS -l cput=02:00:00
#PBS -l wd
#PBS -l ncpus=4
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ]; then
    echo "EG usage: qsub run_tests.sh"
    exit 0
fi

# run the tests script, send stdout and stderr to log.tests
echo "Running tests.py &> logs/log.tests"
python3 tests.py &> logs/log.tests

echo "Finished job ${PBS_JOBID}"

