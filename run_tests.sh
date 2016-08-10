#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N TestsRun
#PBS -l walltime=01:00:00
#PBS -l mem=25000MB
#PBS -l cput=04:00:00
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

#------------------
# make sure correct python environment is going, then run python
# function
#------------------
#source activate jwg366_hdf5
source activate jwg_py3

# run the test methods
#python - <<END
#import test_fio
#test_fio.test_apriori()
#END

# run the script
python tests.py

echo "Finished job ${PBS_JOBID}"

