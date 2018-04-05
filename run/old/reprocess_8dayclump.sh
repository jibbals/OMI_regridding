#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N ReprocessCorrected
#PBS -l walltime=04:00:00
#PBS -l mem=20000MB
#PBS -l cput=32:00:00
#PBS -l wd
#PBS -l ncpus=8
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${START} ]; then
    echo "EG usage: qsub -v START=9 run/reprocess_8dayclump.sh"
    exit 0
fi

# run the things in runscript.py
python3 runreprocess.py --start=$START &> logs/log.reprocess$START

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

