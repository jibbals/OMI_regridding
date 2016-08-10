#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N TestCheckReprocessed
#PBS -l walltime=00:08:00
#PBS -l mem=10000MB
#PBS -l cput=00:20:00
#PBS -l wd
#PBS -l ncpus=4
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ]; then
    echo "EG usage: qsub run_check_reprocessed.sh"
    exit 0
fi

# run the test methods
python - <<END
import tests
tests.check_reprocessed()
END


#------------------
# Append job diagnostics to qstats log file
#------------------
qstat -f $PBS_JOBID >> log.qstat

echo "Finished job ${PBS_JOBID}"

