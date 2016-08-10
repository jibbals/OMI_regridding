#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N ReprocessCorrected
#PBS -l walltime=06:00:00
#PBS -l mem=20000MB
#PBS -l cput=32:00:00
#PBS -l wd
#PBS -l ncpus=8
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
if [ -z ${PBS_O_LOGNAME} ]; then
    echo "EG usage: qsub run_reprocess_corrected.sh"
    exit 0
fi

#------------------
# make sure correct python environment is going, then run python
# function
#------------------
#source activate jwg366_hdf5
source activate jwg_py3

# run the test methods
python - <<END
import reprocess
from datetime import datetime
day=datetime(2005,1,1)
#day9=datetime(2005,1,9)
#reprocess.create_omhchorp_1(day9)
reprocess.Reprocess_N_days(day, latres=0.25, lonres=0.3125, days=8, processes=8, remove_clouds=True)
reprocess.create_omhchorp_8(day, latres=0.25,lonres=0.3125)
END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat -f $PBS_JOBID >> log.qstat

echo "Finished job ${PBS_JOBID}"

