#!/bin/bash
#PBS -P m19
#PBS -q express
#PBS -N RunInversion
#PBS -l walltime=00:30:00
#PBS -l mem=10000MB
#PBS -l cput=00:50:00
#PBS -l wd
#PBS -l ncpus=4
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
#if [ -z ${PBS_O_LOGNAME} ]; then # || [ -z ${STARTYEAR} ] || [ -z ${STARTMONTH} ] || [ -z ${ENDYEAR} ] || [ -z ${ENDMONTH} ]; then
#    echo "EG usage: qsub -v STARTYEAR=2005,STARTMONTH=9,ENDYEAR=2006,ENDMONTH=3 run_inversion.sh"
#    echo "   run from 2005 startmonth to startmonth+nmonths "
#    exit 0
#fi

STARTYEAR=2005
STARTMONTH=1
ENDYEAR=2005
ENDMONTH=2

python3 Inversion.py &> runInversion.out
#python3 <<END
#import Inversion
#import utilities.fio # For saving outputs to hd5
#from datetime import datetime

# For now just check output
#Inversion.check_against_MEGAN()
#date0=datetime($STARTYEAR,$STARTMONTH,1)
#date1=datetime($ENDYEAR,$ENDMONTH,1)
#Emissions_series(day=date0)

#END

#------------------
# Append job diagnostics to qstats log file
#------------------
qstat $PBS_JOBID >> logs/log.qstat

echo "Finished job ${PBS_JOBID}"

