#!/bin/bash

if [ $# -lt 2 ]; then
    echo "EG usage: ${0} 2005 1"
    echo "    To submit jobs for January 2005"
    exit 0
fi

# Zero pad the day and month:
printf -v mm "%02d" $2

# run the amf utility independently for each day:
#
for i in `seq 1 31`;
do
    printf -v dd "%02d" $i 
    
    # First run the code to make satellite data -> csv files for lidort code
    # 
    echo "running: qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_$i run/omi_csv_oneday.sh"
    job1=`qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_${1}${mm}${dd} run/omi_csv_oneday.sh`
    
    
    # when that one finishes run PP_AMF code
    # 
    job1_id=${job1%%.*} # remove .r-man2 from string
    echo "running: qsub -W depend=afterok:${job1_id} -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.PP_amf_${1}${mm}${dd} run/PP_AMF_oneday.sh"
    qsub -W depend=afterok:${job1_id} -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.PP_amf_${1}${mm}${dd} run/PP_AMF_oneday.sh
done



