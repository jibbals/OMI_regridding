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
    echo "running: qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.PP_amf_${1}${mm}${dd} run/PP_AMF_oneday.sh"
    qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.PP_amf_${1}${mm}${dd} run/PP_AMF_oneday.sh
done



