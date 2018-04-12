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

    # check if file already exists:
    outfile=Data/pp_amf/tropchem/amf_$1$mm$dd.csv
    if [ -f $outfile ]; then
        # Skip if file already exists
        echo "$outfile exists already"
        continue
    else 
        echo "submitting jobs to make $outfile"
    fi
    
    # First run the code to make satellite data -> csv files for lidort code
    # 
    job1=`qsub -N omicsv${mm}${dd} -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/make_amf_${1}${mm}${dd}.log run/omi_csv_oneday.sh`
    echo "running: omi_csv_oneday.sh (${job1})"
    
    
    # when that one finishes run PP_AMF code
    # 
    job1_id=${job1%%.*} # remove .r-man2 from string
    job2=`qsub -W depend=afterok:${job1_id} -N ppamf${mm}${dd} -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/PP_amf_${1}${mm}${dd}.log run/PP_AMF_oneday.sh`
    echo "running: PP_AMF_oneday.sh (${job2}) after ${job1} completes"
done



