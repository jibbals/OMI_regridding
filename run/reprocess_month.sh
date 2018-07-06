#!/bin/bash

##
## This script reprocesses one month of shapefactors and swath files
## into omhcho_1 data
##

if [ $# -lt 2 ]; then
    echo "EG usage: run/reprocess_month.sh 2005 7"
    echo " To reprocess and get omhchorp for july 2005"
    exit 0
fi

printf -v mm "%02d" $2

for i in `seq 1 31`;
do
    printf -v dd "%02d" $i
    qsub -N RP$1${mm}${dd} -v dstr=${1}${mm}${dd} -o logs/reprocess${1}${mm}${dd}.log run/reprocess_day.sh
    echo "creating omhchorp for ${1}${mm}${dd}"
done

