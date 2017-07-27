#!/bin/bash

#---------------------------------
# check inputs
# --------------------------------
if [ $# -lt 2 ]; then
    echo "EG usage: $0 2005 1"
    echo " To run for January 2005 "
    exit 0
fi


# run the things
#
printf -v mm "%02d" $MONTH # 01 to 12 (leading zeros)
for i in `seq 1 31`;
do
    printf -v dd "%02d" $i # leading zeros added
    echo "running: qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_$i run/omi_csv_oneday.sh"
    qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_${1}${mm}${dd} run/omi_csv_oneday.sh
done


