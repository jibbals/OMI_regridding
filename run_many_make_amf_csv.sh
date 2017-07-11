#!/bin/bash

#---------------------------------
# check inputs
# --------------------------------
if [ $# -lt 2 ]; then
    echo "EG usage: ./run_many_make_amf_csv.sh 2005 1"
    echo " To run for January 2005 "
    exit 0
fi


# run the things
#
for i in `seq 1 31`;
do
    echo "running: qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_$i run_make_amf_csv.sh"
    qsub -v DAY=$i,MONTH=$2,YEAR=$1 -o logs/log.make_amf_$i run_make_amf_csv.sh
done


