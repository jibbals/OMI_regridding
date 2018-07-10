#!/bin/bash

##
## This script reprocesses one year of swath files
## into omhchorp dataset
##

if [ $# -lt 1 ]; then
    echo "EG usage: $0 2005"
    echo " To reprocess and get omhchorp for july 2005"
    exit 0
fi

printf -v mm "%02d" $2

for i in `seq 1 12`;
do
    ./run/reprocess_month.sh $1 $i
done

