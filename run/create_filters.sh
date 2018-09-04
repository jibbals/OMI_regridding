#!/bin/bash

if [ $# -lt 1 ]; then
    echo "example:"
    echo "$0 2005"
    echo "    runs create filter scripts for 2005"
    exit 0
fi


echo "submit job for anthro mask:"
qsub -N anthro$1 -o logs/anthromask_$1 -v year=$1 run/create_filter_anthro.sh

echo "submit job for fire mask:"
qsub -N fire$1 -o logs/firemask_$1 -v year=$1 run/create_filter_fire.sh

echo "submit job for smoke mask:"
qsub -N smoke$1 -o logs/smokemask_$1 -v year=$1 run/create_filter_smoke.sh
