#!/bin/bash

if [ $# -lt 1 ]; then
    echo "EG 1: $0 2005"
    echo "   to save E_new fOR 2005"
    exit 0
fi

for i in `seq 1 12`;
do
    printf -v mm "%02d" $i
    yyyymm=$1$mm

    qsub -N invers${yyyymm} -o logs/inversion_${yyyymm}.log -v MONTH=${yyyymm} run/inversion_month.sh
done

