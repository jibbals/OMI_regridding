#!/bin/bash

if [ $# -lt 1 ]; then
    echo "EG usage: ${0} 2005"
    echo "    To submit jobs for 2005"
    exit 0
fi

for mm in `seq 1 12`;
do
    run/make_pp_amf_month.sh $1 $mm
done



