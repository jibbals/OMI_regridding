#!/bin/bash

###
### Reprocess groups of 8 days 
###


if [ $# -lt 2 ]; then
    echo "EG usage: $0 9 50"
    echo "  To run 8 day clumps between days 9 and 50 (20050101 is day 1)"
    exit 0
fi

## Loop over 1 to $2 with step size of 8
for jj in `seq 1 8 $2`;
do
    if [ $jj -lt $1 ]; then
        continue
    fi
    echo "running qsub -v START=$jj run/reprocess_8dayclump.sh"
    qsub -v START=$jj -o logs/log.qsubreprocess$jj run/reprocess_8dayclump.sh 
done

