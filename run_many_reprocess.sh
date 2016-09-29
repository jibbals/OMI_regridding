#!/bin/bash

for jj in `seq 1 8 121`;
do
    echo "running qsub -v START=$jj run_reprocess.sh"
    qsub -v START=$jj -o logs/log.qsubreprocess$jj run_reprocess.sh 
done

