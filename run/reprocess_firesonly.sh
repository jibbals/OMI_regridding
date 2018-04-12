#!/bin/bash
#PBS -P m19
#PBS -q normal
#PBS -N RP_fires
#PBS -l walltime=00:20:00
#PBS -l mem=1000MB
#PBS -l cput=00:20:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -j oe

# Make omhchorp product for a few days before 20050101 in order to make firemask
for i in `seq 22 31`; do
dstr=200412${i}

# run the day
python3 << END
from datetime import datetime
import reprocess
date=datetime.strptime(str(${dstr}),"%Y%m%d")

reprocess.create_omhchorp_justfires(date)

END

done
