#!/bin/bash

# Run this script to find missing files within a year
if [ $# -lt 1 ]; then
    echo "EG usage: $0 2005"
    echo "   to see if any missing files are in 2005"
    exit 0
fi

year=$1

# print missing statement
function missing {
    echo "$year $month $day missing $1"
}

# loop over months (-w adds zero padding)
for month in `seq -w 1 12`; do
    #how many days in month
    dim=$( cal $month $year | awk 'NF {DAYS = $NF}; END {print DAYS}' )
    # loop over days
    for day in $( seq -w 1 $dim ); do
        
        ### Check for model files 
        ###
        
        ### Check for my outputs
        ###
        
        ### CHECK FOR PP OUTPUTS
        
        ### FILTERS FROM SATELLITE PRODUCTS
        ###
        
        ## look for OMNO2d
        # file looks like: Data/OMNO2d/data/OMI-Aura_L3-OMNO2d_2008m1231_v003....he5
        # easy to use ls due to wildcard, pipe output to null, ls returns true if any results
        if ! ls Data/OMNO2d/data/OMI-Aura_L3-OMNO2d_${year}m${month}${day}* 1> /dev/null 2>&1; then
            missing "OMNO2d"
        fi
        
        ## look for mod14a files
        # Data/MOD14A1_D_FIRE/2006/MOD14A1_D_FIRE_2006-01-01.CSV
        # No wildcard needed so easy to run file existance check
        if [[ ! -f Data/MOD14A1_D_FIRE/$year/MOD14A1_D_FIRE_$year-$month-$day.CSV ]]; then
            missing "MOD14A1"
        fi
        
        ## Look for omaeruv files
        # Data/OMAERUVd/OMI-Aura_L3-OMAERUVd_2004m1022_v003-2017m0821t131222.he5
        if ! ls Data/OMAERUVd/OMI-Aura_L3-OMAERUVd_${year}m${month}${day}* 1> /dev/null 2>&1; then
            #echo "OMAERUVd is missing date $year $month $day"
            missing "OMAERUVd"
        fi
        
    done
done


