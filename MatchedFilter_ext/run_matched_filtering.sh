#!/bin/bash
# Purpose: Read Comma Separated CSV File
# Author: Vivek Gite under GPL v2.0+
# ------------------------------------------
INPUT=segments.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read i start end 
do
	bsub -n 16 -W 02:00 ./run_search.sh $start $end
done < $INPUT
IFS=$OLDIFS
