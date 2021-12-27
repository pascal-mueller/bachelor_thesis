#!/bin/bash
dir=../data/training_data
./mock/generate_data.py \
--data-set 1 \
--output-injection-file $dir/injection-file.hdf \
--output-foreground-file $dir/foreground-file.hdf \
--output-background-file $dir/background-file.hdf \
--seed 42 \
--start-offset 0 \
--duration 660 \
--verbose \
--force
