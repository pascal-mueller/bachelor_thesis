#!/bin/bash
dir=../data/test_data
../MLGWSC/mock/generate_data.py \
--data-set 1 \
--output-injection-file $dir/injection_parameters.hdf \
--output-foreground-file $dir/foreground_file.hdf \
--output-background-file $dir/background_file.hdf \
--seed 42 \
--start-offset 0 \
--duration 86400 \
--verbose \
--force
