#!/bin/bash
output_dir=../data/output
test_data_dir=../data/test_data-big

../MLGWSC/mock/evaluate.py \
--injection-file $test_data_dir/injection_parameters.hdf \
--foreground-events $output_dir/foreground_events.hdf \
--foreground-files $test_data_dir/foreground_file.hdf \
--background-events $output_dir/background_events.hdf \
--output-file $output_dir/eval-output.hdf \
--verbose \
--force
