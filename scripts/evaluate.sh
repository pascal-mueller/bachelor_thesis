#!/bin/bash
output_dir=../data/output
test_data_dir=../data/test_data-big

../MLGWSC/mock/evaluate.py \
--injection-file $test_data_dir/injection_parameters.hdf5 \
--foreground-events $output_dir/foreground_events_whitened.hdf5 \
--foreground-files $test_data_dir/foreground_file_whitened.hdf5 \
--background-events $output_dir/background_events_whitened.hdf5 \
--output-file $output_dir/eval-output_whitened.hdf \
--verbose \
--force
