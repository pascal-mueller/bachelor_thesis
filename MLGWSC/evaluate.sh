#!/bin/bash
output_dir=data/eval_ouput
events_dir=data/trained_model
input_dir=data/training_data
./mock/evaluate.py \
--injection-file $input_dir/injection-file.hdf \
--foreground-events $events_dir/foreground-events.hdf \
--foreground-files $input_dir/foreground-file.hdf \
--background-events $events_dir/background-events.hdf \
--output-file $output_dir/eval-output.hdf \
--verbose \
--force
