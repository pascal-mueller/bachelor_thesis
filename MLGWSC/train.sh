#!/bin/bash
input_dir=data/training_data
output_dir=data/trained_model
rm $output_dir/*hdf
./mock/examples/example_torch.py \
$input_dir/background-file.hdf $output_dir/background-events.hdf \
--verbose \
--output-training $output_dir \
--training-samples 5 5 \
--validation-samples 5 5 \
--train \
--weights data/trained_model/weights.pt
./mock/examples/example_torch.py \
$input_dir/foreground-file.hdf $output_dir/foreground-events.hdf \
--verbose \
--output-training $output_dir \
--training-samples 10 10 \
--validation-samples 10 10 \
--train \
--weights data/trained_model/weights.pt
