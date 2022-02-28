#!/bin/bash

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
plot_file=../data/output/sensitivity_plot_whitened.$current_time.png
eval_output_file=../data/output/eval-output_whitened.hdf
plot_script=../MLGWSC/mock/contributions/sensitivity_plot.py

cmd="python $plot_script --files $eval_output_file --output $plot_file --no-legend"

echo "Running $cmd"
$cmd

feh $plot_file

echo "Plot: $plot_file"
