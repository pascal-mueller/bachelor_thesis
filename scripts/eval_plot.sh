#!/bin/bash

eval_output_file=../data/output/eval-output.hdf
plot_file=../data/output/sensitivity_plot.png
plot_script=../MLGWSC/mock/contributions/sensitivity_plot.py

cmd="python $plot_script --files $eval_output_file --output $plot_file"

echo "Running $cmd"
$cmd

feh ../data/output/sensitivity_plot.png
