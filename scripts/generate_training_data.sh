#!/bin/bash

N_noise=2000
N_signal=2000
N_sample=2000
stride=400
filename=../data/training_data/training_eval_data-small.hdf5

cmd="mpiexec -n 4 python generate_training_data.py $N_noise $N_signal $N_sample $stride $filename"

echo "Running $cmd"

$cmd
