samples_file="../data/training_data/training_eval_data-small.hdf5"
device=cuda

echo "Start training the network on $device. :)"

cmd="python train.py --train --device=$device --samples_file=$samples_file"

echo "Running command:"
echo $cmd

$cmd
