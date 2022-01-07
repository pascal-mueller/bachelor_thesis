samples_file="../data/training_data/200k.hdf5"
device=cuda

echo "Start training the network on $device. :)"

cmd="python train.py --train --device=$device --samples_file=$samples_file"

echo "Running command:"
echo $cmd

$cmd
