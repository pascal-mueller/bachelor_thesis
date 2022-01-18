weights_file="best_weights.pt"
dir=../data/test_data-big
output_dir=../data/output
device="cuda"
trigger_threshold=0.4
cluster_threshold=0.3

# Apply trained neural network to background file
inputfile=$dir/background_file.hdf5
output_file=$output_dir/background_events.hdf5
predictions_file=$output_dir/background_labels-big.hdf5

cmd="python apply.py $trigger_threshold $cluster_threshold $weights_file $inputfile $output_file $predictions_file --device=$device"

echo "Running"
echo " >>> $cmd"
$cmd


# Apply trained neural network to foreground file
inputfile=$dir/foreground_file.hdf5
output_file=$output_dir/foreground_events.hdf5
predictions_file=$output_dir/foreground_labels.hdf5

cmd="python apply.py $trigger_threshold $cluster_threshold $weights_file $inputfile $output_file $predictions_file --device=$device"

echo "Running"
echo " >>> $cmd"
$cmd
