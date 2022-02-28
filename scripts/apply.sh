weights_file="best_weights.pt"
dir='../data/test_data-big'
output_dir='../data/output'
device="cuda"
# Original von 22.2.22 abend
#trigger_threshold=0.4
#cluster_threshold=0.3

trigger_threshold=0.1
cluster_threshold=0.2 # 0.2 seconds window

# Apply trained neural network to background file
inputfile=$dir/background_file_whitened.hdf5
output_file=$output_dir/background_events_whitened.hdf5
predictions_file=$output_dir/background_labels-big_whitened.hdf5

cmd="python apply.py $trigger_threshold $cluster_threshold $weights_file $inputfile $output_file $predictions_file --device=$device"

#echo "Running"
#echo " >>> $cmd"
#$cmd


# Apply trained neural network to foreground file
inputfile=$dir/foreground_file_whitened.hdf5
output_file=$output_dir/foreground_events_whitened.hdf5
predictions_file=$output_dir/foreground_labels-big_whitened.hdf5

cmd="python apply.py $trigger_threshold $cluster_threshold $weights_file $inputfile $output_file $predictions_file --device=$device"

echo "Running"
echo " >>> $cmd"
$cmd
