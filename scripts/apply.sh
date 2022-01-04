weights_file="best_weights.pt"
device="cuda"

# Apply trained neural network to background file
inputfile="../data/test_data-big/background_file.hdf"
outputfile="../data/output/background_events.hdf"
cmd="python apply.py $weights_file $inputfile $outputfile --device=$device"

echo "Running"
echo " >>> $cmd"
$cmd

exit 0
# Apply trained neural network to foreground file
inputfile="../data/test_data-big/foreground_file.hdf"
outputfile="../data/output/foreground_events.hdf"
cmd="python apply.py $weights_file $inputfile $outputfile --device=$device"

echo ""
echo "Running"
echo " >>> $cmd"
$cmd

