echo "Whitening background file"
inputfile="../data/test_data-big/background_file.hdf5"
outputfile="../data/test_data-big/background_file_whitened.hdf5"

python ../MLGWSC/mock/contributions/whiten.py \
$inputfile \
$outputfile \
--verbose \
--low-frequency-cutoff 18 \
