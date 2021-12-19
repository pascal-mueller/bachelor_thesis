import h5py
f = h5py.File('data/training_data/foreground-file.hdf', 'r')

datasets = f['H1']
ds_key = list(datasets.keys())[0]
dataset = f['H1'][ds_key]

print(dataset[0])

quit()
keys = f.keys() # H1, L1

datasets = f['H1']

number_of_datasets = len(datasets)

ds_keys = list(datasets.keys())

dataset = datasets[ds_keys[0]]

shape = dataset.shape

print(len(dataset))
