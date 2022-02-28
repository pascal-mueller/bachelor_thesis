import h5py
import numpy as np
import csv

segments_file = open('segments.csv')
segments = csv.reader(segments_file)

header = next(segments)

chirp_distance = []
coa_phase = []
dec = []
distance = []
inclination = []
mass1 = []
mass2 = []
mchirp = []
polarization = []
q = []
ra = []
tc_new = []

with h5py.File('injections.hdf', 'r') as f:
    tc = f['tc'][:]

    for i, start, end in segments:
        start = int(start)
        end = int(end)

        # Get indices of times in segment
        idx = np.where(np.logical_and(tc>=start, tc<=end))

        chirp_distance.extend(f['chirp_distance'][idx])
        coa_phase.extend(f['coa_phase'][idx])
        dec.extend(f['coa_phase'][idx])
        distance.extend(f['distance'][idx])
        inclination.extend(f['inclination'][idx])
        mass1.extend(f['mass1'][idx])
        mass2.extend(f['mass2'][idx])
        mchirp.extend(f['mchirp'][idx])
        polarization.extend(f['polarization'][idx])
        q.extend(f['q'][idx])
        ra.extend(f['ra'][idx])
        tc_new.extend(f['tc'][idx])

with h5py.File('correct_injections.hdf', 'w') as ff:
    ff.create_dataset("chirp_distance", (len(chirp_distance), ), dtype='d')
    ff.create_dataset("coa_phase", (len(coa_phase), ), dtype='d')
    ff.create_dataset("dec", (len(dec), ), dtype='d')
    ff.create_dataset("distance", (len(distance), ), dtype='d')
    ff.create_dataset("inclination", (len(inclination), ), dtype='d')
    ff.create_dataset("mass1", (len(mass1), ), dtype='d')
    ff.create_dataset("mass2", (len(mass2), ), dtype='d')
    ff.create_dataset("mchirp", (len(mchirp), ), dtype='d')
    ff.create_dataset("polarization", (len(polarization), ), dtype='d')
    ff.create_dataset("q", (len(q), ), dtype='d')
    ff.create_dataset("ra", (len(ra), ), dtype='d')
    ff.create_dataset("tc", (len(tc_new), ), dtype='d')

    ff['chirp_distance'][:] = chirp_distance
    ff['coa_phase'][:] = coa_phase
    ff['dec'][:] = dec
    ff['distance'][:] = distance
    ff['inclination'][:] = inclination
    ff['mass1'][:] = mass1
    ff['mass2'][:] = mass2
    ff['mchirp'][:] = mchirp
    ff['polarization'][:] = polarization
    ff['q'][:] = q
    ff['ra'][:] = ra
    ff['tc'][:] = tc_new

