# Classification

## Goal
The goal is to decide if/where there is a singal in a given strain.
There might be more than one signal. The goal is to get the time of the merger.

## Workflow
The basic workflow is as follow:

1. Generate training data
2. Generate validation data
3. Train model
4. Use trained model to classify actual data

## Generating Training & Validation Data
Both training and validation data are generated using the `generate_data()`
function.

It consists of either 1 second of pure noise or 1 second of signal+noise around
the merger.

Data is being generated using pyCBC. We create `N_signal` GW signal samples and
`N_noise` noise samples. The noise is gaussian noise.

All data will be projected onto the `H1` detector. This is needed because what
we actually measure in practice is a signal at some specific detector. The same
signal looks different for each detector mainly because they are at different
locations on earth and thus also different noise.

### Generating Noise
To generate the noise, we use the `aLIGOZeroDetHighPower` PSD. This particular
PSD function takes the three arguments `length`, `delta_f` and `low_freq_cutoff`
- `length`: Length of the frequency series in samples. (Number of samples)
- `delta_f`: Frequency resolution of the frequency series.("stepsize" in hz)
- `low_freq_cutoff`: Frequencies below this value are set to zero. 

We evaluate it for `length = 1281`, `delta_f = 4.0/5.0` and
`low_freq_cufott = 18.0`.

### Generating Waveform
We use pyCBC to generate a waveform [1]. Masses are drawn uniform between 10.0 and
50.0 solar masses. `coa_phase`, `inclination` and `pol_angle` are drawn from an
uniform distribution between 0 and 2pi.

Each waveform has the properties:
- `delta_t`: 1.0 / 2048 - The time step used to generate the waveform (in s).
- `f_lower`: 18.0 - The starting frequency of the waveform (in Hz).
- `approximant`: IMRPhenomD
- `mass1`: max(masses)
- `mass2`: min(masse)
- `coa_phase`: Coalesence phase of the binary (in rad).
- `inclination`: nclination (rad), defined as the angle between the orbital
angular momentum L and the line-of-sight at the reference frequency.

We also need the `declination`, `right_ascension` and `pol_angle` to project the
waveform onto the detector.

### Labels
We use a 2D label. `[1.0, 0.0]` for waveforms and `[0.0, 1.0]` for noise.

### Inject waveform into noise
1. Choose `injection_time` uniformly between 20s and 40s.
2. Move `start_time` by `injection_time`.
3. Set start time of `hp` and `hc` to `start_time`.
4. Project to detector
5. Move merger by 0.5-0.7s to make model more robust (because it won't learn the
merger time)
6. Get 1.25s around the merger. We will only use 1s but cause whitening border
effects we add 0.25s.
7. Rescale using SNR (WHAT WHAT WHAT?)
8. Inject signal into noise by simple addition.

### Whitening
We use pyCBC's timeseries whiten() function.

# References
[1] https://pyc hbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
