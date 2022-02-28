# Signal Generation

- m1, m2 in [10, 50]M   m1 >= m2
- phi\_0 in [0, 2pi]
- Generate 5 phases for each pair (m1, m2)
- Distance: scaled to target\_snr
- Sky position: Overhead H1
- Inclination = 0
- Polarization = 0
- Low freq. cutoff = 20 Hz
- Sample rate = 2048 Hz
- Approximant = SEOBNRv4\_opt

Procedure:
1. Generate signal
2. Shift position of merger by uniform([-1, 1]) s
3. Project onto detector
4. Whiten signal
5. Scale waveform to optimal SNR of 1. See eq. 1 for optimal SNR.
6. Extract 1s window s.t. original merger time (not shifted one) is located at 0.7 s 

7. Generate noise from same PSD used to whiten signal in step 4.
8. Whiten noise using PSD used to create it.

Training:
- The whitened signals and noise samples are combined during training.
- This allows us to rescale the signals at runtime to a desired strength.
- Adam optimizer (beta1 = 0.9, beta2 = 0.999, eps = 10^(-8))
- Learning rate = 10^(-5)
- Loss: reg\_BCELoss(eps=10^-6)
- Labels: (1,0) for signal, (0,1) for pure noise
- Batch size = 32
- Metric: Efficiency (instead of accuracy).
- Output: p-score in [0,1]

Efficiency:
The efficiency is the true-positive probaility at a **fixed** false-positive
probability (FAP). To do so, we sort the p-score outpus of the network on the
**noise** efficiency (for me evaluation) set and use the x-th largest as a
threhold, where choose x  = (eq. 3)


Training set:
- 20'000 unique combinations of component masses
- Generate 5 waveform with random coalescence phases for each 20'000 mass pairs
=> 100'000 individual signals

- Generate 200'000 independent noise samples.
- Use 100'000 with signals and 100'000 as pure noise.
=> Total 200'000 samples


