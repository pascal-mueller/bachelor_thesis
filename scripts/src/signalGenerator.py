import pycbc.detector, pycbc.waveform
import matplotlib.pyplot as plt
import numpy as np

class SignalGenerator:
    def __init__(self, file=None):
        np.random.seed(80)
        if file != None:
            self.dataset = file['signals']

        self.file = file 
        self.detector = pycbc.detector.Detector('H1')
        self.duration = 1.25 # Or 1.25s?
        self.sample_rate = 2048
        self.rng = np.random.default_rng()

    def generate(self, params):
        strains = np.zeros((len(params), 2560)) 

        for (i, param) in enumerate(params):
            # Take the injection time randomly in the LIGO O3a era              
            injection_time = self.rng.uniform(1238166018, 1253977218)

            # Generate the full waveform - VARIABLE LENGTH
            waveform_kwargs = param['waveform_kwargs']
            waveform = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            hp, hc = waveform

            # Note: The start_time of the waveform might be negative because
            # time axis is moved s.t. t=0 is at the actual merger.
            
            # Note: hp.get_sample_times()[0] can be negative since t=0 is at
            # merger time.
            
            # Place merger at injection_time
            start_time = injection_time + hp.get_sample_times()[0]
            hp.start_time = start_time
            hc.start_time = start_time

            # The waveform generator stops the waveform soon after the merger.
            # So we append some zeros to be sure that we actually have enough
            # long of a waveform to slice it later.
            hp.append_zeros(self.duration * self.sample_rate)
            hc.append_zeros(self.duration * self.sample_rate)

            # Project generated waveform onto detector. Note that this
            # projection depends on the start_time. That is because earth is
            # moving.
            right_ascension = param['right_ascension']
            declination = param['declination']
            pol_angle = param['pol_angle']
            strain = self.detector.project_wave(hp, hc, right_ascension,
                    declination, pol_angle)

            # The merger is where the maximal value is. We get the index of that
            # maximal value.
            # TODO: Is this really true for any merger?
            idx_merger = np.argmax(strain)

            # Cut 1.25s around merger.
            k = int(self.duration * self.sample_rate)
            k_half = int(0.5 * k)
            
            # Vary the merger by at most 0.2s seconds 
            k_vary = int(self.rng.uniform(-0.2, 0.2) * self.sample_rate)
            idx_merger += k_vary
            
            # Set start and end index for cutting window
            start = idx_merger - k_half
            end = idx_merger + k_half
            
            # If merger is close to left or right end of the strain, we might
            # not be able to center our cutting window around it but have to
            # respect the left or right border.
            if start < 0:
                start = 0 # Set start to left border
                end = start + k

            if end > len(strain):
                end = len(strain) # Set end to right border
                start = end - k

            # Cut the 1.25s time slice from out strain
            strain = strain[start : end]

            #idx = param['index']
            strains[i] = strain
            #self.dataset[idx] = strain

        idx = params[0]['index']
        k = len(params)

        if self.file != None:
            self.dataset[idx : idx + k] = strains
        else:
            return strains

