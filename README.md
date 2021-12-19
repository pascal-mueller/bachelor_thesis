# Bachelor Thesis

The code can be found in `codebase/`. Read the README.md in there to get started.

# Questions
1. Why do we whiten the data i.e. noise and signal?
2. Why do we use SPD for "equal power contirbution"?

# Notes

## Paper 1: Training Strategies
- DL algorithms can generalize low SNR to high SNR but not
  vice versa.
=> No high SNR signals needed during training.
=> Fastest convergence is achieved when low SNR samples
   are provided early on.

- Network are sometimes unable to recover signals when
  false alarm probability < 10^-3.
  => Numerical issue, use USR (or double presicion?)

- ML search retains >= 97.5% of sensitivity of
  matched-filter search fown to FAR of 1 per month.

**Question:** false alarm prob. vs FAR?
**Answer:** False alarm probability of 10^-3 means 1 in
1000 pure noise samples falsely classified as containing
a signal.

!!! FAPs do not directly translate fo FARs on
continuous data streams. => The approproate question to
ask is how many false signals does the network identiy
per time interval of continous data, as opposed to how
many uncorrelated data chunks are falsely identified
as containing a signal.

- For statistics see paper 26
- Duration: 1s @ 2048 Hz
- p-score in [0,1]: Higher p-score => contains signal

- Randomly init. the network

- Efficienty: Fraction of correctly classified input
  sample containing a signal at a given FAP. (It
  drops to 0 beyond FAP of 10^-3 cause numerical issue.)
  => USR

- Network applied with sliding window 0.1s
