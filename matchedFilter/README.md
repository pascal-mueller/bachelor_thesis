# Matched Filter

1. Generate template bank (a .hdf file) using CLI (e.g. `pycbc_brute_bank`)
2. Run search using `pycbc_inspiral`. 
    - Because we have a month of data, we need to split it up. 
    - We create chunk sizes of 4096 seconds (needs to be bigger than 512 
      seconds).
    - Need `injections.hdf`
3. We need to reformat triggers from 2. s.t. we can compare it to our NN results.
   - Throw away anything with SNR < 5

