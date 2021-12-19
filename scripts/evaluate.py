"""
    See: Paper 26

    FAR & SENSITIVITY

    FAR: Any event that is reported by the search needs to be
          assigned a FAR to express the confidence in its detection.
          For a given value R, the FAR is the number of false alarms
          with a ranking statistic of at least R per unit time.
          To estimate it on mock data, the number of false detections exceeding
          a ranking statistic R is divided by the duration of the analyzed data.

    SENSITIVITY: The ability of the search to recover signals is quantified
                 by the sensitivity which is a function of the FAR lower
                 bound. It is often given in terms of the fraction of recovered
                 injections.

                 Estimated by: #found_inj / #total_inj
                 whereas
                    #found inj: detected injections with a FAR ≤ F

                See also: pycbc::volume_montecarlo

    Input: List of candidate events ranked by some ranking statistic which
    signifies how likely the data is to contain a signal.

    1. Get list of candidate events ranked by R computed by our algorithm
    2. Get input data with known injections
    3. Using those two files, we determine which signals were found and which
       weren't. We call the later "false alarms".


    Files:
        - foreground_file: Result from our algorithm for "real" input
        - background_file: Result from our algorithm for noise only
        - injection_file: "real" input
"""

# The inj_file specifies injection for the whole month.
# Our foreground file might only be like a week or so. So we need to figure out
# which injections we actually made.
def find_injection_times(inj_file):
    duration = o

    # Compute total duration
    
    # Read injection times from inj_file

    # Find injections

    # Returns the injection in inj_file which were used to inject a signal
    # into the foreground file.


if __name__=='__main__':
    duration, idxs = find_injection_times(inj_file)

