import numpy as np

# make one step prediction
def prediction_step(trajectoryHistory_current,
                    featureVector_current,
                    coefficient_values):
    # trajectoryHistory_current - numpy array of dimensions (system) dim by _
    # featureVector_current - feature vector for trajectoryHistory_current
    # coefficient_values - coefficient values calculated in regression step

    prediction = trajectoryHistory_current + coefficient_values @ featureVector_current
    
    return prediction

