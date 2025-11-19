import numpy as np

# make one step prediction
def prediction_step(trajectoryHistory_current, featureVector_current, coefficient_values):
    # documentation

    prediction = trajectoryHistory_current + coefficient_values @ featureVector_current
    return prediction

