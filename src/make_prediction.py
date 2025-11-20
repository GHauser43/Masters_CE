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


def generate_prediction(prediction, delayTime_pts, featureVector, coefficient_values):

    # loop over number of predictions to be made
    for i in range(delayTime_pts + 1, prediction.shape[1]):
        # populate trajectoryHistory_current with previous trajectory values
        trajectoryHistory_current = prediction[:,i-delayTime_pts-1:i]
        # generate feature vector for trajectoryHistory_current
        featureVector_current = featureVector.construct_featureVector(trajectoryHistory_current)  # noqa: E501
        # pupulate index i in prediction array with prediction of trajectory
        prediction[:,i] = prediction_step(trajectoryHistory_current,
                        featureVector_current,
                        coefficient_values)[:,-1]

    return prediction




