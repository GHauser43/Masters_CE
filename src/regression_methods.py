import numpy as np
from sklearn.linear_model import Lasso


def Lasso_regularization():
    print('a')


def Ridge_regression(featureVector,target,lambda2):
    # preforms ridge regression, solving for coefficient values
    coefficient_values = (target @ featureVector.T) @ np.linalg.pinv(featureVector @ featureVector.T + lambda2 * np.identity(featureVector.shape[0]))
    
    return coefficient_values



# different regression methods
# update here and perform_regression() function if add new methods above
regression_methods_list = [
    'ridge'
    ]


def perform_regression(featureVector,
                       target,
                       regMethod,
                       lambda1,
                       lambda2,
                       tol):
    # performs regression to map featureVector to target
    # target is data[t] - data[t-1] for some time step t
    # uses lambda1, lambda2, tol when appropriate for a given 
    #    regression method regMethod
    # outputs coefficient_Values produced by regression

    if regMethod not in regression_methods_list:
        raise ValueError(f"Regression method '{regMethod}' not recognized. Available methods: {regression_methods_list}")  # noqa: E501

    if regMethod == 'ridge':
        coefficient_values = Ridge_regression(featureVector,target,lambda2) 
    

    return coefficient_values






def perform_regression_grid_search():
    print('c')
