import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
# explore skglm package for faster fitting
# look into sklearn precompute flag


def Lasso_regularization(featureVector,
                         target,
                         lambda1,
                         tol,
                         maxIter):
    # transform data into correct shape for sklearn lasso
    X_data = featureVector.T
    y_data = target.T
    # create instance of lasso regression
    lasso = Lasso(alpha=lambda1,
                  max_iter=maxIter,
                  tol=tol)
    # fit data
    lasso.fit(X_data, y_data)
    # extract coefficient values from lasso object
    coefficient_values = np.array(lasso.coef_)

    iterations_per_variable = lasso.n_iter_
    if maxIter in iterations_per_variable:
        print('convergence met: False')
    else:
        print('convergence met: True')

    return coefficient_values


def Ridge_regression(featureVector,
                     target,
                     lambda2):
    # preforms ridge regression, solving for coefficient values
    coefficient_values = (target @ featureVector.T) @ np.linalg.pinv(featureVector @ featureVector.T + lambda2 * np.identity(featureVector.shape[0]))  # noqa: E501

    return coefficient_values


def elasticNet_regularization(featureVector,
                              target,
                              lambda1,
                              lambda2,
                              tol,
                              maxIter):
    # transform data into correct shape for sklearn lasso
    X_data = featureVector.T
    y_data = target.T
    # convert lambda1 and lambda2 into sklearn ElasticNet paramaters
    # multiply by 2 to correct for sklearns implementation
    # see documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html  # noqa: E501
    alpha = lambda1 + 2.0 * lambda2
    l1_ratio = lambda1 / (lambda1 + 2.0 * lambda2)
    # create instance of elasticNet regression
    elasticNet = ElasticNet(alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=maxIter,
                            tol=tol)
    # fit data
    elasticNet.fit(X_data, y_data)
    # extract coefficient values from elasticNet object
    coefficient_values = np.array(elasticNet.coef_)

    iterations_per_variable = elasticNet.n_iter_
    if maxIter in iterations_per_variable:
        print('convergence met: False')
    else:
        print('convergence met: True')

    return coefficient_values


# different regression methods
# update here and perform_regression() function if add new methods above
regression_methods_list = ['lasso',
                           'ridge',
                           'elasticNet']


def perform_regression(featureVector,
                       target,
                       regMethod,
                       lambda1,
                       lambda2,
                       tol,
                       maxIter):
    # performs regression to map featureVector to target
    # target is data[t] - data[t-1] for some time step t
    # uses lambda1, lambda2, tol when appropriate for a given
    #    regression method regMethod
    # outputs coefficient_Values produced by regression

    if regMethod not in regression_methods_list:
        raise ValueError(f"Regression method '{regMethod}' not recognized. Available methods: {regression_methods_list}")  # noqa: E501

    if regMethod == 'ridge':
        coefficient_values = Ridge_regression(featureVector,
                                              target,
                                              lambda2)
    elif regMethod == 'lasso':
        coefficient_values = Lasso_regularization(featureVector,
                                                  target,
                                                  lambda1,
                                                  tol,
                                                  maxIter)
    elif regMethod == "elasticNet":
        coefficient_values = elasticNet_regularization(featureVector,
                                                       target,
                                                       lambda1,
                                                       lambda2,
                                                       tol,
                                                       maxIter)

    return coefficient_values


def perform_regression_grid_search():
    # TO-DO
    print('c')
