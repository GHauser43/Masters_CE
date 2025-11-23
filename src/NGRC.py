# import libraries
import numpy as np
import argparse
import yaml
import time
import data_generation as dg
import feature_vector as fv
import regression_methods as rm
import make_prediction as mp
import plot


def main():
    print('-----------------------------')
    print("Starting NGRC code")

    start_time = time.perf_counter()

    # TO-DO: add checks for parser arguments on assignment
    # TO-DO: make config file templates

    # loads config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    f = open(args.config, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    # time parameters
    dt = config['dt']
    t0 = config['startTime']
    warmupTime = config['warmupTime']
    trainTime = config['trainTime']
    testTime = config['testTime']
    plotTime = config['plotTime']
    errorTime = config['errorTime']

    # data generation parameters
    system = config['system']
    numIntegrator = config['numerical_integrator']

    # feature vector parameters
    k = config['k']
    s = config['s']
    p = config['p']

    # regression parameters
    regMethod = config['regression_method']
    # initialize variables
    lambda1 = None
    lambda2 = None
    tol = None
    maxIter = None
    # gets necessary parameters for given regression method
    # may need to update if add more regression methods
    if regMethod == 'lasso':
        lambda1 = config['lambda1']
        tol = config['tolerance']
        maxIter = config['maxIterations']
    if regMethod == 'ridge':
        lambda2 = config['lambda2']
    if regMethod == 'elasticNet':
        lambda1 = config['lambda1']
        lambda2 = config['lambda2']
        tol = config['tolerance']
        maxIter = config['maxIterations']

    # discretize time points
    warmupTime_pts = int(warmupTime/dt)
    trainTime_pts = int(trainTime/dt)
    testTime_pts = int(testTime/dt)
    plotTime_pts = int(plotTime/dt)
    errorTime_pts = int(errorTime/dt)
    warmtrainTime_pts = warmupTime_pts + trainTime_pts
    totalTime_pts = warmupTime_pts + trainTime_pts + testTime_pts
    delayTime_pts = (k - 1) * s

    # check necessary conditions for program to run
    # time values are logical
    if plotTime > testTime:
        raise ValueError("plotTime must be less than or equal to testTime")
    if errorTime > testTime:
        raise ValueError("errorTime must be less than or equal to testTime")
    if warmupTime_pts <= (k-1)*s:
        raise ValueError("required that WarmupTime_pts > (k-1)*s, increase warmupTime")  # noqa: E501
    # regression methods have needed varaibles
    if regMethod == 'lasso':
        if lambda1 is None:
            raise ValueError("lambda1 is required for lasso regression")
        if tol is None:
            raise ValueError("tolerance is required for lasso regression")
        if maxIter is None:
            raise ValueError("maxIterations is required for lasso regression")
    if regMethod == 'ridge':
        if lambda2 is None:
            raise ValueError("lambda2 is required for ridge regression")
    if regMethod == 'elasticNet':
        if lambda1 is None:
            raise ValueError("lambda1 is required for elasticNet regression")
        if lambda2 is None:
            raise ValueError("lambda2 is required for elasticNet regression")
        if tol is None:
            raise ValueError("tolerance is required for elasticNet regression")
        if maxIter is None:
            raise ValueError("maxIterations is required for elasticNet regression")  # noqa: E501

    # output parameter values
    print('-----------------------------')
    print('time parameters')
    print('  dt:         ', dt)
    print('  startTime:  ', t0)
    print('  warmupTime: ', warmupTime)
    print('  trainTime:  ', trainTime)
    print('  testTime:   ', testTime)
    print('  plotTime:   ', plotTime)
    print('  errorTime:  ', errorTime)
    print('data generation parameters')
    print('  system:               ', system)
    print('  numerical_integrator: ', numIntegrator)
    print('feature vector parameters')
    print('  k: ', k)
    print('  s: ', s)
    print('  p: ', p)
    print('regression parameters')
    print('  regression_method: ', regMethod)
    if regMethod == 'lasso':
        print('  lambda1:   ', lambda1)
        print('  tolerance: ', tol)
    if regMethod == 'ridge':
        print('  lambda2: ', lambda2)
    if regMethod == 'elasticNet':
        print('  lambda1:   ', lambda1)
        print('  lambda2:   ', lambda2)
        print('  tolerance: ', tol)

    # generate data
    print('-----------------------------')
    print('data generation - started')
    trajectoryHistory, timeHistory, dim = dg.generate_data(numIntegrator,
                                                           system,
                                                           t0,
                                                           dt,
                                                           totalTime_pts)

    # variance in generated data, used later for error calculations
    data_variance = np.var(trajectoryHistory)

    # splits data into training and testing blocks
    trajectoryHistory_train, timeHistory_train, trajectoryHistory_test, timeHistory_test = dg.train_test_data_split(trajectoryHistory,  # noqa: E501
                          timeHistory,
                          warmupTime_pts,
                          warmtrainTime_pts,
                          delayTime_pts,
                          totalTime_pts)
    print('data generation - finished')

    # Construct feature vector
    print('-----------------------------')
    print('feature vector construction - started')
    # create instance of feature vector class
    # construct feature vector for training data
    featureVector = fv.FeatureVector(dim, k, s, p)

    featureVector_train = featureVector.construct_featureVector(trajectoryHistory_train[:, :-2])  # noqa: E501
    print('feature vector construction - finished')

    # Preform regression
    print('-----------------------------')
    print('preform regression - started')
    # creates target output for change in dynamics over one time step
    target = trajectoryHistory_train[:, 2:-1]-trajectoryHistory_train[:, 1:-2]

    # perform regression to get coefficient_values
    # that maps featureVector to target
    coefficient_values = rm.perform_regression(featureVector_train,
                                               target,
                                               regMethod,
                                               lambda1,
                                               lambda2,
                                               tol,
                                               maxIter)
    print('coefficient_values:')
    print(coefficient_values)
    # TO-DO: add regression grid search option?
    print('preform regression - finished')

    # make one prediction step to test training fit accuracy
    print('-----------------------------')
    print('calculate training fit error - started')

    prediction_train = mp.prediction_step(trajectoryHistory_train[:, 1:-2],
                                          featureVector_train,
                                          coefficient_values)

    difference_train = prediction_train - trajectoryHistory_train[:, 2:-1]
    NRMSE_train = np.sqrt(np.mean(difference_train**2)/data_variance)
    print(f'training NRMSE:  {NRMSE_train:.4e}')

    print('calculate training fit error - finished')

    # TO-DO: Prediction
    print('-----------------------------')
    print('calculate prediction - started')

    # initialize storage for prediction
    prediction = np.zeros([dim, testTime_pts + delayTime_pts + 1])
    # copy over starting value with delay
    prediction[:, 0:delayTime_pts + 1] = trajectoryHistory_train[:, -delayTime_pts - 1:]  # noqa: E501
    # generates full prediction, removes initial delay taps
    prediction = mp.generate_prediction(prediction,
                                        delayTime_pts,
                                        featureVector,
                                        coefficient_values)[:, delayTime_pts:]

    print('calculate prediction - finished')

    print('-----------------------------')
    print('calculate prediction error - started')

    # calculates error for generated predictions, first column in array is IC
    difference_test = prediction[:, 1:errorTime_pts] - trajectoryHistory_test[:, 1:errorTime_pts]  # noqa: E501
    NRMSE_test = np.sqrt(np.mean(difference_test**2)/data_variance)
    print(f'testing NRMSE:  {NRMSE_test:.4e}')

    print('calculate prediction error - finished')

    # plotting
    print('-----------------------------')
    print('generate plot - started')

    plot.make_plot(trajectoryHistory,
                   timeHistory,
                   prediction_train,
                   timeHistory_train[2:-1],
                   prediction,
                   timeHistory_test,
                   dim)

    print('generate plot - finished')

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('-----------------------------')
    print("Finished NGRC code")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print('-----------------------------')


if __name__ == "__main__":
    main()
