# import libraries
import numpy as np
import argparse
import yaml
import data_generation as dg
import feature_vector as fv
import regression_methods as rm


# input paramaters
def parse_args():
    return 1


def main():
    print('-----------------------------')
    print("Starting NGRC code")

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
    # gets necessary parameters for given regression method
    # may need to update if add more regression methods
    if regMethod == 'lasso':
        lambda1 = config['lambda1']
        tol = config['tolerance']
    if regMethod == 'ridge':
        lambda2 = config['lambda2']
    if regMethod == 'elasticNet':
        lambda1 = config['lambda1']
        lambda2 = config['lambda2']
        tol = config['tolerance']

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
            raise ValueError("tol is required for lasso regression")
    if regMethod == 'ridge':
        if lambda2 is None:
            raise ValueError("lambda2 is required for ridge regression")
    if regMethod == 'elasticNet':
        if lambda1 is None:
            raise ValueError("lambda1 is required for elasticNet regression")
        if lambda2 is None:
            raise ValueError("lambda2 is required for elasticNet regression")
        if tol is None:
            raise ValueError("tol is required for elasticNet regression")

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
    # splits data into training and testing blocks
    trajectoryHistory_train, timeHistory_train, trajectoryHistory_test, timeHistory_test = dg.train_test_data_split(trajectoryHistory,  # noqa: E501
                          timeHistory,
                          warmupTime_pts,
                          warmtrainTime_pts,
                          delayTime_pts,
                          totalTime_pts)

 
    ### testing (temp)
    print(timeHistory)
    print('----------------------')
    print(timeHistory_train)
    print('----------------------')
    print(timeHistory_test)

    print('data generation - finished')

    # Construct feature vector
    print('-----------------------------')
    print('feature vector construction - started')
    # create instance of feature vector class
    featureVector = fv.FeatureVector(dim, k, s, p) # construct feature vector for training data
    featureVector_train = featureVector.construct_featureVector(trajectoryHistory_train)  # noqa: E501
    print('feature vector construction - finished')

    # Preform regression
    print('-----------------------------')
    print('preform regression - started')
    # creates target output for change in dynamics over one time step
    target = dg.split_data(trajectoryHistory, warmupTime_pts, warmtrainTime_pts) - dg.split_data(trajectoryHistory, warmupTime_pts-1, warmtrainTime_pts-1)  # noqa: E501
    # perform regression to get coefficient_values
    # that maps featureVector to target
    coefficient_values = rm.perform_regression(featureVector_train,
                                               target,
                                               regMethod,
                                               lambda1,
                                               lambda2,
                                               tol)
    print('coefficient_values:')
    print(coefficient_values) 
    # TO-DO: add regression grid search option?
    print('preform regression - finished')


    # TO-DO: Prediction
    print('-----------------------------')
    print('calculate prediction - started')




    print('calculate prediction - finished')
    # TO-DO: Error and plotting

    # print('-----------------------------')
    # print("Finished NGRC code")
    # print('-----------------------------')


if __name__ == "__main__":
    main()
