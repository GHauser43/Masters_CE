# import libraries
import numpy as np
import argparse
import data_generation as dg
import feature_vector as fv


# input paramaters
def parse_args():
    parser = argparse.ArgumentParser(description="Generalized Next Generation Resivoir Computing code")  # noqa: E501
    # data generation
    parser.add_argument('--numIntegrator',
                        type=str,
                        default='rk4',
                        help='numerical integration method( scipy: RK23, RK45, DOP853, Radau, BDF, LSODA; self defined: rk4 )')  # noqa: E501
    parser.add_argument('--dt',
                        type=float,
                        default=0.0025,
                        help='time step for data generation')
    parser.add_argument('--system',
                        type=str,
                        default='Lorenz_63',
                        help='system of equations for data generation (Lorenz...)')  # noqa: E501
    # simmulation times
    parser.add_argument('--startTime',
                        type=float,
                        default=0,
                        help='time the system starts from')
    parser.add_argument('--warmupTime',
                        type=float,
                        default=2,
                        help='warmup time before training data')
    parser.add_argument('--trainTime',
                        type=float,
                        default=.01,
                        help='total time for training data')
    parser.add_argument('--testTime',
                        type=float,
                        default=1,
                        help='total time for testing data')
    parser.add_argument('--plotTime',
                        type=float,
                        default=1,
                        help='total time for plotting data')
    # TO-DO: plotTime <= testTime
    parser.add_argument('--errorTime',
                        type=float,
                        default=1,
                        help='total time for error calculation')
    # TO-DO: errorTime <= testTime
    return parser.parse_args()


def main():
    print('-----------------------------')
    print("Starting NGRC code")
    args = parse_args()
    # TO-DO: add feat vec construction inputs to argparse
    k = 2
    s = 1
    p = 2
    # TO-DO: print out all parser arguments at beginning output
    # TO-DO: add necessary checks for parser arguments
    # TO-DO: config file

    # time parameters
    dt = args.dt
    t0 = args.startTime
    warmupTime = args.warmupTime
    trainTime = args.trainTime
    testTime = args.testTime
    plotTime = args.plotTime
    errorTime = args.errorTime

    if plotTime > testTime:
        raise ValueError("plotTime must be less than or equal to testTime")
    if errorTime > testTime:
        raise ValueError("errorTime must be less than or equal to testTime")

    # discretized number of time points
    warmupTime_pts = int(warmupTime/dt)
    trainTime_pts = int(trainTime/dt)
    testTime_pts = int(testTime/dt)
    plotTime_pts = int(plotTime/dt)
    errorTime_pts = int(errorTime/dt)
    warmtrainTime_pts = warmupTime_pts + trainTime_pts
    totalTime_pts = warmupTime_pts + trainTime_pts + testTime_pts
    delayTime_pts = (k - 1) * s
    # TO-DO: add check that num warmup points > (k-1)*s

    # data generation parameters
    numIntegrator = args.numIntegrator
    system = args.system

    # generate data
    print('-----------------------------')
    print('data generation - started')
    trajectoryHistory, timeHistory, dim = dg.generate_data(numIntegrator,
                                                           system,
                                                           t0,
                                                           dt,
                                                           totalTime_pts)
    # splits data into training and testing blocks
    trajectoryHistory_train = dg.split_data(trajectoryHistory,
                                            warmupTime_pts - delayTime_pts,
                                            warmtrainTime_pts)
    timeHistory_train = dg.split_data(timeHistory,
                                      warmupTime_pts - delayTime_pts,
                                      warmtrainTime_pts)
    trajectoryHistory_test = dg.split_data(trajectoryHistory,
                                           warmtrainTime_pts,
                                           totalTime_pts)
    timeHistory_test = dg.split_data(timeHistory,
                                     warmtrainTime_pts,
                                     totalTime_pts)
    print('data generation - finished')

    # Construct feature vector
    print('-----------------------------')
    print('feature vector construction - started')
    # create instance of feature vector class
    featureVector = fv.FeatureVector(dim, k, s, p)
    # construct feature vector for training data
    featureVector_train = featureVector.construct_featureVector(trajectoryHistory_train)  # noqa: E501
    print('feature vector construction - finished')

    # TO-DO: Preform regression
    # TO-DO: Prediction
    # TO-DO: Error and plotting

    # print('-----------------------------')
    # print("Finished NGRC code")
    # print('-----------------------------')


if __name__ == "__main__":
    main()
