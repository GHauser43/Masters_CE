# import libraries
import data_generation as dg
import numpy as np
import argparse


### input paramaters
def parse_args():
    parser = argparse.ArgumentParser(description="Generalized Next Generation Resivoir Computing code")
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
                        help='system of equations for data generation (Lorenz...)')
    # simmulation times
    parser.add_argument('--startTime',
                        type=float,
                        default=0,
                        help='time the system starts from')
    parser.add_argument('--warmupTime',
                        type=float,
                        default=5,
                        help='warmup time before training data')
    parser.add_argument('--trainTime',
                        type=float,
                        default=10,
                        help='total time for training data')
    parser.add_argument('--testTime',
                        type=float,
                        default=10,
                        help='total time for testing data')
    parser.add_argument('--plotTime',
                        type=float,
                        default=10,
                        help='total time for plotting data')
    # TO-DO: input how many data points to skip
    # TO-DO: plotTime <= testTime
    parser.add_argument('--errorTime',
                        type=float,
                        default=1,
                        help='total time for error calculation')
    # TO-DO: errorTime <= testTime
    return parser.parse_args()


def main():

    args = parse_args()

    
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

    # generate data
    trajectory_history, time_history, dim = dg.generate_data('rk4',
                                                           'Lorenz_9dim',
                                                           t0,
                                                           .1,
                                                           1000)



    print(trajectory_history)


    # TO-DO: Generate feature vector
    # TO-DO: Preform regression
    # TO-DO: Prediction
    # TO-DO: Error and plotting



if __name__ == "__main__":
    main()
