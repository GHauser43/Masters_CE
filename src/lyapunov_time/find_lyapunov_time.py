import argparse
import numpy as np
import variational_data_generation as vdg


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--system',
                        type=str,
                        required=True)
    parser.add_argument('--timePts',
                        type=int,
                        default=10000)
    parser.add_argument('--t0',
                        type=float,
                        default=0.0)
    parser.add_argument('--dt',
                        type=float,
                        default=0.0025)
    parser.add_argument('--numIntegrator',
                        type=str,
                        default='rk4')

    args = parser.parse_args()

    system = args.system
    timePts = args.timePts
    t0 = args.t0
    dt = args.dt
    numIntegrator = args.numIntegrator

    system_dim = vdg.generate_data(None,
                                   system,
                                   None,
                                   None,
                                   None,
                                   None,
                                   get_dim=True)

    # initialize variational matrix
    variational_matrix = np.zeros([system_dim ,system_dim])

    ### TEMP
    for i in range(system_dim):
        variational_matrix[i,i]=1
    
    print('------------------------')
    print('Variational Matrix is:')
    print(variational_matrix)
    print('------------------------')
    trajectory_history, time_history, dim = vdg.generate_data(numIntegrator,
                                                              system,
                                                              t0,
                                                              dt,
                                                              timePts,
                                                              variational_matrix)

    delta_matrix = np.array([trajectory_history[3:6,-1],
                             trajectory_history[6:9,-1],
                             trajectory_history[9:12,-1]]).T
                             #  transpose this or not? old code does, but think this may be wrong

    print('------------------------')
    print('The delta Matrix is:')
    print(delta_matrix)
    print('------------------------')
    x_variation = sum(delta_matrix[:,0])
    y_variation = sum(delta_matrix[:,1])
    z_variation = sum(delta_matrix[:,2])
    print('variantion in x is: ',x_variation )
    print('variantion in y is: ',y_variation )
    print('variantion in z is: ',z_variation )
    print('------------------------')
    
    print('eigenvalues',np.linalg.eigvals(delta_matrix))

    print('lyuponov exponents are')
    print(np.log(np.linalg.eigvals(delta_matrix))/(dt*timePts)) 
    print(f'for t={dt*timePts}')


if __name__=="__main__":
    main()
