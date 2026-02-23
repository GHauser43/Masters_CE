import argparse
import numpy as np
import matplotlib.pyplot as plt
import systems


class Packing:

    def __init__(self, vec_dim, mat_rows, mat_cols):
        # class to pack/unpack matrix and vectors
        # vec_dim - length of vec
        # mat_rows - # of rown in matrix
        # mat_cols - # of cols in matrix
        self.vec_dim = vec_dim
        self.mat_rows = mat_rows
        self.mat_cols = mat_cols
        self.full_dim = vec_dim + mat_rows * mat_cols

    def pack(self, vec, matrix):
        # flattens matrix (row order) and vec to create full_vector
        full_vec = np.zeros(self.full_dim)
        full_vec[0:self.vec_dim] = vec
        for i in range(0, self.mat_rows):
            full_vec[(1+i)*self.mat_cols:(2+i)*self.mat_cols] = matrix[i, :]

        return full_vec

    def unpack(self, full_vec):
        # reconstructs matrix and vector from full_vector
        vec = np.zeros(self.vec_dim)
        matrix = np.zeros([self.mat_rows, self.mat_cols])
        vec = full_vec[0:self.vec_dim]
        for i in range(0, self.mat_rows):
            matrix[i, :] = full_vec[(i+1)*self.mat_cols: (2+i)*self.mat_cols]

        return vec, matrix


def runge_kutta_4(system, X0, t, h):
    k1 = system.evaluate(X0, t)
    k2 = system.evaluate(X0 + (h/2)*k1, t + h/2)
    k3 = system.evaluate(X0 + (h/2)*k2, t + h/2)
    k4 = system.evaluate(X0 + h*k3, t + h)

    x1 = X0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x1


def main():
    # system - name of system defined in variational_data_generation.py
    # timePts_transient - number of points to generate to get on attractor
    # timePts - number of points per renormalization step
    # tau - number of renormalizations
    #   ( total time points is timePts * tau
    # t0 - start time of simulation
    # dt - time step

    parser = argparse.ArgumentParser()

    parser.add_argument('--system',
                        type=str,
                        required=True)

    parser.add_argument('--timePts_transient',
                        type=int,
                        default=500)
    parser.add_argument('--timePts',
                        type=int,
                        default=10)
    parser.add_argument('--tau',
                        type=int,
                        default=40000)
    parser.add_argument('--t0',
                        type=float,
                        default=0.0)
    parser.add_argument('--dt',
                        type=float,
                        default=0.0025)

    args = parser.parse_args()

    system = args.system
    timePts_transient = args.timePts_transient
    timePts = args.timePts
    tau = args.tau
    t0 = args.t0
    dt = args.dt

    print('system:', system)
    # creates instance of system defined in systems.py
    #    converts string input to instance of class object
    #    using systems_of_equations_map
    system = systems.system_of_equations_map[system]()

    # data generation to get on attractor
    # setup IC with dummy variation (not numerically efficient, good enough)
    dummy_variational = np.zeros([system.dim])
    dummy_variational[0:system.system_dim] = system.X0
    # initialize trajectory storage
    trajectroy_history_transient = np.zeros([system.system_dim,
                                             timePts_transient+1])
    # time points
    time_history_transient = np.linspace(t0,
                                         t0+dt*timePts_transient,
                                         timePts_transient+1)
    # generate data - transient
    X0 = dummy_variational
    for i in range(0, timePts_transient+1):
        t = time_history_transient[i]
        trajectroy_history_transient[:, i] = X0[0:system.system_dim]
        X0 = runge_kutta_4(system, X0, t, dt)

    # Lyapunov exponent calculations
    # Benettin, Giancarlo, et al.
    # "Lyapunov characteristic exponents for smooth dynamical systems and for
    # Hamiltonian systems; a method for computing all of them.
    # Part 1: Theory." Meccanica 15.1 (1980): 9-20.

    # define variational_matrix (must be orthonormal)
    variational_matrix = np.identity(system.system_dim)

    # initialize S - accumulator for log growth rates
    S = np.zeros(system.system_dim)

    print('initial condition is:')
    print(trajectroy_history_transient[:, -1],
          'at time', time_history_transient[-1])

    print('')
    print('time steps per renormalization (timePts):', timePts)
    print('number of renormalization steps (tau)    ', tau)
    print('time per step (dt):                      ', dt)
    print('total simulation time:                   ', tau * timePts * dt)
    print('renormalization interval:                ', timePts * dt)
    print('')

    # create instance of Packing class
    packing = Packing(system.system_dim,
                      variational_matrix.shape[0],
                      variational_matrix.shape[1])

    # update t0
    t0 = time_history_transient[-1]
    # initial condition
    X0 = trajectroy_history_transient[:, -1]

    # time history
    time_history = np.linspace(t0,
                               t0+dt * timePts * tau,
                               timePts * tau)
    # initialize trajectory history storage
    trajectroy_history = np.zeros([system.dim, timePts*tau])

    for i in range(0, tau):
        X = packing.pack(X0, variational_matrix)
        for j in range(0, timePts):
            t = time_history[i * timePts + j]
            trajectroy_history[:, i*timePts + j] = X
            X = runge_kutta_4(system, X, t, dt)
        X0, variational_matrix_evolved = packing.unpack(X)

        # QR decomposition
        variational_matrix, R = np.linalg.qr(variational_matrix_evolved)

        S += np.log(np.abs(np.diag(R)))

    lyapunov_exponents = S / (tau * timePts * dt)
    print('lyapunov exponents are:')
    print(lyapunov_exponents)
    print('sum of exponents:', np.sum(lyapunov_exponents))
    print('')
    lyapunov_time = 1/np.max(lyapunov_exponents)
    print('lyapunov time is:', lyapunov_time)


if __name__ == "__main__":
    main()
