
import numpy as np

# systems of equations

# template for creating new system of equations
# class [system name]:
#     # dimension of system
#     system_dim = [dimention of system]
#     # dimension of system and variational matrix elements
#     dim = system_dim + system_dim**2

#     # constants
#     [constant_1] = [value_1]
#     [constant_2] = [value_2]
#     ...

#     # initial condition
#     X0 = np.array([value_1, value_2, ... , value_dim])

#     def evaluate(self, X, t):
#         # initialize storage for output values
#         output_values = np.zeros(self.dim)

#         # unpack state variables
#         x1 = X[0]
#         x2 = X[1]
#         ...
#         # variational matrix elements
#         dxx = X[]
#         dxy = X[]
#         ...

#         # define the system of equations
#         output_values[0] = [equation_1]
#         output_values[1] = [equation_2]
#         ...
#         # jacobian dot variation matrix
#         output_value[] = [    ]
#         ...

#         return output_values


class Lorenz_63:
    # dimension of system
    system_dim = 3
    # dimension of system and variational matrix elements
    dim = system_dim + system_dim**2
    #  constants
    a = 10.
    b = 8 / 3
    r = 28.

    # initial condition
    X0 = np.array([17.67715816276679, 12.931379185960404, 43.91404334248268])

    def evaluate(self, X, t):
        # initialize storage for output values
        output_values = np.zeros(self.dim)

        # unpack state variables
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        # initialize jacobian storage (vector data structure)
        J = np.zeros(self.system_dim**2)
        # fill in jacobian values
        J[0] = -self.a
        J[1] = self.a
        J[2] = 0
        J[3] = self.r - x3
        J[4] = -1
        J[5] = -x1
        J[6] = x2
        J[7] = x1
        J[8] = -self.b

        # convert varation from matrix to vector to simplify computations
        variational_matrix = np.zeros([self.system_dim,self.system_dim])
        for i in range(0,self.system_dim):
            variational_matrix[i,:] = X[self.system_dim*(i+1):self.system_dim*(i+2)]

        # define the system of equations
        output_values[0] = self.a * (x2 - x1)
        output_values[1] = self.r * x1 - x2 - x1 * x3
        output_values[2] = x1 * x2 - self.b * x3
        # compute jacobian dot variational matrix
        for j in range(self.system_dim):  # loop over jacobian rows
            for v in range(self.system_dim):  # loop over variational columns
                output_values[self.system_dim + j*self.system_dim + v] = np.dot(J[j*self.system_dim:(j+1)*self.system_dim], variational_matrix[:,v])
 
        if t == 0:
            print("First timestep output_values:")
            print(output_values)
        return output_values


# dictionary to map user input to system of equations class
system_of_equations_map = {
    'Lorenz_63': Lorenz_63,
    #'Lorenz_9dim': Lorenz_9dim,
    #'Jerk1': Jerk1,
    #'Jerk2': Jerk2,
    }


# numerical integrators

# runge kutta 4th order
def runge_kutta_4(system, X0, t, h):
    k1 = system.evaluate(X0, t)
    k2 = system.evaluate(X0 + (h/2)*k1, t + h/2)
    k3 = system.evaluate(X0 + (h/2)*k2, t + h/2)
    k4 = system.evaluate(X0 + h*k3, t + h)

    x1 = X0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x1


# solver function to generate trajectory
def solver(numIntegrator, system, X0, h, n, t0):
    #  numIntegrator - which numerical integrator to use
    # system - system class
    # X0 - initial condition
    # h - time step
    # n - number of time steps
    # t0 - initial time

    # list of times to evaluate
    time_steps = np.linspace(t0, t0+h*n, n+1)
    # initialize storage for trajectory
    trajectory_history = np.zeros([len(X0), len(time_steps)])

    for i in range(len(time_steps)):
        t = time_steps[i]
        trajectory_history[:, i] = X0
        X0 = numIntegrator(system, X0, t, h)

    return trajectory_history, time_steps


# self defined numerical integrators
defined_integrate_map = {
    'rk4': runge_kutta_4
    }


def generate_data(numIntegrator,
                  system,
                  t0,
                  dt,
                  totalTime_pts,
                  variational_matrix,
                  get_dim=False):

    # initialize system of equations
    if system in system_of_equations_map:
        system = system_of_equations_map[system]()

    else:
        raise ValueError(f"System '{system}' not recognized. Available systems: {list(system_of_equations_map.keys())}")  # noqa: E501

    # get system dimension and initial condition
    system_dim = system.system_dim
    dim = system.dim
    IC = system.X0
    
    if get_dim == True:
        return system_dim

    print(f'System dimension: {dim}')
    print(f'Initial condition: {IC}')
    
    # initialize IC and variational storage
    X0 = np.zeros(dim)
    # copy in IC and variational vector
    X0[0:system_dim] = IC
    for i in range(0,system_dim):
        X0[system_dim*(i+1):system_dim*(i+2)] = variational_matrix[i,:]

    # initialize numerical integrator
    if numIntegrator in defined_integrate_map:
        # use integrator defined in this file
        numIntegrator = defined_integrate_map[numIntegrator]

    else:
        raise ValueError(f"Integrator '{numIntegrator}' not recognized. Available integrators: {list(defined_integrate_map.keys())}")  # noqa: E501

    # data generation
    trajectory_history, time_history = solver(numIntegrator,
                                              system,
                                              X0,
                                              dt,
                                              totalTime_pts,
                                              t0)

    if np.isinf(trajectory_history).any():
        raise ValueError('The generated trajectory contains Inf values at point ' + str(np.isinf(trajectory_history).argmax()) + ' out of ' + str(trajectory_history.shape[1]) + '. Try reducing the time step or changing the integrator.')  # noqa: E501
    if np.isnan(trajectory_history).any():
        raise ValueError('The generated trajectory contains NaN values at point ' + str(np.isnan(trajectory_history).argmax()) + ' out of ' + str(trajectory_history.shape[1]) + '. Try reducing the time step or changing the integrator.')  # noqa: E501
    if trajectory_history.shape[1] != (totalTime_pts + 1):
        raise ValueError('trajectory_history incorrect size: lenght ' + str(trajectory_history.shape[1]) + ' out of ' + str(totalTime_pts + 1) + ' data points. May be due to scipy solve_ivp encoruntering Inf or Nan values and failing silently')  # noqa: E501

    return trajectory_history, time_history, dim

