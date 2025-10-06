
import numpy as np
from scipy.integrate import solve_ivp

# systems of equations

# template for creating new system of equations
# class [system name]:
#     # dimension of system
#     dim = [dimension]

#     #  constants
#     [constant_1] = [value_1]
#     [constant_2] = [value_2]
#     ...

#     initial condition
#     X0 = np.array([value_1, value_2, ... , value_dim])

#     def evaluate(self, X, t):
#         initialize storage for output values
#         output_values = np.zeros(self.dim)

#         unpack state variables
#         x1 = X[0]
#         x2 = X[1]
#         ...

#         define the system of equations
#         output_value[0] = [equation_1]
#         output_value[1] = [equation_2]
#         ...

#         return output_values


class Lorenz_63:
    # dimension of system
    dim = 3
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

        # define the system of equations
        output_values[0] = self.a * (x2 - x1)
        output_values[1] = self.r * x1 - x2 - x1 * x3
        output_values[2] = x1 * x2 - self.b * x3

        return output_values


class Lorenz_9dim:
    # dimension of system
    dim = 9
    #  constants
    a = 1/2
    sigma = 0.5
    r = 28.

    b1 = 4 * (1 + a**2)/(1 + 2 * a**2)
    b2 = (1 + 2 * a**2)/(2 * (1 + a**2))
    b3 = 2 * (1 - a**2)/(1 + a**2)
    b4 = a**2 / (a + a**2)
    b5 = 8 * a**2 / (1 + 2 * a**2)
    b6 = 4 / (1 + 2 * a**2)

    # initial condition
    X0 = np.array([
        0.96803247, -4.49827049, -1.56463134, -2.59560938, -0.95404047,
        -5.51041908, 0.35826517, -14.78422783, -4.50578669
        ])

    def evaluate(self, X, t):
        # initialize storage for output values
        output_values = np.zeros(self.dim)

        # unpack state variables
        c1 = X[0]
        c2 = X[1]
        c3 = X[2]
        c4 = X[3]
        c5 = X[4]
        c6 = X[5]
        c7 = X[6]
        c8 = X[7]
        c9 = X[8]

        # update constants
        sigma = self.sigma
        r = self.r
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        b4 = self.b4
        b5 = self.b5
        b6 = self.b6

        # define the system of equations
        output_values[0] = - sigma * b1 * c1 - c2 * c4 + b4 * c4**2 + b3 * c3 * c5 - sigma * b2 * c7  # noqa: E501
        output_values[1] = - sigma * c2 + c1 * c4 - c2 * c5 + c4 * c5 - sigma * c9 / 2  # noqa: E501
        output_values[2] = - sigma * b1 * c3 + c2 * c4 - b4 * c2**2 - b3 * c1 * c5 + sigma * b2 * c8  # noqa: E501
        output_values[3] = - sigma * c4 - c2 * c3 - c2 * c5 + c4 * c5 + sigma * c9 / 2  # noqa: E501
        output_values[4] = - sigma * b5 * c5 + c2**2 / 2 - c4 * c9
        output_values[5] = - b6 * c6 + c2 * c9 - c4 * c9
        output_values[6] = - b1 * c7 - r * c1 + 2 * c5 * c8 - c4 * c9
        output_values[7] = - b1 * c8 + r * c3 - 2 * c5 * c7 + c2 * c9
        output_values[8] = - c9 - r * c2 + r * c4 - 2 * c2 * c6 + 2 * c4 * c6 + c4 * c7 - c2 * c8  # noqa: E501

        return output_values


# dictionary to map user input to system of equations class
system_of_equations_map = {
    'Lorenz_63': Lorenz_63,
    'Lorenz_9dim': Lorenz_9dim,
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
    'rk4': runge_kutta_4,
    }

# numerical integrators from scipy.integrate solve_ivp
scipy_integrate_list = [
    'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'
    ]
# TO-DO: implement all scipy.integrate solve_ivp methods


def generate_data(numIntegrator, system, t0, dt, totalTime_pts):
    # initialize system of equations
    if system in system_of_equations_map:
        system = system_of_equations_map[system]()

    else:
        raise ValueError(f"System '{system}' not recognized. Available systems: {list(system_of_equations_map.keys())}")  # noqa: E501

    # initialize numerical integrator
    scipy_integrator = False  # flag for which type of integrator to use (defined in file or from scipy.integrate) # noqa: E501
    if numIntegrator in scipy_integrate_list:
        # use integrator from scipy.integrate solve_ivp
        scipy_integrator = True

    elif numIntegrator in defined_integrate_map:
        # use integrator defined in this file
        numIntegrator = defined_integrate_map[numIntegrator]

    else:
        raise ValueError(f"Integrator '{numIntegrator}' not recognized. Available integrators: {scipy_integrate_list + list(defined_integrate_map.keys())}")  # noqa: E501

    # get system dimension and initial condition
    dim = system.dim
    X0 = system.X0

    print(f'System dimension: {dim}')
    print(f'Initial condition: {X0}')

    # data generation
    if scipy_integrator is True:

        def wrapped_system(t, X):  # wrapper to match solve_ivp input format
            return system.evaluate(X, t)

        time_history = np.linspace(t0, t0 + totalTime_pts*dt,
                                   totalTime_pts + 1)
        solution = solve_ivp(wrapped_system,
                             (t0, t0 + totalTime_pts * dt),
                             X0,
                             t_eval=time_history,
                             method=numIntegrator)
        trajectory_history = solution.y

    else:
        trajectory_history, time_history = solver(numIntegrator,
                                                  system,
                                                  X0,
                                                  dt,
                                                  totalTime_pts,
                                                  t0)

    if np.isinf(trajectory_history).any():
        raise ValueError("The generated trajectory contains Inf values. Try reducing the time step or changing the integrator.")  # noqa: E501
    if np.isnan(trajectory_history).any():
        raise ValueError("The generated trajectory contains NaN values. Try reducing the time step or changing the integrator.")  # noqa: E501

    return trajectory_history, time_history, dim
