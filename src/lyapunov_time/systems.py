import numpy as np


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

        # define the system of equations
        output_values[0] = self.a * (x2 - x1)
        output_values[1] = self.r * x1 - x2 - x1 * x3
        output_values[2] = x1 * x2 - self.b * x3

        # convert varation from vector to matrix to simplify computations
        variational_matrix = np.zeros([self.system_dim, self.system_dim])
        for i in range(0, self.system_dim):
            variational_matrix[i, :] = X[self.system_dim*(i+1):
                                         self.system_dim*(i+2)]

        # compute jacobian dot variational matrix
        for j in range(self.system_dim):  # loop over jacobian rows
            for v in range(self.system_dim):  # loop over variational columns
                output_values[self.system_dim + j*self.system_dim + v] = \
                    np.dot(J[j*self.system_dim:(j+1)*self.system_dim],
                           variational_matrix[:, v])

        return output_values


# dictionary to map string to system of equations class
system_of_equations_map = {
    'Lorenz_63': Lorenz_63,
    #  'Lorenz_9dim': Lorenz_9dim,
    #  'Jerk1': Jerk1,
    #  'Jerk2': Jerk2,
    }
