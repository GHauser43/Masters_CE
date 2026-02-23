import numpy as np

#  if add system, update system_of_equations_map at bottom of file

# Template:
# class [system name]:
#     # dimension of system
#     system_dim = [dimension]
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

#         # define the system of equations
#         output_values[0] = [equation_1]
#         output_values[1] = [equation_2]
#         ...

#         # initialize jacobian storage (vector data structure)
#         J = np.zeros(self.system_dim**2)
#         # fill in jacobian values
#         J[0] = [dx1/dx1]
#         J[1] = [dx2/dx1]
#         ...

#         # convert varation from vector to matrix to simplify computations
#         variational_matrix = np.zeros([self.system_dim, self.system_dim])
#         for i in range(0, self.system_dim):
#             variational_matrix[i, :] = X[self.system_dim*(i+1):
#                                          self.system_dim*(i+2)]

#         # compute jacobian dot variational matrix
#         for j in range(self.system_dim):  # loop over jacobian rows
#             for v in range(self.system_dim):  # loop over variational columns
#                 output_values[self.system_dim + j*self.system_dim + v] = \
#                     np.dot(J[j*self.system_dim:(j+1)*self.system_dim],
#                            variational_matrix[:, v])

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

        # define the system of equations
        output_values[0] = self.a * (x2 - x1)
        output_values[1] = self.r * x1 - x2 - x1 * x3
        output_values[2] = x1 * x2 - self.b * x3

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


class Lorenz_9dim:
    # dimension of system
    system_dim = 9
    # dimension of system and variational matrix elements
    dim = system_dim + system_dim**2
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

        # initialize jacobian storage (vector data structure)
        J = np.zeros(self.system_dim**2)
        # fill in jacobian values
        # wrt c1
        # - sigma * b1 * c1 - c2 * c4 + b4 * c4**2
        # + b3 * c3 * c5 - sigma * b2 * c7
        J[0] = - sigma * b1
        J[1] = - c4
        J[2] = b3 * c5
        J[3] = - c2 + b4 * 2 * c4
        J[4] = b3 * c3
        J[5] = 0
        J[6] = - sigma * b2
        J[7] = 0
        J[8] = 0
        # wrt c2
        # - sigma * c2 + c1 * c4 - c2 * c5 + c4 * c5 - sigma * c9 / 2
        J[9] = c4
        J[10] = - sigma - c5
        J[11] = 0
        J[12] = c1 + c5
        J[13] = -c2 + c4
        J[14] = 0
        J[15] = 0
        J[16] = 0
        J[17] = -sigma / 2
        # wrt c3
        # - sigma * b1 * c3 + c2 * c4 - b4 * c2**2
        # - b3 * c1 * c5 + sigma * b2 * c8
        J[18] = - b3 * c5
        J[19] = c4 - b4 * 2 * c2
        J[20] = -sigma * b1
        J[21] = c2
        J[22] = -b3 * c1
        J[23] = 0
        J[24] = 0
        J[25] = sigma * b2
        J[26] = 0

        # wrt c4
        # - sigma * c4 - c2 * c3 - c2 * c5 + c4 * c5 + sigma * c9 / 2
        J[27] = 0
        J[28] = - c3 - c5
        J[29] = - c2
        J[30] = - sigma + c5
        J[31] = - c2 + c4
        J[32] = 0
        J[33] = 0
        J[34] = 0
        J[35] = sigma / 2
        # wrt c5
        # - sigma * b5 * c5 + c2**2 / 2 - c4 * c9
        J[36] = 0
        J[37] = c2
        J[38] = 0
        J[39] = -c9
        J[40] = - sigma * b5
        J[41] = 0
        J[42] = 0
        J[43] = 0
        J[44] = - c4
        # wrt c6
        # - b6 * c6 + c2 * c9 - c4 * c9
        J[45] = 0
        J[46] = c9
        J[47] = 0
        J[48] = - c9
        J[49] = 0
        J[50] = - b6
        J[51] = 0
        J[52] = 0
        J[53] = c2 - c4
        # wrt c7
        # - b1 * c7 - r * c1 + 2 * c5 * c8 - c4 * c9
        J[54] = -r
        J[55] = 0
        J[56] = 0
        J[57] = - c9
        J[58] = 2 * c8
        J[59] = 0
        J[60] = - b1
        J[61] = 2 * c5
        J[62] = - c4
        # wrt c8
        # - b1 * c8 + r * c3 - 2 * c5 * c7 + c2 * c9
        J[63] = 0
        J[64] = c9
        J[65] = r
        J[66] = 0
        J[67] = - 2 * c7
        J[68] = 0
        J[69] = - 2 * c5
        J[70] = - b1
        J[71] = c2
        # wrt c9
        # - c9 - r * c2 + r * c4 - 2 * c2 * c6
        # + 2 * c4 * c6 + c4 * c7 - c2 * c8
        J[72] = 0
        J[73] = - r - 2 * c6 - c8
        J[74] = 0
        J[75] = r + 2 * c6 + c7
        J[76] = 0
        J[77] = - 2 * c2 + 2 * c4
        J[78] = c4
        J[79] = - c2
        J[80] = -1

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


class Jerk1:
    # dimension of system
    system_dim = 3
    # dimension of system and variational matrix elements
    dim = system_dim + system_dim**2

    #  constants
    a = 3.6

    # initial condition
    X0 = np.array([0.5, 0.5, 0.5])

    def evaluate(self, X, t):
        # initialize storage for output values
        output_values = np.zeros(self.dim)

        # unpack state variables
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]

        # define the system of equations
        output_values[0] = x2
        output_values[1] = x3
        output_values[2] = - self.a * x3 + x1 * x2**2 - x1**3

        # initialize jacobian storage (vector data structure)
        J = np.zeros(self.system_dim**2)
        # fill in jacobian values
        J[0] = 0
        J[1] = 1
        J[2] = 0
        J[3] = 0
        J[4] = 0
        J[5] = 1
        J[6] = x2**2 - 3 * x1**2
        J[7] = x1 * 2 * x2
        J[8] = -self.a

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


# chaotic jerk system #2
class Jerk2:
    # dimension of system
    system_dim = 3
    # dimension of system and variational matrix elements
    dim = system_dim + system_dim**2

    #  constants
    a = 3.6
    b = 1.3
    c = 0.1

    # initial condition
    X0 = np.array([0.5, 0.5, 0.5])

    def evaluate(self, X, t):
        # initialize storage for output values
        output_values = np.zeros(self.dim)

        # unpack state variables
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]

        # define the system of equations
        output_values[0] = x2
        output_values[1] = x3
        output_values[2] = - self.a * x3 - self.b * x1 + self.c * x2 + x1 * x2**2 - x1**3  # noqa: E501

        # initialize jacobian storage (vector data structure)
        J = np.zeros(self.system_dim**2)
        # fill in jacobian values
        J[0] = 0
        J[1] = 1
        J[2] = 0
        J[3] = 0
        J[4] = 0
        J[5] = 1
        J[6] = -self.b + x2**2 - 3 * x1**2
        J[7] = self.c + x1 * 2 * x2
        J[8] = - self.a

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
    'Lorenz_9dim': Lorenz_9dim,
    'Jerk1': Jerk1,
    'Jerk2': Jerk2,
    }
