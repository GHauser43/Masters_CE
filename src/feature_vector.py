
import numpy as np


class FeatureVector:
    # takes in an array and constructs a feature vector
    # inputs:
    #   data - numpy array of shape ( _ , dim)
    #   dim - dimension of the system
    #   (k-1) - number of current and previous time steps spaced by s
    #   (s-1) - number of skipped steps between concscutive observations
    #   p - polynomial degree

    def __init__(self, dim, k, s, p):
        self.d = dim
        self.k = k
        self.s = s
        self.p = p

        self.dlin = self.d * self.k
        self.dnonlin = int(self.dlin * (self.dlin + 1) / 2)
        self.dtot = 1 + self.dlin + self.dnonlin

    def construct_constant(self, N):
        return np.ones((1, N))

    def construct_linear(self, N, data):
        # initialize storage
        linear_featureVector = np.zeros((self.dlin, N))
        for delay in range(self.k):
            for j in range(delay, N):
                # linear_featureVector[self.d * delay:self.d * (delay + 1), j] = data[:, j - delay]  # noqa: E501 (no s case)
                linear_featureVector[self.d * delay:self.d * (delay + 1), j] = data[:, j - ((self.s)*(delay))]  # noqa: E501
        return linear_featureVector

    def construct_nonlinear(self, N, linear_featureVector):
        # gathers all the unique polynomials of power self.p and
        #   populates the nonlinear_featureVector accordingly
        # can handle p >= 9

        # unique_keys keeps track of what unique polynomials
        #   are in the featue vector. initialized with key values
        #   from linear part
        unique_keys = np.zeros(linear_featureVector.shape[0], dtype=np.int64)
        # initialize with linear entries
        for dim in range(0, linear_featureVector.shape[0]):
            unique_keys[dim] = 10**((linear_featureVector.shape[0] - 1) - dim)
        # unique_featureVector holds values associated with unique_key values
        unique_featureVector = np.empty((0, N))  # initialize array
        # have running feat vec and keys to record all (non unique) values
        running_featureVector = linear_featureVector
        running_keys = unique_keys

        # iterates over the power (polynomial) being calculated
        # value of power doesn't matter, just corrent number of iterations
        for power in range(1, self.p):
            # add_to_running records what data to add to running_featureVector
            #   after each power loop
            add_to_running = np.empty((0, N))
            for row1_idx in range(linear_featureVector.shape[0]):
                row1_key = unique_keys[row1_idx]
                for row2_idx in range(running_featureVector.shape[0]):
                    row2_key = running_keys[row2_idx]

                    new_key = row1_key + row2_key
                    new_row = linear_featureVector[row1_idx, :] * running_featureVector[row2_idx, :]  # noqa: E501
                    running_keys = np.append(running_keys, new_key)
                    add_to_running = np.vstack((add_to_running, new_row))

                    if new_key not in unique_keys:
                        unique_keys = np.append(unique_keys, new_key)
                        unique_featureVector = np.vstack((
                                               unique_featureVector,
                                               new_row))

            running_featureVector = np.vstack((running_featureVector,
                                               add_to_running))

        nonlinear_featureVector = unique_featureVector
        return nonlinear_featureVector

    def construct_featureVector(self, data):
        N = data.shape[1]  # number of time points
        # generate constant part of feature vector
        constant_featureVector = self.construct_constant(N)
        # generate linear part of feature vector
        linear_featureVector = self.construct_linear(N, data)
        # generate nonlinear part of feature vector
        nonlinear_featureVector = self.construct_nonlinear(N, linear_featureVector)  # noqa: E501
        # combine all parts of feature vector
        full_featureVector = np.vstack((constant_featureVector,
                                        linear_featureVector,
                                        nonlinear_featureVector))
        # remove delay taps, then return full feature vector
        return full_featureVector[:, self.s * (self.k-1):]
