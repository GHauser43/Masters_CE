
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
        # initialize storage
        nonlinear_featureVector = np.zeros((self.dnonlin, N))
        count = 0
        for row in range(self.dlin):
            for col in range(row, self.dlin):
                nonlinear_featureVector[count, :] = linear_featureVector[row, :] * linear_featureVector[col, :]  # noqa: E501
                count += 1
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
