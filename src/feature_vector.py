
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
        self.dim = dim
        self.k = k
        self.s = s
        self.p = p

    def construct_constant(self, N):
        print('constant feature vector')

    
    def construct_linear(self, N, data):
        print('linear feature vector')


    def construct_nonlinear(self, N, data):
        print('nonlinear feature vector')


    def construct_featureVector(self, data):
        print('constructing feature vector')
        N = data.shape[0]

        print(np.shape(data))
