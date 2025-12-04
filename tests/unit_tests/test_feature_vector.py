import sys
import unittest
import numpy as np
import numpy.testing as npt

sys.path.append('src/')  # noqa

import feature_vector as fv


class TestFeatureVector(unittest.TestCase):

    def test_FeatureVector_class_case1(self):
        dim = 2
        k = 2
        s = 1
        p = 2
        featureVector = fv.FeatureVector(dim, k, s, p)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        solution = np.array([[1.,  1.],
                             [2.,  3.],
                             [5.,  6.],
                             [1.,  2.],
                             [4.,  5.],
                             [4.,  9.],
                             [10., 18.],
                             [2.,  6.],
                             [8., 15.],
                             [25., 36.],
                             [5., 12.],
                             [20., 30.],
                             [1.,  4.],
                             [4., 10.],
                             [16., 25.]])
        result = featureVector.construct_featureVector(data)

        npt.assert_array_equal(result, solution)

    def test_FeatureVector_class_case2(self):
        dim = 2
        k = 3
        s = 1
        p = 2
        featureVector = fv.FeatureVector(dim, k, s, p)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        solution = np.array([[1.],
                             [3.],
                             [6.],
                             [2.],
                             [5.],
                             [1.],
                             [4.],
                             [9.],
                             [18.],
                             [6.],
                             [15.],
                             [3.],
                             [12.],
                             [36.],
                             [12.],
                             [30.],
                             [6.],
                             [24.],
                             [4.],
                             [10.],
                             [2.],
                             [8.],
                             [25.],
                             [5.],
                             [20.],
                             [1.],
                             [4.],
                             [16.]])
        result = featureVector.construct_featureVector(data)
        npt.assert_array_equal(result, solution)

    def test_FeatureVector_class_case3(self):
        dim = 2
        k = 3
        s = 2
        p = 2
        featureVector = fv.FeatureVector(dim, k, s, p)
        data = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
        solution = np.array([[1.],
                             [5.],
                             [8.],
                             [3.],
                             [6.],
                             [1.],
                             [4.],
                             [25.],
                             [40.],
                             [15.],
                             [30.],
                             [5.],
                             [20.],
                             [64.],
                             [24.],
                             [48.],
                             [8.],
                             [32.],
                             [9.],
                             [18.],
                             [3.],
                             [12.],
                             [36.],
                             [6.],
                             [24.],
                             [1.],
                             [4.],
                             [16.]])
        result = featureVector.construct_featureVector(data)
        npt.assert_array_equal(result, solution)

    # add test for different p values
    def test_construct_nonlinear_case1(self):
        dim = 3
        k = 2
        s = 1
        p = 2
        featureVector = fv.FeatureVector(dim, k, s, p)
        fv_linear = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [0, 1, 2],
                              [3, 4, 5],
                              [6, 7, 8]])
        solution = np.array([
                            [1., 4., 9.],
                            [4., 10., 18.],
                            [7., 16., 27.],
                            [0., 2., 6.],
                            [3., 8., 15.],
                            [6., 14., 24.],
                            [16., 25., 36.],
                            [28., 40., 54.],
                            [0., 5., 12.],
                            [12., 20., 30.],
                            [24., 35., 48.],
                            [49., 64., 81.],
                            [0., 8., 18.],
                            [21., 32., 45.],
                            [42., 56., 72.],
                            [0., 1., 4.],
                            [0., 4., 10.],
                            [0., 7., 16.],
                            [9., 16., 25.],
                            [18., 28., 40.],
                            [36., 49., 64.]])
        nonlinear_featureVector = featureVector.construct_nonlinear(3,
                                                                    fv_linear)
        npt.assert_array_equal(nonlinear_featureVector, solution)

    def test_construct_nonlinear_case2(self):
        dim = 2
        k = 2
        s = 1
        p = 3
        featureVector = fv.FeatureVector(dim, k, s, p)
        fv_linear = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [0, 1, 2],
                              [3, 4, 5]])
        nonlinear_featureVector = featureVector.construct_nonlinear(3,
                                                                    fv_linear)
        solution = np.array([
                            [1., 4., 9.],
                            [4., 10., 18.],
                            [0., 2., 6.],
                            [3., 8., 15.],
                            [16., 25., 36.],
                            [0., 5., 12.],
                            [12., 20., 30.],
                            [0., 1., 4.],
                            [0., 4., 10.],
                            [9., 16., 25.],
                            [1., 8., 27.],
                            [4., 20., 54.],
                            [0., 4., 18.],
                            [3., 16., 45.],
                            [16., 50., 108.],
                            [0., 10., 36.],
                            [12., 40., 90.],
                            [0., 2., 12.],
                            [0., 8., 30.],
                            [9., 32., 75.],
                            [64., 125., 216.],
                            [0., 25., 72.],
                            [48., 100., 180.],
                            [0., 5., 24.],
                            [0., 20., 60.],
                            [36., 80., 150.],
                            [0., 1., 8.],
                            [0., 4., 20.],
                            [0., 16., 50.],
                            [27., 64., 125.]])
        npt.assert_array_equal(nonlinear_featureVector, solution)


if __name__ == '__main__':
    unittest.main()
