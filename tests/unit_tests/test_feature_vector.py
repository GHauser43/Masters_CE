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


if __name__ == '__main__':
    unittest.main()
