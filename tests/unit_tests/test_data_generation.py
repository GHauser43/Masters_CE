import sys
import unittest
import numpy as np
import numpy.testing as npt

sys.path.append('src/') # noqa

import data_generation as dg

class TestDataGeneration(unittest.TestCase):

    def test_split_data(self):
        array = np.array([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10]])
        array_result = dg.split_data(array, 1, 4)
        array_expected = np.array([[2, 3, 4],
                                    [7, 8, 9]])
        npt.assert_array_equal(array_result, array_expected)

        array = np.array([1, 2, 3, 4, 5])
        array_result = dg.split_data(array, 0, 3)
        array_expected = np.array([1, 2, 3])
        npt.assert_array_equal(array_result, array_expected)

        # add test for split data with feature vector params
    def test_train_test_data_split(self):
        trajectoryHistory = np.array([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.],[-0.,-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.,-11.]])
        timeHistory = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.])
        warmupTime_pts = 3 
        warmtrainTime_pts = 7 
        delayTime_pts = 1
        totalTime_pts = 11
    
        trajectoryHistory_train, timeHistory_train, trajectoryHistory_test, timeHistory_test = dg.train_test_data_split(trajectoryHistory,  # noqa: E501
                          timeHistory,
                          warmupTime_pts,
                          warmtrainTime_pts,
                          delayTime_pts,
                          totalTime_pts)
        
        npt.assert_array_equal(timeHistory_train,np.array([1.,2.,3.,4.,5.,6.,7]))
        npt.assert_array_equal(timeHistory_test,np.array([7.,8.,9.,10.,11.]))


if __name__ == '__main__':
    unittest.main()
