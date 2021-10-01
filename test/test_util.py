import unittest

import numpy as np

from util import accuracy

"""
You are encouraged to write unit tests for your functions and algorithms to make debugging (and grading) easier.
"""


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(1., accuracy([1, 1, 1, 1], [1, 1, 1, 1]))
        self.assertEqual(0.5, accuracy(np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1])))

    #def test_precision(self):
    #    self.assertEqual(1., accuracy([1, 1, 1, 1], [1, 1, 1, 1]))
    #    self.assertEqual(0.5, accuracy(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])))

    #def test_recall(self):
    #    self.assertEqual(1., accuracy([1, 1, 1, 1], [1, 1, 1, 1]))
    #    self.assertEqual(0.5, accuracy(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])))

    #def test_auc(self):
    #    self.assertEqual(1., accuracy([1, 1, 1, 1], [1., 1., 1., 1.]))
    #    self.assertEqual(0.5, accuracy(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])))
