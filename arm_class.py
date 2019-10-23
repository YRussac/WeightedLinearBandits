#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Arm class
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)
# importation

import numpy as np


class Arm(object):
    def pull(self, theta, sigma_noise):
        print('pulling from the parent class')
        pass

    def get_expected_reward(self, theta):
        print('Receiving reward from the parent class')


class ArmGaussian(Arm):
    """
    Arm vector with gaussian noise
    """

    def __init__(self, vector):
        """
        Constructor
        """
        assert isinstance(vector, np.ndarray), 'np.array required'
        self.features = vector  # action for the arm, numpy-array
        self.dim = vector.shape[0]

    def get_expected_reward(self, theta):
        """
        Return dot(A_t,theta)
        """
        assert isinstance(theta, np.ndarray), 'np.array required for the theta vector'
        return np.dot(self.features, theta)

    def pull(self, theta, sigma_noise):
        """
        We are in the stochastic setting.
        The reward is draw according to Normal(dot(A_t,theta),sigma_noise**2)
        """
        return np.random.normal(self.get_expected_reward(theta), sigma_noise)
