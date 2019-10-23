#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Description of the environment.
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importation
from arm_class import ArmGaussian
import numpy as np


class Environment:
    """
    param:
        - K: Number of arms (int)
        - d: Full dimension of the problem, dimension of the actions (int)
        - theta: d-dimensional vector, key hidden parameter of the problem
    """
    def __init__(self, d, theta, sigma_noise, verbose, param=None):
        self.theta = theta
        self.arms = []
        self.dim = d
        self.verbose = verbose
        self.sigma_noise = sigma_noise
        self.param = param

    def get_arms(self, k, unif_min=-1, unif_max=1):
        """
        Sample k arms on the sphere
        param:
            - k: number of arms generated
            - unif_min: minimum bound for the uniform distribution
            - unif_max: maximum bound for the uniform distribution

        """
        self.arms = []
        for i in range(k):
            new_arm = np.random.uniform(unif_min, unif_max, size=self.dim)
            norm = np.linalg.norm(new_arm)
            if norm > 1:
                new_arm = new_arm / norm
            self.arms.append(ArmGaussian(new_arm))
        if self.verbose:
            print('Actions received are: ', [arms.features for arms in self.arms])
        return self.arms

    def get_static_arms(self, k):
        """
        param:
            - k: Number of arms generated
        Output:
        -------
        If the arms remain the same during the entire experiment
        """
        if len(self.arms) == 0:
            for i in range(k):
                new_arm = np.random.uniform(-1, 1, size=self.dim)
                norm = np.linalg.norm(new_arm)
                if norm > 1:
                    new_arm = new_arm/norm
                self.arms.append(ArmGaussian(new_arm))
        if self.verbose:
            print('Actions received are: ', [arms.features for arms in self.arms])
        return self.arms

    def get_gaussian_arms(self, k, mu=None, sigma=None):
        """
        Sample k arms using gaussian distribution, then normalize them
        param:
            - mu: mu parameter for generating multivariate_normal
            - sigma: covariance matrix
            - k: number of arms generated
        """
        self.arms = []
        if mu is None:
            mu = np.zeros(self.dim)
        if sigma is None:
            sigma = np.identity(self.dim)
        for i in range(k):
            new_arm = np.random.multivariate_normal(mu, sigma)
            norm = np.linalg.norm(new_arm)
            if norm > 1:
                new_arm = new_arm / norm
            self.arms.append(ArmGaussian(new_arm))
        if self.verbose:
            print('Actions received are: ', [arms.features for arms in self.arms])
        return self.arms

    def get_static_canonical_arms(self, k):
        """
        param:
            - k: Number of arms generated
        Output:
        -------
        Return some static arms, with the arm i is the i-th canonical vector
        """

        if len(self.arms) == 0:
            for i in range(min(k, self.dim)):
                new_arm = np.zeros(self.dim)
                new_arm[i] = 1.0
                self.arms.append(ArmGaussian(new_arm))
            if self.verbose:
                print('Actions received are: ', [arms.features for arms in self.arms])
        return self.arms

    def play(self, choice):
        """
        Play arms and return the corresponding rewards
        Choice is an int corresponding to the number of the action
        """
        assert isinstance(choice, np.integer), 'Choice type should be np.int !'
        reward = self.arms[choice].pull(self.theta, self.sigma_noise)
        action_played = self.arms[choice].features
        return reward, action_played

    def get_expected_rewards(self):
        """
        Return the expected payoff of the contextualized arm armIndex
        """
        return [arm.get_expected_reward(self.theta) for arm in self.arms]

    def get_best_arm(self):
        """
        Return the indices of the best arm
        """
        current_rewards = self.get_expected_rewards()  # list of the expected rewards
        assert len(current_rewards) > 0, "Error: No action generated, current_rewards is empty"
        best_arm = np.argmax(current_rewards)  # number of the best arm
        assert isinstance(best_arm, np.integer), "Error: bestArm type should be int"
        best_reward = current_rewards[best_arm]  # current reward for this arm
        return best_arm, best_reward

    def display(self):
        for index, arm in enumerate(self.arms):
            print('===========================')
            print('ARM : %d', index + 1)
            print('arm features: %f', arm.features)
