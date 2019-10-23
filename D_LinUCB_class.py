#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Implementation of the D-LinUCB policy presented in the paper
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import numpy as np
from math import log
from numpy.linalg import pinv


class DLinUCB(object):
    def __init__(self, d, delta, alpha, lambda_, s, gamma, name, sm, sigma_noise, verbose):
        """
        Implementation of the class for the Discounted Linear UCB
        param:
            - d: dimension of the action vectors
            - delta: probability of theta in the confidence bound
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: constant such that L2 norm of theta smaller than s
            - gamma: discount parameter
            - name: additional suffix when comparing several policies (optional)
            - sm: Should Sherman-Morisson formula be used for inverting matrices ?
            - sigma_noise: square root of the variance of the noise
            - verbose: To print information
            - omniscient: Does the policy knows when the breakpoints happen ?
        ACTION NORMS ARE SUPPOSED TO BE BOUNDED BE 1
        """
        # immediate attributes from the constructor
        self.delta = delta
        self.dim = d
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
        self.name = name
        self.verbose = verbose
        self.sigma_noise = sigma_noise
        self.sm = False  # S-M cannot be used with this model for the moment
        self.omniscient = False

        # build attributes
        self.c_delta = 2 * log(1 / self.delta)  # first term in square root

        # attributes for the re-init
        self.t = 0
        self.hat_theta = np.zeros(self.dim)
        self.cov = self.lambda_ * np.identity(self.dim)
        self.cov_squared = self.lambda_ * np.identity(self.dim)
        self.invcov = 1 / self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)
        self.s = s
        self.gamma2_t = 1

    def select_arm(self, arms):
        """
        Selecting an arm according to the D-LinUCB policy
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(arms) == list, 'List of arms as input required'
        k_t = len(arms)  # available actions at time t
        ucb_s = np.zeros(k_t)  # upper-confidence bounds for every action
        const1 = np.sqrt(self.lambda_) * self.s
        beta_t = const1 + self.sigma_noise *\
            np.sqrt(self.c_delta + self.dim * np.log(1 + (1-self.gamma2_t)/(self.dim *
                    self.lambda_*(1 - self.gamma**2))))
        for (i, a) in enumerate(arms):
            a = a.features
            invcov_a = np.inner(self.invcov @ self.cov_squared @ self.invcov, a.T)
            ucb_s[i] = np.dot(self.hat_theta, a) + self.alpha * beta_t * np.sqrt(np.dot(a, invcov_a))
        mixer = np.random.random(ucb_s.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucb_s)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        if self.verbose:
            # Sanity checks
            print('-- lambda:', self.lambda_)
            print('--- beta_t:', beta_t)
            print('--- theta_hat: ', self.hat_theta)
            print('--- Design Matrix:', self.cov)
            print('--- b matrix:', self.b)
            print('--- UCBs:', ucb_s)
            print('--- Chosen arm:', chosen_arm)
        return chosen_arm

    def update_state(self, features, reward):
        """
        Updating the main parameters for the model
        param:
            - features: Feature used for updating
            - reward: Reward used for updating
        Output:
        -------
        Nothing, but the class instances are updated
        """
        assert isinstance(features, np.ndarray), 'np.array required'
        aat = np.outer(features, features.T)
        self.gamma2_t *= self.gamma ** 2
        self.cov = self.gamma * self.cov + aat + (1-self.gamma) * self.lambda_ * np.identity(self.dim)
        self.cov_squared = self.gamma ** 2 * self.cov + aat + (1-self.gamma**2) * self.lambda_ * np.identity(self.dim)
        self.b = self.gamma * self.b + reward * features
        if not self.sm:
            self.invcov = pinv(self.cov)
        else:
            raise NotImplementedError("Method SM is not implemented for D-LinUCB")
        self.hat_theta = np.inner(self.invcov, self.b)
        if self.verbose:
            print('AAt:', aat)
            print('Policy was updated at time t= ' + str(self.t))
            print('Reward received  =', reward)
        self.t += 1

    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyperparameters
        """
        self.t = 0
        self.hat_theta = np.zeros(self.dim)
        self.cov = self.lambda_ * np.identity(self.dim)
        self.invcov = 1 / self.lambda_ * np.identity(self.dim)
        self.cov_squared = self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)
        self.gamma2_t = 1
        if self.verbose:
            print('Parameters of the policy reinitialized')
            print('Design Matrix after init: ', self.cov)

    def __str__(self):
        return 'D-LinUCB' + self.name

    @staticmethod
    def id():
        return 'D-LinUCB'
