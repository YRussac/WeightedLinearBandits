#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Implementation of the SW-LinUCB policy by Cheung et. al
    "Learning to optimize under Non-Stationarity" AISTATS19
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import numpy as np
from numpy.linalg import pinv


class SWLinUCB(object):
    def __init__(self, d, delta, alpha, lambda_, s, tau, name, sm, sigma_noise, verbose):
        """
        Implementation of the class for the Discounted Linear UCB
        param:
            - d: dimension of the action vectors
            - delta: probability of theta in the confidence bound
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: constant such that L2 norm of theta smaller than s
            - tau: sliding window parameter
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
        self.tau = tau
        self.name = name
        self.verbose = verbose
        self.sigma_noise = sigma_noise
        self.sm = False  # TODO: implement the Sherman-Morisson trick here.
        self.s = s
        self.omniscient = False

        # build attributes
        self.beta_t = np.sqrt(self.lambda_) * self.s + self.sigma_noise * \
            np.sqrt(self.dim * np.log((1+self.tau/self.lambda_)/self.delta))
        # Same confidence parameter as in Cheung 1 al. paper

        # attributes for the re-init
        self.t = 0
        self.hat_theta = np.zeros(self.dim)
        self.cov = self.lambda_ * np.identity(self.dim)
        self.invcov = 1 / self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)

        self.a_tau = []
        self.reward_tau = []

    def select_arm(self, arms):
        """
        Selecting an arm according to the SW-LinUCB policy
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(arms) == list, 'List of arms as input required'
        k_t = len(arms)  # available actions at time t
        ucb_s = np.zeros(k_t)  # upper-confidence bounds for every action
        for (i, a) in enumerate(arms):
            a = a.features
            invcov_a = np.inner(self.invcov, a.T)
            ucb_s[i] = np.dot(self.hat_theta, a) + self.alpha * self.beta_t * np.sqrt(np.dot(a, invcov_a))
        mixer = np.random.random(ucb_s.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucb_s)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        if self.verbose:
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
        if self.t < self.tau:
            aat = np.outer(features, features.T)
            self.cov = self.cov + aat
            self.b += reward * features
            self.a_tau.append(features)
            self.reward_tau.append(reward)
            if not self.sm:
                self.invcov = pinv(self.cov)
            else:
                print("To be checked")
                a = features[:, np.newaxis]
                const = 1 / (1 + np.dot(features, np.inner(self.invcov, features)))
                const2 = np.matmul(self.invcov, a)
                self.invcov = self.invcov - const * np.matmul(const2, const2.T)
        else:
            aat = np.outer(features, features.T)
            act_delayed = self.a_tau.pop(0)
            if self.verbose:
                print('--- act_delayed:', act_delayed)
            aat_delayed = np.outer(act_delayed, act_delayed.T)
            rew_delayed = self.reward_tau.pop(0)
            self.cov = self.cov + aat - aat_delayed
            self.b = self.b + reward * features - rew_delayed * act_delayed
            self.a_tau.append(features)
            self.reward_tau.append(reward)
            if not self.sm:
                self.invcov = pinv(self.cov)
            else:
                print("To be checked")
                a = features[:, np.newaxis]
                a_old = act_delayed[:, np.newaxis]
                const = 1 / (1 + np.dot(features, np.inner(self.invcov, features)))
                const2 = np.matmul(self.invcov, a)
                c = self.invcov - const * np.matmul(const2, const2.T)
                const = 1 / (1 - np.dot(act_delayed, np.inner(c, act_delayed)))
                const2 = np.matmul(c, a_old)
                self.invcov = c + const * np.matmul(const2, const2.T)
        self.hat_theta = np.inner(self.invcov, self.b)
        if self.verbose:
            print('--- t:', self.t)
            print('--- a_tau:', self.a_tau)
            print('--- reward_tau:', self.reward_tau)
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
        self.b = np.zeros(self.dim)
        self.a_tau = []
        self.reward_tau = []

        if self.verbose:
            print('Parameters of the policy reinitialized')
            print('Design Matrix after init: ', self.cov)

    def __str__(self):
        return 'SW-LinUCB' + self.name

    @staticmethod
    def id():
        return 'SW-LinUCB'
