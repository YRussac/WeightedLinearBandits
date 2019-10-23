#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Implementation of the Random Decision model
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import numpy as np


class PolicyRandom(object):
    def __init__(self, d, name, verbose, omniscient=False):
        """
        param:
            - d: dimension of the action vectors
            - name: additional suffix when comparing several policies (optional)
            - verbose: To print informations
            - omniscient: Does the policy knows when the breakpoints happen ?
        ACTION NORMS ARE SUPPOSED TO BE BOUNDED BE 1
        """
        # immediate attributes from the constructor
        self.name = name
        self.verbose = verbose
        self.omniscient = omniscient

        # attributes for the re-init
        self.t = 0
        self.dim = d
        self.hat_theta = np.zeros(self.dim)

    def select_arm(self, arms):
        """
        Selecting a random action
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(arms) == list, 'List of arms as input required'
        kt = len(arms)  # available actions at time t, k for the moment
        chosen_arm = np.random.randint(kt, dtype='int')
        return chosen_arm

    def update_state(self, features, reward):
        """
        Updating the main parameters for the model, nothing to update in this random policy
        :param features: Feature used for updating
        :param reward: Reward used for updating
        :return: Nothing, but the class instances are updated
        """
        self.t += 1

    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyper-parameters
        """
        self.t = 0

    def __str__(self):
        return 'randomPol' + self.name

    @staticmethod
    def id():
        return 'randomPol'
