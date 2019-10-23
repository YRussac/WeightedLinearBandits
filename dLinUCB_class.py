#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Class extracted from https://github.com/qw2ky/NonstationaryBanditLib modified
    to have something consistent with the simulator we have built.
    Code associated with the paper: Qingyun Wu, Naveen Iyer, and Hongning Wang. Learning Contextual Bandits in a
    Non-stationary Environment. The 41th International ACM SIGIR Conference on Research and Development in Information
    Retrieval (SIGIR'2018), https://doi.org/10.1145/3209978.3210051
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import numpy as np
from LinUCB_class import PolicyLinUCB
from scipy.special import erfinv


class MasterBandit(object):
    """
    Implementation of the algorithm of the article Learning Contextual Bandits in a Non-Stationary
    Environment. This class designs the master bandit in the algorithm.
    The master bandit allow us to choose the best candidate among the slave bandits
    """
    def __init__(self):
        """
        Initialization of the MasterBandit
        """
        self.SLAVES = []  # will contain different LinUCB slave models
        self.selectedAlg = None

    def re_init(self):
        """
        :return: set the attributes of the master bandit to the initial ones
        """
        self.SLAVES = []
        self.selectedAlg = None


class SlaveLinUCB(PolicyLinUCB):
    """
    Implementation of the algorithm of the article Learning Contextual Bandits in a Non-Stationary
    Environment. This class designs the structure of the slave bandits
    """
    def __init__(self, d, delta, delta_2, alpha, lambda_, s, sigma_noise, create_time):
        """
        Initialization of an Slave Object. This class inherit from the PolicyLinUCB class
        Therefore, it is assumed here that all the slaves are LinUCB models.
        param:
            - d: dimension of the action vectors
            - delta: probability of theta in the confidence bound
            - delta_2: Confidence bound for the Badness (see article)
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: Constant such that $\Vert \theta \Vert \leq S$
            - sigma_noise: square root of the variance of the noise of the reward function
            - create_time: instant time when a slave was created
        """
        PolicyLinUCB.__init__(self, d, delta, alpha, lambda_, s, '', True, sigma_noise, False)
        self.alpha = alpha
        self.create_time = create_time
        self.slave_time = create_time
        self.lambda_ = lambda_
        self.sigma_noise = sigma_noise
        self.delta_2 = delta_2
        self.badness = 0.
        self.badness_CB = 0.5
        self.fail = 0.0
        self.success = 0.0
        self.failList = []

    def get_badness(self, tau):
        """
        param
            - tau: Number of previous steps considered for estimating the badness
        Output:
        -------
        Estimate of the badness, and Confidence Bound on the Badness
        """
        if len(self.failList) < tau:
            obs_num = len(self.failList)
            self.badness = sum(self.failList) / obs_num
        else:
            obs_num = tau
            self.badness = sum(self.failList[-tau:]) / obs_num

        self.badness_CB = np.sqrt(np.log(1 / self.delta_2) / (2 * obs_num))
        return self.badness, self.badness_CB

    def select_arm_slave(self, arms):
        """
        This functions is largely inspired by the selectArm of the LinUCB class,
        the difference lies in the self.slave_time used for computing beta_t
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(arms) == list, 'List of arms as input required'
        k_t = len(arms)  # available actions at time t
        ucb_s = np.zeros(k_t)  # upper-confidence bounds for every action
        beta_t = self.const1 + self.sigma_noise * np.sqrt(self.c_delta + self.dim * np.log(1 + self.slave_time /
                                                                                           (self.lambda_ * self.dim)))
        for (i, a) in enumerate(arms):
            a = a.features
            invcov_a = np.inner(self.invcov, a.T)
            ucb_s[i] = np.dot(self.hat_theta, a) + self.alpha * beta_t * np.sqrt(np.dot(a, invcov_a))
        mixer = np.random.random(ucb_s.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucb_s)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        return chosen_arm

    def update_state_slave(self, features, reward):
        """
        Function for updating a Slave model (LinUCB models)
        param:
            - features: The feature selected at time t that is used for updating
            - reward: The reward used for updating
        Output:
        -------
        All the parameters associated to this slave bandit are updated
        """
        assert isinstance(features, np.ndarray), 'np.array required'
        aat = np.outer(features, features.T)
        self.cov = self.cov + aat
        self.b = self.b + reward * features
        a = features[:, np.newaxis]
        const = 1 / (1 + np.dot(features, np.inner(self.invcov, features)))
        const2 = np.matmul(self.invcov, a)
        self.invcov = self.invcov - const * np.matmul(const2, const2.T)
        self.hat_theta = np.inner(self.invcov, self.b)
        self.slave_time += 1

    def slave_prediction_info(self, feature):
        """
        Function for getting the information on a slave, necessary for knowing if the model is kept
        updated or deleted by the master bandit
        param:
            - feature: The feature of interest
        Output:
        -------
        Dictionary containing mean, var, beta_t keys
        Rq: Keys of the dictionary are probably not well chosen, it is consistent to the initial code of the article
        """
        a = feature
        mean = np.dot(self.hat_theta, a)
        var = np.inner(self.invcov, a.T)
        var = np.sqrt(np.dot(a, var))
        beta_t = self.const1 + self.sigma_noise * np.sqrt(self.c_delta + self.dim * np.log(1 + self.slave_time /
                                                                                           (self.lambda_ * self.dim)))
        return {'mean': mean, 'var': var, 'beta_t': beta_t}


class DynamicLinUCB(object):
    """
    Implementation of the dLinUCB policy
    """
    def __init__(self, d, delta=0.01, alpha=1, lambda_=0.01, s=1, tau=200, name='', filename='', sm=True, sigma_noise=1,
                 delta_2=0.01, tilde_delta=0.002, verbose=False, omniscient=False):
        """
        Initialization of the model
        param:
            - d: dimension of the action vectors
            - delta: probability of theta in the confidence bound
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: constant such that L2 norm of theta smaller than s
            - tau: Number of previous steps considered for estimating the badness (usually 200)
            - name: additional suffix when comparing several policies (optional)
            - sm: Should Sherman-Morisson formula be used for inverting matrices ?
            - sigma_noise: square root of the variance of the noise
            - delta_2: Confidence bound for the Badness (see article)
            - tilde_delta: Parameter for not creating new model (see article)
            - verbose: To print information
            - omniscient: Does the policy knows when the breakpoints happen ?
        """
        self.dim = d
        self.delta = delta
        self.delta_2 = delta_2
        self.tilde_delta = tilde_delta
        self.alpha = alpha
        self.lambda_ = lambda_
        self.sigma_noise = sigma_noise
        self.tau = tau
        self.name = name
        self.sm = sm
        self.omniscient = omniscient
        self.verbose = verbose
        self.s = s
        self.eta = np.sqrt(2) * sigma_noise * erfinv(1 - self.delta)
        self.t = 0
        self.master = MasterBandit()
        self.master.SLAVES.append(SlaveLinUCB(self.dim, self.delta, self.delta_2, self.alpha,
                                              self.lambda_, self.s, self.sigma_noise, create_time=self.t))
        self.hat_theta = self.master.SLAVES[0].hat_theta
        self.detections = {}
        self.filename = filename

    def select_arm(self, arms):
        """
        Selecting an arm according to the dLinUCB policy
        The function will first select the best slave model based on the Lower Bound for the
        badness, then the best actions is selected for this slave model
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        # Selection of a Slave based on the LCB of the Badness
        min_badness = float('+inf')
        best_slave = self.master.SLAVES[0]
        for slave in self.master.SLAVES:
            lcb_badness = slave.badness - np.sqrt(np.log(self.tau)) * slave.badness_CB
            if lcb_badness < min_badness:
                min_badness = lcb_badness
                best_slave = slave

        idx_slave = self.master.SLAVES.index(best_slave)
        # The correct slave is now chosen
        self.master.selectedAlg = best_slave
        self.hat_theta = self.master.SLAVES[idx_slave].hat_theta
        return best_slave.select_arm_slave(arms)

    def update_state(self, features, reward):
        """
        Updating the main parameters for the different slaves models
        param:
            - features: Feature used for updating
            - reward: Reward used for updating
        Output:
        -------
        Nothing, but the class instances are updated
        """
        self.t += 1
        # Create a new slave model as long as the reward is beyond the reward confidence bound
        # We are out of bound only if EVERY slave is out of bounds i.e. none of them are in bounds
        # If one slave is in bound, we therefore set CreateNewFlag to false
        create_new_flag = True
        to_remove_alg = []
        for slave in self.master.SLAVES:
            data = slave.slave_prediction_info(features)
            slave_total_reward_diff = abs(data['mean'] - reward)
            if slave_total_reward_diff <= data['var'] * data['beta_t'] + self.eta:
                # Updating the good models
                slave.success += 1
                slave.update_state_slave(features, reward)
                slave.failList.append(0)  # Not a failure
            else:
                slave.fail += 1
                slave.failList.append(1)  # Failure for this model

            slave_badness, slave_badness_cb = slave.get_badness(self.tau)
            if slave_badness < slave_badness_cb + self.tilde_delta:
                # At least one slave is not too bad
                create_new_flag = False
            elif slave_badness >= slave_badness_cb + self.delta:
                # Detection of bad slaves
                to_remove_alg.append(slave)

        # Discarding bad slaves
        for slave in to_remove_alg:
            self.master.SLAVES.remove(slave)

        # Create new slave in CreateNewFlag is True or all models were deleted
        if create_new_flag or len(self.master.SLAVES) == 0:
            f = open('detection_out/%s' % str(self.filename), "a")
            f.write(str(self.t) + '\n')
            f.close()
            print('Creation of a new model at time:' + str(self.t))
            if self.t in self.detections.keys():
                self.detections[self.t] += 1
            else:
                self.detections[self.t] = 1
            self.master.SLAVES.append(SlaveLinUCB(self.dim, self.delta, self.delta_2, self.alpha,
                                                  self.lambda_, self.s, self.sigma_noise, create_time=self.t))

    def re_init(self):
        """
        Reinitializing the parameter of the model
        """
        self.t = 0
        self.master.re_init()
        self.master.SLAVES.append(SlaveLinUCB(self.dim, self.delta, self.delta_2, self.alpha,
                                              self.lambda_, self.s, self.sigma_noise, create_time=self.t))

    def __str__(self):
        return 'dLinUCB' + self.name

    @staticmethod
    def id():
        return 'dLinUCB'
