# -*- coding: utf-8 -*-

"""
    - author: Yoan Russac
    - Simulator file used with the exponential weight: w_t = gamma^(-t)
"""

# Importation

import numpy as np
import multiprocessing
from tqdm import tqdm
import time
from utils import generate_smooth_theta


class Simulator(object):
    """
    Simulator of stochastic games.

    Params:
    -------
    MAB: list
        List of arms.

    policies: list
        List of policies to test.

    K: int
        Number of items (arms) in the pool.

    d: int
        Dimension of the problem

    T: Number of steps in each round

    """

    def __init__(self, mab, theta, policies, k, d, steps, bp_dico, verbose):
        """"
        global init function, initializing all the internal parameters that are fixed
        """
        self.policies = policies
        self.mab = mab
        self.theta = theta
        self.steps = steps
        self.d = d
        self.k = k
        self.verbose = verbose
        self.bp_dico = bp_dico
        if self.verbose:
            print("real theta is ", self.mab.theta)

    def from_proxy_dict_to_cumregret(self, regret, n_mc, steps, n_process, t_saved):
        """
        Post-processing of the regret object after the computation on all processors
        :param regret: Manager dict, this is the object exchanged between the different processors.
        :param n_mc: Number of Monte-Carlo launched on a process. The total number of experiment will be
        n_mc * n_process (int)
        :param steps: Time horizon for one experiment (int)
        :param n_process: Number of processors used (int)
        :param t_saved: Trajectory of points saved to store fewer than steps points on a trajectory.
                        (numpy array ndim = 1)
        :return:
        """
        if t_saved is None:
            t_saved = [i for i in range(steps)]
        tdeb = time.time()
        dico_pol = {}
        n_policies = len(self.policies)
        for policy in self.policies:
            name = policy.__str__()
            dico_pol[name] = np.zeros((n_process * n_mc, len(t_saved)))
        for nb_process in range(n_process):
            for i, policy in enumerate(self.policies):
                name = policy.__str__()
                for nb_exp in range(n_mc):
                    temp = np.cumsum(regret[nb_process][nb_exp * n_policies * steps + i * steps:
                                                        nb_exp * n_policies * steps + i * steps + steps])
                    dico_pol[name][nb_process * n_mc + nb_exp, :] = np.take(temp, t_saved)
        print('Conversion time:', time.time() - tdeb)
        return dico_pol

    def simulation_parallel(self, l, steps, n_mc, regret):
        """
        Function that is launched on a single processor of the machine
        The MAB is played during the entire trajectory, same function as the run function but with modifications
        for running on a processor
        :param l: Number of the processor used (int)
        :param steps: Time horizon for one experiment (int)
        :param n_mc: Number of Monte-Carlo launched on a process. The total number of experiment will be
        n_mc * n_process (int)
        :param regret: Manager dict, this is the object exchanged between the different processors.
        :return: None but regret will contain the result of all the experiments for all the policies launched
        """
        np.random.seed()
        reg_res = np.array([0.] * (steps * n_mc * len(self.policies)))

        for policy in self.policies:
            name = policy.__str__()

        for nExp in range(n_mc):
            for i, policy in enumerate(self.policies):
                policy.re_init()  # Reinitialize the policy
                self.mab.theta = self.theta
                optimal_rewards = np.zeros(steps)
                rewards = np.zeros(steps)
                for t in range(steps):
                    if t in self.bp_dico.keys():
                        theta_new = self.bp_dico[t]
                        self.mab.theta = theta_new
                        if policy.omniscient:
                            policy.re_init()
                    available_arms = self.mab.get_arms(self.k)  # receiving K action vectors
                    _, instant_best_reward = self.mab.get_best_arm()
                    a_t = policy.select_arm(available_arms)
                    round_reward, a_t_features = self.mab.play(a_t)
                    policy.update_state(a_t_features, round_reward)
                    expected_reward_round = self.mab.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward
                    rewards[t] = expected_reward_round
                    reg_res[nExp * len(self.policies) * steps + i * steps + t] = (optimal_rewards[t] - rewards[t])
        regret[l] = reg_res
        return None

    def simulation_parallel_smooth(self, l, step_1, steps, n_mc, R, angle_init, angle_end, regret):
        """
        Function that is launched on a single processor of the machine
        The MAB is played during the entire trajectory, same function as the run function but with modifications
        for running on a processor
        :param l: Number of the processor used (int)
        :param step_1: Length of sequential allocations with smoothly changing environment (int)
        :param steps: Time horizon for one experiment (int)
        :param n_mc: Number of Monte-Carlo launched on a process. The total number of experiment will be
        n_mc * n_process (int)
        :param angle_init: Initial angle for the smoothly changing environment (float)
        :param angle_end: Final angle for the smoothly changing environment (float)
        :param regret: Manager dict, this is the object exchanged between the different processors.
        :return: None but regret will contain the result of all the experiments for all the policies launched
        """
        np.random.seed()
        reg_res = np.array([0.] * (steps * n_mc * len(self.policies)))
        for nExp in range(n_mc):
            for i, policy in enumerate(self.policies):
                policy.re_init()  # Reinitialize the policy
                self.mab.theta = generate_smooth_theta(0, step_1, R, angle_init, angle_end, self.d)
                optimal_rewards = np.zeros(steps)
                rewards = np.zeros(steps)
                for t in range(steps):
                    if t <= step_1:
                        self.mab.theta = generate_smooth_theta(t, step_1, R, angle_init, angle_end, self.d)
                    available_arms = self.mab.get_arms(self.k)  # receiving K action vectors
                    _, instant_best_reward = self.mab.get_best_arm()
                    a_t = policy.select_arm(available_arms)
                    round_reward, a_t_features = self.mab.play(a_t)
                    policy.update_state(a_t_features, round_reward)
                    expected_reward_round = self.mab.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward
                    rewards[t] = expected_reward_round
                    reg_res[nExp * len(self.policies) * steps + i * steps + t] = (optimal_rewards[t] - rewards[t])
        regret[l] = reg_res
        return None

    def run_multiprocessing(self, n_process, steps, n_mc, q, t_saved=None):
        """
        Running the experiments on several processors with the use of simulation_parallel function
        :param n_process: Number of processors used (int)
        :param steps: Number of steps for an experiment (int)
        :param n_mc: Total number of Monte Carlo experiment (int)
        :param q: Quantile (int). ex: q=5%
        :param t_saved: Trajectory of points saved to store fewer than steps points on a trajectory.
                        (numpy array ndim = 1)
        :return: regret, lower quantile, upper quantile for every policy
        """
        t0 = time.time()
        manager = multiprocessing.Manager()
        regret = manager.dict()
        avg_regret = dict()
        q_regret = dict()
        up_q_regret = dict()
        jobs = [multiprocessing.Process(target=self.simulation_parallel,
                                        args=(j, steps, n_mc // n_process,
                                              regret))
                for j in range(n_process)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        t1 = time.time() - t0
        print('Total Time Experiment: ', t1)
        dico_pol = self.from_proxy_dict_to_cumregret(regret, n_mc // n_process, steps, n_process, t_saved)
        t = time.time()
        for policy in self.policies:
            name = policy.__str__()
            avg_regret[name] = np.mean(dico_pol[name], 0)
            q_regret[name] = np.percentile(dico_pol[name], q, 0)
            up_q_regret[name] = np.percentile(dico_pol[name], 100 - q, 0)
        print('Post-Processing Time:', time.time() - t)
        return avg_regret, q_regret, up_q_regret

    def run_multiprocessing_smooth(self, n_process, step_1, steps, n_mc, R, angle_init, angle_end, q, t_saved=None):
        """
        Running the experiments on several processors with the use of simulation_parallel_smooth function
        :param n_process: Number of processors used (int)
        :param steps: Number of steps for an experiment (int)
        :param n_mc: Total number of Monte Carlo experiment (int)
        :param q: Quantile (int). ex: q=5%
        :param t_saved: Trajectory of points saved to store fewer than steps points on a trajectory.
                        (numpy array ndim = 1)
        :return: regret, lower quantile, upper quantile for every policy
        """
        t0 = time.time()
        manager = multiprocessing.Manager()
        regret = manager.dict()
        avg_regret = dict()
        q_regret = dict()
        up_q_regret = dict()
        jobs = [multiprocessing.Process(target=self.simulation_parallel_smooth,
                                        args=(j, step_1, steps, n_mc // n_process, R, angle_init, angle_end, regret))
                for j in range(n_process)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        t1 = time.time() - t0
        print('Total Time Experiment: ', t1)
        dico_pol = self.from_proxy_dict_to_cumregret(regret, n_mc // n_process, steps, n_process, t_saved)
        t = time.time()
        for policy in self.policies:
            name = policy.__str__()
            avg_regret[name] = np.mean(dico_pol[name], 0)
            q_regret[name] = np.percentile(dico_pol[name], q, 0)
            up_q_regret[name] = np.percentile(dico_pol[name], 100 - q, 0)
        print('Post-Processing Time:', time.time() - t)
        return avg_regret, q_regret, up_q_regret

    def run(self, steps, n_mc, q, n_scatter, t_saved=None):
        """
        Runs an experiment with steps points and n_mc Monte Carlo repetition
        :param steps: Number of steps for an experiment (int)
        :param n_mc: Total number of Monte Carlo experiment (int)
        :param q: Quantile (int). ex: q=5%
        :param n_scatter: Frequency of the plot of the estimate for the scatter plot (only in 2D problems)
        :param t_saved: Trajectory of points saved to store fewer than steps points on a trajectory.
                        (numpy array ndim = 1)
        :return:
        """
        if t_saved is None:
            t_saved = [i for i in range(steps)]
        cum_regret = dict()
        n_sub = np.size(t_saved)  # Number of points saved for each trajectory
        avg_regret = dict()
        q_regret = dict()
        up_q_regret = dict()
        timedic = dict()
        theta_hat = dict()

        for policy in self.policies:
            name = policy.__str__()
            cum_regret[name] = np.zeros((n_mc, n_sub))
            timedic[name] = 0
            if self.d == 2:
                theta_hat[name] = np.zeros((steps // n_scatter, self.d))
                # theta_true = np.zeros((steps // n_scatter, self.d))
                action_check = np.zeros((steps // n_scatter, self.k, self.d))

        # run n_mc independent simulations
        for nExp in tqdm(range(n_mc)):
            if self.verbose:
                print('--------')
                print('Experiment number: ' + str(nExp))
                print('--------')

            for i, policy in enumerate(self.policies):
                # Reinitialize the policy
                time_init_pol = time.time()
                policy.re_init()
                self.mab.theta = self.theta
                name = policy.__str__()
                optimal_rewards = np.zeros(steps)
                rewards = np.zeros(steps)
                for t in range(steps):
                    if t in self.bp_dico.keys():
                        theta_new = self.bp_dico[t]
                        self.mab.theta = theta_new
                        if policy.omniscient:
                            policy.re_init()
                    if self.d == 2:
                        if (nExp == 0) and (t % n_scatter == 0):
                            print(t, policy.__str__(), policy.hat_theta, self.mab.theta)
                        if t % n_scatter == 0:
                            nb_line = t // n_scatter
                            if nExp == 0:
                                # theta_true[nb_line, :] = self.mab.theta
                                theta_hat[name][nb_line, :] = policy.hat_theta
                            else:
                                theta_hat[name][nb_line, :] = 1 / (nExp + 1) * policy.hat_theta + nExp / (nExp + 1) * \
                                                              theta_hat[name][nb_line, :]
                    if self.verbose:
                        print('time t=' + str(t))

                    available_arms = self.mab.get_arms(self.k)  # receiving K action vectors

                    if self.d == 2:
                        if (nExp == 0) and (t % n_scatter == 0):
                            nb_line = t // n_scatter
                            for nb_arm in range(self.k):
                                action_check[nb_line, nb_arm, :] = np.array(available_arms[nb_arm].features)

                    _, instant_best_reward = self.mab.get_best_arm()
                    a_t = policy.select_arm(available_arms)  # number of the action
                    round_reward, a_t_features = self.mab.play(a_t)  # action_played is the feature vector
                    policy.update_state(a_t_features, round_reward)
                    expected_reward_round = self.mab.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward
                    rewards[t] = expected_reward_round
                if self.verbose:
                    print('optimal_rewards: ', optimal_rewards)
                    print('rewards: ', rewards)
                    print('regret: ', cum_regret[name])
                cum_regret[name][nExp, :] = np.cumsum(optimal_rewards - rewards)[t_saved]
                time_end_pol = time.time() - time_init_pol
                timedic[name] += time_end_pol

        print("-- Building data out of the experiments ---")

        for policy in self.policies:
            name = policy.__str__()
            cum_reg = cum_regret[name]  # Cumulative regret only on the t_saved points
            avg_regret[name] = np.mean(cum_reg, 0)
            q_regret[name] = np.percentile(cum_reg, q, 0)
            up_q_regret[name] = np.percentile(cum_reg, 100 - q, 0)

        print("--- Data built ---")
        return avg_regret, q_regret, up_q_regret, timedic, theta_hat, action_check

    def run_smooth_environment(self, step_1, steps, n_mc, q, R, angle_init, angle_end, n_scatter, n_scatter_true,
                               t_saved=None):
        """
        Runs an experiment with parameters T and N.
        Warning d should be 2-dimensional vector

        It returns a dictionary whose keys are policies and whose values
        are the regret obtained by these policies over the experiments and
        averaged over N runs.

        Parameters
        ----------
        step_1: int
            Length of the sequential allocations with smooth modifications

        steps: int
            Length of the sequential allocations without modifications

        n_mc: int
            Number of Monte Carlo repetitions.

        q: int
            Quantile parameter (e.g. 25 -> quartiles)

        angle_init: float
            Initial angle for the smoothly changing environment

        angle_end: float
            Final angle for the smoothly changing environment

        t_saved: numpy array (ndim = 1)
            Points to save on each trajectory.
            Index of the points to save

        n_scatter: How frequently we plot the estimate for the scatter plot

        n_scatter_true: How frequently we plot the evolution of the true unknown parameter
        """
        if t_saved is None:
            t_saved = [i for i in range(steps)]
        cum_regret = dict()
        n_sub = np.size(t_saved)  # Number of points saved for each trajectory
        avg_regret, q_regret, up_q_regret = dict(), dict(), dict()
        timedic = dict()
        theta_hat = dict()

        for policy in self.policies:
            name = policy.__str__()
            cum_regret[name] = np.zeros((n_mc, n_sub))
            timedic[name] = 0
            if self.d == 2:
                theta_hat[name] = np.zeros((steps // n_scatter, self.d))
                theta_true = np.zeros((step_1 // n_scatter_true + 1, self.d))

        # run N independent simulations
        for nExp in tqdm(range(n_mc)):
            if self.verbose:
                print('--------')
                print('Experiment number: ' + str(nExp))
                print('--------')

            for i, policy in enumerate(self.policies):
                time_init_pol = time.time()
                # Reinitialize the policy
                policy.re_init()
                self.mab.theta = generate_smooth_theta(0, step_1, R, angle_init, angle_end, self.d)
                name = policy.__str__()
                optimal_rewards = np.zeros(steps)
                rewards = np.zeros(steps)
                for t in range(steps):
                    if t <= step_1:
                        self.mab.theta = generate_smooth_theta(t, step_1, R, angle_init, angle_end, self.d)
                        if self.d == 2:
                            if t % n_scatter_true == 0:
                                nb_line = t // n_scatter_true
                                if nExp == 0:
                                    theta_true[nb_line, :] = self.mab.theta
                        #  smooth modification of the theta value
                    if self.d == 2:
                        if (nExp == 0) and (t % n_scatter == 0):
                            print(t, policy.__str__(), policy.hat_theta, self.mab.theta)
                        if t % n_scatter == 0:
                            nb_line = t // n_scatter
                            if nExp == 0:
                                theta_hat[name][nb_line, :] = policy.hat_theta
                            else:
                                theta_hat[name][nb_line, :] = 1 / (nExp + 1) * policy.hat_theta + nExp / (nExp + 1) * \
                                                              theta_hat[name][nb_line, :]
                    if self.verbose:
                        print('time t=' + str(t))
                    available_arms = self.mab.get_arms(self.k)  # receiving K action vectors
                    _, instant_best_reward = self.mab.get_best_arm()
                    a_t = policy.select_arm(available_arms)  # number of the action
                    round_reward, a_t_features = self.mab.play(a_t)  # action_played is the feature vector
                    policy.update_state(a_t_features, round_reward)  # updating with the noisy reward
                    expected_reward_round = self.mab.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward
                    rewards[t] = expected_reward_round
                if self.verbose:
                    print('optimal_rewards: ', optimal_rewards)
                    print('rewards: ', rewards)
                    print('regret: ', cum_regret[name])
                cum_regret[name][nExp, :] = np.cumsum(optimal_rewards - rewards)[t_saved]
                time_end_pol = time.time() - time_init_pol
                timedic[name] += time_end_pol

        print("-- Building data out of the experiments ---")

        for policy in self.policies:
            name = policy.__str__()
            cum_reg = cum_regret[name]  # Cumulative regret only on the t_saved points
            avg_regret[name] = np.mean(cum_reg, 0)
            q_regret[name] = np.percentile(cum_reg, q, 0)
            up_q_regret[name] = np.percentile(cum_reg, 100 - q, 0)

        print("--- Data built ---")
        return avg_regret, q_regret, up_q_regret, timedic, theta_true, theta_hat
