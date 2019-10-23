#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Useful functions
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=3)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath} \boldmath"]
styles = ['o', '^', 's', 'D', 'p', 'v', '*']
colors = current_palette[0:11]


def plot_regret(data, t_saved, filename, log=False, qtl=False, loc=0, font=10, bp=None, bp_2=None):
    """
    param:
        - data:
        - t_saved: numpy array (ndim = 1), index of the points to save on each trajectory
        - filename: Name of the file to save the plot of the experiment, if None then it is only plotted
        - log: Do you want a log x-scale
        - qtl: Plotting the lower and upper quantiles. Other effect: If qtl == False then only t_saved
               are printed in the other case everything is printed
        - loc: Location of the legend for fine-tuning the plot
        - font: Font of the legend for fine-tuning the plot
        - bp: Dictionary for plotting the time steps where the breakpoints occur
        - bp_2: Dictionary for plotting the time steps where the breakpoints where detected for d-LinUCB
    Output:
    -------
    Plot it the out/filename file
    """
    fig = plt.figure(figsize=(7, 6))
    if log:
        plt.xscale('log')
    i = 0

    if t_saved is None:
        len_tsaved = len(data[1][1])
        t_saved = [i for i in range(len_tsaved)]

    for key, avgRegret, qRegret, QRegret in data:
        label = r"\textbf{%s}" % key
        plt.plot(t_saved, avgRegret, marker=styles[i],
                 markevery=0.1, ms=10.0, label=label, color=colors[i])
        if qtl:
            plt.fill_between(t_saved, qRegret, QRegret, alpha=0.15,
                             linewidth=1.5, color=colors[i])
        i += 1
    plt.legend(loc=loc, fontsize=font).draw_frame(True)
    plt.xlabel(r'Round $\boldsymbol{t}$', fontsize=20)
    plt.ylabel(r'Regret $\boldsymbol{R(T)}$', fontsize=18)
    for x in bp:
        plt.axvline(x, color='red', linestyle='--', lw=1)
    for x in bp_2:
        plt.axvline(x, color='blue', linestyle='--', lw=1)
    if filename:
        plt.savefig('out/%s.png' % filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def scatter_abrupt(theta_hat, filename, theta, bp, loc=0, font=10, circle=False):
    """
    param
        - theta_hat: Result of the simulator, values of the estimate of the unknown parameter
        - filename: filename for saving the plot
        - theta: True unknown parameter
        - bp: Dictionary for the breakpoints
        - loc: Localisation of the legend
        - font: Size of the legend
        - circle: Draw a circle to help for the visualisation
    Output:
    -------
    Scatter plot in the abruptly-changing environment
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    i = 0
    for key, theta_hat_val in theta_hat.items():
        label = r"\textbf{%s}" % key
        plt.scatter(theta_hat_val[:, 0], theta_hat_val[:, 1],
                    marker=styles[i], label=label, color=colors[i])
        plt.plot(theta_hat_val[:, 0], theta_hat_val[:, 1], color=colors[i], linewidth=0.5, linestyle="--")
        i += 1
    plt.plot(theta[0], theta[1], color=colors[9], linewidth=1.5, linestyle="", markersize=12,
             marker=styles[5], label=r"\textbf{True Param}")
    plt.annotate(r"\textbf{1}", xy=theta, xytext=(-15, +15), textcoords="offset points", fontsize=20)

    for i, val in enumerate(bp.values()):
        plt.plot(val[0], val[1], color=colors[9], linewidth=1.5, linestyle="", markersize=12,
                 marker=styles[5])
        plt.annotate(r"\textbf{%s}" % str(i + 2), xy=val, xytext=(20, 5), textcoords="offset points", fontsize=20)

    plt.legend(loc=loc, fontsize=font).draw_frame(True)
    plt.grid()
    if circle:
        circle1 = plt.Circle((0, 0), 1, color='r', fill=False)
        plt.gcf().gca().add_artist(circle1)
    if filename:
        plt.savefig('out/%s.png' % filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def scatter_smooth(theta_hat, filename, theta_true, loc=0, font=10, circle=False):
    """
    param:
        - theta_hat: the data used for creating the plot
        - filename: Name of the file to save the plot of the experiment
        - theta_true: Evolution of the unknown parameter
        - loc: Position of the legend on the plot
        - font: Size of the chars
        - circle: Draw the circle around the data
    Output:
    -------
    Plot it the out/filename file or in the jupyter notebook
    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-0.05, 1))
    ax.set_ylim((0, 1.1))
    i = 0
    for key, theta_hat_val in theta_hat.items():
        label = r"\textbf{%s}" % key
        plt.scatter(theta_hat_val[:, 0], theta_hat_val[:, 1],
                    marker=styles[i],
                    label=label, color=colors[i])
        i += 1
    plt.plot(theta_true[:, 0], theta_true[:, 1], color=colors[9], linewidth=1.5, linestyle="", markersize=10,
             marker=styles[5], label=r"\textbf{True Param}")
    plt.legend(loc=loc, fontsize=font).draw_frame(True)
    if circle:
        circle1 = plt.Circle((0, 0), 1, color='r', fill=False)
        plt.gcf().gca().add_artist(circle1)

    if filename:
        plt.savefig('out/%s.png' % filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def generate_smooth_theta(t, T, R, angle_init, angle_end, d):
    """
    Function allowing us to generate theta^{\star} parameter that are evolving smoothly.
    Knowing the initial angle (angle_init) and the final angle (angle_end), the parameter
    at time t will be a combination of the initial angle and the final one.
    Warning: 2D-vectors only
    param:
        - t: Current time step
        - T: Horizon time of the experiment
        - R: Radius of the circle (if unit circle then R=1)
        - angle_init: Initial Angle
        - angle_end: Final Angle
        - d: Dimensionality of the vectors in the bandit problem
    Output:
    -------
    Return x,y coordinates
    """
    angle_t = angle_init + t/T*(angle_end - angle_init)
    if d == 2:
        x_t = R * np.cos(angle_t)
        y_t = R * np.sin(angle_t)
        return np.array([x_t, y_t])
    else:
        return 0


def action_check(a_check, t):
    """
    Plotting the different actions received at time t
    param:
        - a_check: Action vectors
        - t: Time instant
    """
    x = a_check[t, :, 0]
    y = a_check[t, :, 1]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.05, 1.05))
    ax.set_ylim((-1.05, 1.05))
    plt.scatter(x, y)
    plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color='r', fill=False))
    plt.show()


def detection_sorted(dic):
    """
    Function for the detection when using the dLinUCB algorithm (file dLinUCB_class.py)
    param:
        - dic: Dictionary of the breakpoints
    """
    dico_new = {}
    for key in sorted(dic.keys()):
        dico_new[key] = dic[key]
    return dico_new


def get_B_T_smooth(steps, R, angle_init, angle_end, d):
    """
    Computing the B_T value when using the generate_smooth_theta function for the
    unknown regression parameters
    param:
        - steps: Number of steps for the smooth changes
        - R: Radius of the circle
        - angle_init: Initial angle
        - angle_end: Final angle
        - d: Dimension of the problem
    """
    res = 0
    theta = generate_smooth_theta(0, steps, R, angle_init, angle_end, d)
    for t in range(1, steps):
        temp = generate_smooth_theta(t, steps, R, angle_init, angle_end, d)
        res += np.linalg.norm(temp - theta)
        theta = temp
    return res
