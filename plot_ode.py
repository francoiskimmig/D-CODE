import argparse
import pickle
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.integrate import solve_ivp
from scipy import interpolate
import pathlib

import data
import equations

from gp_utils import *
from interpolate import get_ode_data
from collections import Counter
from scipy.stats import wasserstein_distance
import scipy.signal as ss
from matplotlib.mlab import psd
import ot

def compute_wasserstein_fourier_dist(signal_1, signal_2, sampling_frequency):
    (ffre1, S1) = ss.periodogram(signal_1[:,0], sampling_frequency, scaling='density')
    (ffre2, S2) = ss.periodogram(signal_2[:,0], sampling_frequency, scaling='density')
    
    S1 /= np.sum(S1)
    S2 /= np.sum(S2)

    distance = ot.wasserstein_1d(ffre1, ffre2, S1, S2, p = 2)
    
    return distance



def set_unique_legend(axe):
    handles, labels = axe.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axe.legend(by_label.values(), by_label.keys())  

def std_RMSE(err_sq):
    rmse_list = []
    for i in range(500):
        new_err = err_sq[np.random.randint(0, len(err_sq), err_sq.shape)]
        rmse_itr = np.sqrt(np.mean(new_err))
        rmse_list.append(rmse_itr)
    return np.std(np.array(rmse_list))

def testo(pickle_file, title_prefix):

    with open(pickle_file, "rb") as f:
        res = pickle.load(f)
    
    original_signal = res["original"]
    computed_signal = res["computed"]
    time_array = res["time_array"]
    discovered_ode = res["discovered_ode"]


    rmse = std_RMSE((original_signal - computed_signal) ** 2)
    sampling_freq = 125
    wassertein = compute_wasserstein_fourier_dist(original_signal, computed_signal, sampling_freq)

    fig1, ax1 = plt.subplots(1, 1, figsize = (12,8), sharex=False)
    ax1.plot(time_array, original_signal, c = "b",label = "original")
    ax1.plot(time_array, computed_signal, c = "r", label = "reconstructed")
    ax1.set_xlabel("Subsequence")
    ax1.set_ylabel("PPG Trajectory")
    fig1.suptitle(title_prefix + "\n ODE: " + discovered_ode + "\n RMSE: {:.4e}".format(rmse)
                   + "\n Wassertein: {:.4e}".format(wassertein))
    set_unique_legend(ax1)
    fig1.savefig("Figures/" + title_prefix + ".pdf")

if __name__ == "__main__":
    pickle_file = "results/solved_ode/ppg_press_press_dot_39_solved.pkl"
    title_prefix = "5 cardiac cycles subsequences"
    testo(pickle_file, title_prefix)

    pickle_file = "results/solved_ode/ppg_press_press_dot_3_cycle39_solved.pkl"
    title_prefix = "3 cardiac cycles subsequences"
    testo(pickle_file, title_prefix)
    title_prefix = "1 cardiac cycle subsequences"
    pickle_file = "results/solved_ode/ppg_press_press_dot_1_cycle39_solved.pkl"
    testo(pickle_file, title_prefix)
    plt.show()
