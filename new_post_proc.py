import argparse
import pickle
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import numpy as np
import json
from scipy.integrate import solve_ivp, trapz
from scipy import interpolate
import pathlib

import data
import equations

from gp_utils import *
from interpolate import get_ode_data
from collections import Counter
import scipy.signal as ss
from matplotlib.mlab import psd
import ot


def set_unique_legend(axe):
    handles, labels = axe.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axe.legend(by_label.values(), by_label.keys())  


def compute_wasserstein_fourier_dist(signal_1, signal_2, sampling_frequency):
    (ffre1, S1) = ss.periodogram(signal_1, sampling_frequency, scaling='density')
    (ffre2, S2) = ss.periodogram(signal_2, sampling_frequency, scaling='density')
    
    S1 /= np.sum(S1)
    S2 /= np.sum(S2)

    distance = ot.wasserstein_1d(ffre1, ffre2, S1, S2, p = 2)
    
    return distance**(1/2)

def compute_Lp_distance(signal_1, signal_2, p=1):

    if (len(signal_1) != len(signal_2)):
        raise ValueError('vector-like argument must have the same length')
    
    return 1/len(signal_1) * np.linalg.norm(signal_1 - signal_2, ord=p)


def compute_fourier_coeff(n_freq, times, signal):
    period = times[-1] - times[0]


    if n_freq > len(times):
        raise ValueError("n_freq should be lower or equal to the number of times points")
    
    if n_freq < 0:
        raise ValueError("'n_freq' must be positive")

    if (len(times) != len(signal)):
        raise ValueError("'times' and 'signal' arguments should have the same length")

    frequencies = 1/period * np.arange(-n_freq, n_freq+1, 1)

    coeff_list = np.zeros(len(frequencies))*1j

    for i, freq in enumerate(frequencies):
        coeff_list[i] = 1/period * trapz(signal * np.exp(-1j * 2*np.pi * (i-n_freq) / period * times), times)
        # coeff_list[i] = 1/period * simpson(signal * np.exp(-1j * 2*np.pi * (i-n_freq) / period * times), times)

    return frequencies, coeff_list

def synthetize_signal(times, freq_list, amplitude_coeff_list, phase_list, time_shift=0):
    value = np.zeros(len(times))

    if (len(freq_list) != len(phase_list)) and (len(freq_list) != len(amplitude_coeff_list)):
        raise ValueError("'freq_list', 'amplitude_coeff_list' and 'phase_list' argument must have the same length")

    for freq, amplitude, phase in zip(freq_list, amplitude_coeff_list, phase_list):

        if (freq == 0):
            value += amplitude
        else:
            value += amplitude * np.cos(2 * np.pi * freq * (times - time_shift) + phase) # factor 2 eliminated because terms are summed twice

    return value

def compute_WF_distance(times_1, signal_1, times_2, signal_2, n_freq=50, p=2):

    freq_1, fourier_coeff_1 = compute_fourier_coeff(n_freq, times_1, signal_1)
    freq_2, fourier_coeff_2 = compute_fourier_coeff(n_freq, times_2, signal_2)

    norm_coeff_1 = np.abs(fourier_coeff_1)**2 / np.linalg.norm(np.abs(fourier_coeff_1)**2, 2)
    norm_coeff_2 = np.abs(fourier_coeff_2)**2 / np.linalg.norm(np.abs(fourier_coeff_2)**2, 2)

    return (ot.wasserstein_1d(freq_1, freq_2, norm_coeff_1, norm_coeff_2, p))**(1/p)



def std_RMSE(err_sq):
    rmse_list = []
    for i in range(500):
        new_err = err_sq[np.random.randint(0, len(err_sq), err_sq.shape)]
        rmse_itr = np.sqrt(np.mean(new_err))
        rmse_list.append(rmse_itr)
    return np.std(np.array(rmse_list))


# def update_slider(res, fig, line_ppg, line_press, line_press_dot, reconstructed_ppg, reconstructed_ppg_subsequence, ax, Ntime_steps, slider_1, slider_2, slider_3):
def update_slider(res, fig, line_ppg, reconstructed_ppg, ax, Ntime_steps, slider_1, slider_2, slider_3):
    def update(val):
        slider_val = slider_1.val
        coeff_1 = slider_2.val
        coeff_2 = slider_3.val

        if slider_val in res:
            Nsubsequences = len(res[slider_val]["ppg"]) / Ntime_steps
            dt = 1  / Ntime_steps
            ppg = res[slider_val]["ppg"]
            press = res[slider_val]["press"]
            press_dot =  res[slider_val]["press_dot"]

            time_array = np.zeros(len(ppg))
            for i in range(1, time_array.shape[0]):
                time_array[i] = time_array[i-1] + dt

            if time_array.shape[0] != len(ppg):
                print("Time array: ", time_array.shape[0])
                print("ppg: ", len(ppg))
                raise ValueError("'time_array' and 'ppg' arguments should have the same length")

            line_ppg.set_data(time_array, ppg)
            # line_press.set_data(time_array, press)
            # line_press_dot.set_data(time_array, press_dot)
            ax.set_xlim([time_array.min(), time_array.max()])

            init_value = ppg[0]
            ppg_reconstructed = np.zeros(time_array.shape[0])
            ppg_reconstructed[0] = init_value
            for i in range(1, time_array.shape[0]):
                ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (coeff_1 * ppg_reconstructed[i-1] * press_dot[i-1] + coeff_2 * press_dot[i-1])

                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (ppg_reconstructed[i-1] * (np.cos(press_dot[i-1] +  3.790234918816544) + np.cos(np.sin(press[i-1]))) + press_dot[i-1])
                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (-7.793805571606759*press[i-1] + (6.716615018977029*press_dot[i-1] + 3.591510630295889)*(np.sin(ppg_reconstructed[i-1])*np.cos(ppg_reconstructed[i-1]) + np.cos(ppg_reconstructed[i-1] + press[i-1])))

                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (coeff_1 * press_dot[i-1]*np.cos(ppg_reconstructed[i-1]))
                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (ppg_reconstructed[i-1] * press_dot[i-1] * (press_dot[i-1] + coeff_1) + coeff_2 * (np.sin(press_dot[i-1]) + np.sin(ppg_reconstructed[i-1] * press_dot[i-1]) + np.cos(ppg_reconstructed[i-1]) - np.cos(ppg_reconstructed[i-1] + press_dot[i-1])))

                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * ((coeff_1 * ppg_reconstructed[i-1] + coeff_2 * press_dot[i-1]) * np.cos(press[i-1]))

                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (coeff_1 * ppg_reconstructed[i-1] + coeff_2 * press_dot[i-1] * np.sin(press_dot[i-1]))

                # ppg_reconstructed[i] = ppg_reconstructed[i-1] + dt * (coeff_1 * (press[i-1]+ press_dot[i-1])*np.sin(ppg_reconstructed[i-1]) + coeff_2 * np.sin(press_dot[i-1]))

            reconstructed_ppg.set_data(time_array, ppg_reconstructed)
            ax.set_ylim([min(min(ppg_reconstructed), min(ppg)), max(max(ppg_reconstructed), max(ppg))])

            subsequence_time_array = np.arange(0, 1 + dt, dt)
            ppg_reconstructed_subsequence = np.zeros(time_array.shape[0])
            Nsubsequences = int(Nsubsequences)
            for i in range(Nsubsequences):
                # ppg_reconstructed_subsequence[i*Ntime_steps] = ppg[i*Ntime_steps]
                for j in range(1, subsequence_time_array.shape[0]-2):
                    ppg_reconstructed_subsequence[i*Ntime_steps + j] = ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + dt * (coeff_1 * ppg_reconstructed_subsequence[i*Ntime_steps + j-1] * press_dot[i*Ntime_steps + j-1] + coeff_2 * press_dot[i*Ntime_steps + j -1 ])

                    ppg_reconstructed_subsequence[i*Ntime_steps + j] = 0

                    # ppg_reconstructed_subsequence[i*Ntime_steps + j] = ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + dt * (coeff_1 * press_dot[i*Ntime_steps + j-1]*np.cos(ppg_reconstructed_subsequence[i*Ntime_steps + j-1]))

                    # ppg_reconstructed_subsequence[i*Ntime_steps + j] = ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + dt * (coeff_1 * ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + coeff_2 * press_dot[i*Ntime_steps + j-1] * np.sin(press_dot[i*Ntime_steps + j-1]))

                    # ppg_reconstructed_subsequence[i*Ntime_steps + j] = ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + dt * (ppg_reconstructed_subsequence[i*Ntime_steps + j-1] * press_dot[i*Ntime_steps + j-1] * (press_dot[i*Ntime_steps + j-1] + coeff_1) + coeff_2 * (np.sin(press_dot[i*Ntime_steps + j-1]) + np.sin(ppg_reconstructed_subsequence[i*Ntime_steps + j-1] * press_dot[i*Ntime_steps + j-1]) + np.cos(ppg_reconstructed_subsequence[i*Ntime_steps + j-1]) - np.cos(ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + press_dot[i*Ntime_steps + j-1])))
                    # ppg_reconstructed_subsequence[i*Ntime_steps + j] = ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + dt * (-7.793805571606759*press[i*Ntime_steps + j-1] + (6.716615018977029*press_dot[i*Ntime_steps + j-1] + 3.591510630295889)*(np.sin(ppg_reconstructed_subsequence[i*Ntime_steps + j-1])*np.cos(ppg_reconstructed_subsequence[i*Ntime_steps + j-1]) + np.cos(ppg_reconstructed_subsequence[i*Ntime_steps + j-1] + press[i*Ntime_steps + j-1])))

            # reconstructed_ppg_subsequence.set_data(time_array, ppg_reconstructed_subsequence)

            # rmse = std_RMSE((np.array(ppg) - ppg_reconstructed_subsequence) ** 2)
            ppg_npa = np.array(ppg)
            # l2_error_continuous = np.sqrt(np.sum(np.power((ppg_npa - ppg_reconstructed), 2)))
            # l2_error_subsequence = np.sqrt(np.sum(np.power((ppg_npa - ppg_reconstructed_subsequence), 2)))
            # l2_error_continuous = compute_Lp_distance(ppg_npa, ppg_reconstructed, 2)
            # l2_error_subsequence = compute_Lp_distance(ppg_npa, ppg_reconstructed_subsequence, 2)
            sampling_freq = 125

            # wassertein_continuous = compute_wasserstein_fourier_dist(ppg_npa, ppg_reconstructed, Ntime_steps)

            # test_wf = compute_WF_distance(time_array, ppg_npa, time_array, ppg_reconstructed)

            # wassertein_subsequence = compute_wasserstein_fourier_dist(ppg_npa, ppg_reconstructed_subsequence, sampling_freq)
            # fig.suptitle("\n \color{red} RMSE: {:.4e}".format(rmse))
            # fig.suptitle("\n Wassertein continuous: {:.4e}".format(wassertein_continuous)
                        #   + "\n Wassertein test: {:.4e}".format(test_wf)
                        #   + "\n Wassertein subsequence: {:.4e}".format(wassertein_subsequence))
                        #   + "\n L2 continuous: {:.4e}".format(l2_error_continuous)
                        #   + "\n L2 subsequence: {:.4e}".format(l2_error_subsequence))
            
            # fig.suptitle("\n \color{red} RMSE: {:.4e}".format(rmse)
            #            + "\n Wassertein: {:.4e}".format(wassertein))
            fig.canvas.draw_idle()

    return update


def void_update(val):
    pass

def main():
    # pickle_file = "real_data/post_proc_ppg_press_press_dot39_Ncycles_5"
    # pickle_file = "real_data/post_proc_ppg_press_press_dot39_Ncycles_1"
    pickle_file = "real_data/post_proc_ppg_press_press_dot65_Ncycles_1"
    
    with open(pickle_file + ".pkl", "rb") as f:
        res = pickle.load(f)

    fig1, ax1 = plt.subplots(1, 1, sharex=True)
    fig1.subplots_adjust(left=0.20, bottom=0.25, right=0.85)
    ax_seq_id = fig1.add_axes([0.25, 0.1, 0.65, 0.03])
    seq_id_slider = Slider(
        ax=ax_seq_id,
        label='Seq ID',
        valmin=1,
        valmax=max(res),
        valinit=1,
        valstep=1,
    )
    

    coeff_1_dict = {"range": [-5, 0], "init": -1}

    # coeff_1_dict = {"range": [0, 10], "init": 2.909}

    # coeff_1_dict = {"range": [0, 10], "init": 3.983506350}

    # coeff_1_dict = {"range": [-5, 5], "init": 1}
    
    # Make a vertically oriented slider to control time shift
    ax_coeff_1 = fig1.add_axes([0.1, 0.25, 0.0225, 0.63])
    coeff_1_slider = Slider(
    ax=ax_coeff_1,
    label="Coeff 1",
    valmin=coeff_1_dict["range"][0],
    valmax=coeff_1_dict["range"][1],
    valinit=coeff_1_dict["init"],
    orientation="vertical"
)

    coeff_2_dict = {"range": [0, 10], "init": 1.85834}


    # coeff_2_dict = {"range": [0, 20], "init": 8.27229926227}

    # coeff_2_dict = {"range": [-5, 5], "init": 1}

    ax_coeff_2 = fig1.add_axes([0.9, 0.25, 0.0225, 0.63])
    coeff_2_slider = Slider(
    ax=ax_coeff_2,
    label="Coeff 2",
    valmin=coeff_2_dict["range"][0],
    valmax=coeff_2_dict["range"][1],
    valinit=coeff_2_dict["init"],
    orientation="vertical"
)

    seq_id = min(res)
    # seq_id = 2
    ppg = res[seq_id]["ppg"]
    press = res[seq_id]["press"]
    press_dot = res[seq_id]["press_dot"]


    line_ppg, = ax1.plot(ppg, label = "original_ppg")
    ax1.set_ylabel("AC PPG")
    reconstructed_ppg = Line2D(np.arange(len(ppg)), ppg, color='red', label = "r_complete")
    ax1.add_line(reconstructed_ppg)
    set_unique_legend(ax1)


    # line_ppg, = ax1[0].plot(ppg)
    # ax1[0].set_ylabel("AC PPG")
    # line_press, = ax1[1].plot(press)
    # ax1[1].set_ylabel("Pressure")
    # line_press_dot, = ax1[2].plot(press_dot)
    # ax1[2].set_ylabel("$\dot{P}_{\mathrm{wrist}}$")
    # reconstructed_ppg = Line2D(np.arange(len(ppg)), ppg, color='red', label = "r_complete")
    # ax1[0].add_line(reconstructed_ppg)
    # reconstructed_ppg_subsequence = Line2D(np.arange(len(ppg)), ppg, color='green', label="r_subsequence")
    # ax1[0].add_line(reconstructed_ppg_subsequence)
    # for ax in ax1:
    #     set_unique_legend(ax)

    # training_pickle_file = "real_data/ppg_press_press_dot_39_Ncycles_5"
    # training_pickle_file = "real_data/ppg_press_press_dot_39_Ncycles_1"
    training_pickle_file = "real_data/ppg_press_press_dot65_Ncycles_1"
    with open(training_pickle_file + ".pkl", "rb") as f:
        training_data = pickle.load(f)
    # Ntime_steps =  training_data["data"].shape[0] / 5
    Ntime_steps =  training_data["data"].shape[0]
    
    update = update_slider(res, fig1, line_ppg, reconstructed_ppg, ax1, Ntime_steps, seq_id_slider, coeff_1_slider, coeff_2_slider)

    # update = update_slider(res, fig1, line_ppg, line_press, line_press_dot, reconstructed_ppg, reconstructed_ppg_subsequence, ax1[0], Ntime_steps, seq_id_slider, coeff_1_slider, coeff_2_slider)

    seq_id_slider.on_changed(update)
    coeff_1_slider.on_changed(update)
    coeff_2_slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()