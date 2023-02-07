import argparse
import pickle
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json

import data
import equations

from gp_utils import *
from interpolate import get_ode_data
from collections import Counter

# def gp_to_pysym(f_star, ode):
#     VarDict = ode.get_var_dict()
#     f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
#     f_star_infix = generator.Generator.prefix_to_infix(
#         f_star_list, variables=var_list, coefficients=coef_list
#     )
#     f_star_infix2 = f_star_infix.replace("{", "").replace("}", "")
#     if f_star_infix2 == f_star_infix:
#         f_star_sympy = generator.Generator.infix_to_sympy(
#             f_star_infix, VarDict, "simplify"
#         )
#         return f_star_sympy

#     f_star_sympy = generator.Generator.infix_to_sympy(
#         f_star_infix2, VarDict, "simplify"
#     )
#     return f_star_sympy


# def std_RMSE(err_sq):
#     rmse_list = []
#     for i in range(500):
#         new_err = err_sq[np.random.randint(0, len(err_sq), err_sq.shape)]
#         rmse_itr = np.sqrt(np.mean(new_err))
#         rmse_list.append(rmse_itr)
#     return np.std(np.array(rmse_list)


def generate_estimated_trajectory(dg, f_hat):
    f_list = []
    for i in range(2):
        if i == 0:
            f_list.append(f_hat.execute)
        else:
            f_list.append(ones_func)

    ode_hat = equations.InferredODE(2, f_hat_list = f_list, T = dg.T)
    dg_hat = data.DataGenerator(
        ode_hat,
        dg.T,
        freq = dg.freq,
        n_sample = 10, # Not very clear as to what it corresponds to.
        noise_sigma = 0.0,
        init_low = (0.99, 0.01),
        init_high = (1.0, 0.0),
    )

    return dg_hat


def set_unique_legend(axe):
    handles, labels = axe.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axe.legend(by_label.values(), by_label.keys())  


def ones_func(x):
    return 1.0


def error_display(input_params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dim_x = input_params["dim_x"]
    n_train = input_params["n_train"]
    path_base = os.path.join(dir_path, "results/{}/sample-{}/dim-{}/".format(input_params["ode_name"], n_train, dim_x))

    data_path = os.path.join(dir_path, "../NeuralODE/Data", input_params["model_name"], (input_params ["data_filename"] + ".pkl"))
    seed_start = 0
    seed_end = input_params["n_seed"]
    x_id = 0

    dg = data.DataGeneratorFromFile(dim_x, n_train, data_path)

    res_list = []
    for s in range(seed_start, seed_end):
        if x_id == 0:
            path = path_base + "grad_seed_{}.pkl".format(s)
        else:
            path = path_base + "grad_x_{}_seed_{}.pkl".format(x_id, s)

        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
            res_list.append(res)
        except FileNotFoundError:
            pass

    f_sym_list = [x["f_hat"] for x in res_list]
    f_hat_list = [x["model"] for x in res_list]
    fitness_list = [x["model"].oob_fitness_ for x in res_list]

    testo = res_list[0]
    print(testo["model"].__dict__.keys())

    fig1, ax1 = plt.subplots(1, 1)
    x_true = dg.yt_test[:, :, 0]
    time = np.arange(0, dg.T, 1 / (dg.freq + 1))
    for i in range(x_true.shape[1]):
        ax1.scatter(time, x_true[:, i] , [4], c = "b", label = "data") # Note that [4] is the size of the markers.

    plot_type = input_params["plot_type"]
    if plot_type == "best_fit":
        best_fit_index = fitness_list.index(min(fitness_list))
        dg_hat = generate_estimated_trajectory(dg, f_hat = f_hat_list[best_fit_index])
        x_pred = dg_hat.xt[:, 0, 0]
        ax1.plot(time, x_pred, c = "r", label = "best fit estimated")
        worst_fit_index = fitness_list.index(max(fitness_list))
        dg_hat = generate_estimated_trajectory(dg, f_hat = f_hat_list[worst_fit_index])
        x_pred = dg_hat.xt[:, 0, 0]
        ax1.plot(time, x_pred, c = "g", label = "worst fit estimated")


    if plot_type == "every_seed":
        best_fit_index = fitness_list.index(min(fitness_list))
        for i in range(len(f_hat_list)):
            dg_hat = generate_estimated_trajectory(dg, f_hat = f_hat_list[i])
            x_pred = dg_hat.xt[:, 0, 0]
            ax1.plot(time, x_pred, label = "estimated seed #" + str(i))

    ax1.set_ylim(0, 1.1)
    fig1.suptitle("Estimated trajectory - " + plot_type)
    set_unique_legend(ax1)

# mask = dg.mask_test

# np.sqrt(np.sum((x_true - x_pred) ** 2 * mask) / np.sum(mask))
# std_RMSE((x_true - x_pred) ** 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "--json_file",
      "-i",
      "--input",
      help = "input parameter .json file",
      type = str,
      )
  
    args = parser.parse_args()
    
    json_file = args.json_file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    if not json_file:
        json_file = os.path.join(dir_path, "../NeuralODE/Data/input_files/default.json")

    if not json_file.endswith(".json"):
        json_file += ".json"
    
    with open(json_file, 'r') as file:
        input_params = json.load(file)
        
    error_display(input_params)
    plt.show()