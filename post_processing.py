import argparse
import pickle
import os
import time
import matplotlib.pyplot as plt
import numpy as np

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


def ones_func(x):
    return 1.0

def error_display(output_path, model):
    path_base = "/Users/francois/Codes/VanderschaarLab/D-CODE/results/real/sample-80/dim-1/"

    dim_x = 1
    n_sample = 80
    data_path = "/Users/francois/Codes/NeuralODE/Data/first_order_NL_power/data_default.pkl"
    seed_s = 0
    seed_e = 2
    x_id = 0

    dg = data.DataGeneratorFromFile(dim_x, n_sample, data_path)

    path = "/Users/francois/Codes/VanderschaarLab/D-CODE/results/real/sample-80/dim-1/grad_seed_4.pkl"

    res_list = []
    for s in range(seed_s, seed_e):
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
    best_fit = fitness_list.index(min(fitness_list))
    print(best_fit)

    ind = 0

    f_list = []
    for i in range(2):
        if i == 0:
            f_list.append(f_hat_list[ind].execute)
        else:
            f_list.append(ones_func)

    ode_hat = equations.InferredODE(2, f_hat_list=f_list, T=dg.T)
    dg_hat = data.DataGenerator(
        ode_hat,
        dg.T,
        freq=dg.freq,
        n_sample=10,
        noise_sigma=0.0,
        init_low=(0.99, 0.01),
        init_high=(1.0, 0.0),
    )

    time = np.arange(0, dg.T, 1/(dg.freq+1))

    plt.plot(time, dg_hat.xt[:, 0, 0])
    plt.ylim(0, 1.1)

    x_true = dg.yt_test[:, :, 0]
    x_pred = dg_hat.xt[:, 0:1, 0]

    for i in range(x_true.shape[1]):
        plt.scatter(time, x_true[:,i]) # TODO: all in the same color       

    # plt.scatter(time, dg.yt_test[:, :, 1], x_true)
    plt.plot(time, x_pred)

    plt.show()
# mask = dg.mask_test

# np.sqrt(np.sum((x_true - x_pred) ** 2 * mask) / np.sum(mask))
# std_RMSE((x_true - x_pred) ** 2)

if __name__ == '__main__':
    error_display('ret', 'wetwe')









# 