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

def my_gp_to_pysym_with_coef(f_star, ode, tol=None, tol2=None, expand=False):
    VarDict = ode.get_var_dict()
    f_star_list, var_list, coef_list = parse_program_to_list(f_star.program)
    f_star_infix = generator.Generator.prefix_to_infix(f_star_list, variables=var_list, coefficients=coef_list)
    f_star_infix2 = f_star_infix.replace('{', '').replace('}', '')
    if f_star_infix2 == f_star_infix:
        f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix, VarDict, "simplify")
        return f_star_sympy

    f_star_sympy = generator.Generator.infix_to_sympy(f_star_infix2, VarDict, "simplify")

    if expand:
        f_star_sympy = sympy.expand(f_star_sympy)

    fs = str(f_star_sympy)
    out_with_coefs = fs

    fs = mask_X(fs)
    if tol is None:
        fs = re.sub(r'([0-9]*\.[0-9]+|[0-9]+)', 'C', fs)
    else:
        consts = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', fs)
        for const in consts:
            if const in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                continue
            if (float(const) < 1 + tol) and (float(const) > 1 - tol):
                fs = fs.replace(const, '1')
            elif (tol2 is not None) and (float(const) < tol2) and (float(const) > -1 * tol2):
                fs = fs.replace(const, '0')
            else:
                fs = fs.replace(const, f"{float(const):.5f}")

    fs = back_X(fs)
    f_star_sympy = generator.Generator.infix_to_sympy(fs, VarDict, "simplify")
    return out_with_coefs, str(f_star_sympy), f_star_sympy


def generate_estimated_trajectory(dg, f_hat, init_cond):
    f_list = []
    for i in range(3):
        if i == 0:
            f_list.append(f_hat.execute)
        else:
            f_list.append(ones_func)

    ode_hat = equations.InferredODE(3, f_hat_list = f_list, T = dg.T)
    print(ode_hat)
    dg_hat = data.DataGeneratorForOutput(
        ode_hat,
        dg.T,
        freq = dg.freq,
        n_sample = 1, # Not very clear as to what it corresponds to.
        noise_sigma = 0.0,
        init_cond = [[init_cond]])
    return dg_hat


def lambdify_ode(sympy_expr):
    symbols_as_tuple = ("X0", "X1", "X2", "cos", "sin")

    ret = sympy.lambdify(symbols_as_tuple, sympy_expr, "numpy")
    
    return ret


def system(t, state, source_terms, f_sympy):
    x1_value = source_terms[0](t)
    x2_value = source_terms[1](t)
    callable_ode = lambdify_ode(f_sympy)

    return [callable_ode(state, x1_value, x2_value, np.cos, np.sin)]


def get_values_and_analytical_form(dg, f_hat, patient_id):
    ode = equations.RealODEPlaceHolder()
    [_, f_analytic, f_sympy] = my_gp_to_pysym_with_coef(f_hat, ode, tol=0.05, expand=True)
    print(f_analytic)

    n_sample = dg.xt.shape[0]
    dt = 1 / (n_sample - 1)
    time_list = np.arange(0, 1 + dt, dt)

    interpolated_x1 = interpolate.interp1d(time_list, dg.xt[:, patient_id, 1], kind='cubic')
    interpolated_x2 = interpolate.interp1d(time_list, dg.xt[:, patient_id, 2], kind='cubic')

    x_pred = solve_ivp(system, t_span = [time_list[0], time_list[-1]], y0=[dg.xt[0, patient_id, 0]], t_eval=time_list, args=([interpolated_x1, interpolated_x2], f_sympy))

    return [x_pred, f_analytic]


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
    symbolic_model = input_params["symbolic_model"]
    x_id = 0

    if symbolic_model == "dcode":
        path_base = os.path.join(dir_path, "results_vi/{}/{}/sample-{}/dim-{}/".format(input_params["model_name"], input_params["data_filename"], n_train, x_id))
    else:
        path_base = os.path.join(dir_path, "results/{}/{}/sample-{}/dim-{}/".format(input_params["model_name"], input_params["data_filename"], n_train, dim_x))

    data_path = os.path.join(dir_path, "../NeuralODE/Data", input_params["model_name"], (input_params ["data_filename"] + ".pkl"))
    seed_start = 0
    seed_end = input_params["n_seed"]

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
    raw_fitness_list = [x["model"].fitness_ for x in res_list]

    plot_type = input_params["plot_type"]
    fig_title = input_params["figure_title"]
    Nplots = input_params["n_trajectories_plots"]

    n_sample, n_patients, n_dim = dg.xt.shape


    n_patients = 100
    dt = 1 / (n_sample - 1)
    time_list = np.arange(0, 1 + dt, dt)

    complete_simulated_value = np.zeros(((n_sample) * n_patients, 1))  
    complete_simulated_time = np.zeros(((n_sample) * n_patients, 1))  
    complete_original  = np.zeros((n_sample * n_patients, 1))  
    complete_original_time  = np.zeros((n_sample * n_patients, 1))  

    fig1, ax1 = plt.subplots(1, 1, sharex=False)
    ax1.plot(fitness_list, c = "b", label = "oob fitness")
    ax1.plot(raw_fitness_list, c = "r", label = "fitness")
    ax1.set_xlabel("SEED")
    ax1.set_ylabel("Fitness")
    fig1.suptitle("Fitness over every seed")
    set_unique_legend(ax1)
    fig1.savefig(dir_path + "/results/figures/" + input_params["data_filename"] + "_fitness.pdf")

    for i in range(n_patients):
        print(str(i+1) + " / " + str(n_patients) + " patients.")
        complete_original[i*n_sample:(i+1)*n_sample,0] = dg.xt[:,i,0]
        complete_original_time[i*n_sample:(i+1)*n_sample,0] = time_list + time_list[-1] * i
    
        if plot_type == "best_fit":
            best_fit_index = fitness_list.index(min(fitness_list))  
            [x_pred, f_analytic] = get_values_and_analytical_form(dg, f_hat_list[best_fit_index], i)

            complete_simulated_time[i*(n_sample):(i+1)*(n_sample), 0] = x_pred.t + x_pred.t[-1] * i
            complete_simulated_value[i*(n_sample):(i+1)*(n_sample),0] = x_pred.y[0]

    fig1, ax1 = plt.subplots(1, 1, sharex=False)
    ax1.plot(complete_original_time, complete_original, c = "b",label = "original")
    ax1.plot(complete_simulated_time, complete_simulated_value, c = "r", label = "reconstructed")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Trajectory")
    fig1.suptitle("SEED #" + str(best_fit_index) + "\n" + f_analytic + "\n Fitness: {:.6f}".format(fitness_list[best_fit_index]))    
    set_unique_legend(ax1)
    fig1.savefig(dir_path + "/results/figures/" + input_params["data_filename"] + ".pdf")

    output_path = pathlib.Path(dir_path + "/results/solved_ode/", input_params["data_filename"] + '_solved.pkl')

    if not output_path.parent.exists():
        os.mkdir(output_path.parent)

    with open(output_path, 'wb') as file:
        pickle.dump({"original": complete_original, "computed": complete_simulated_value, "time_array": complete_original_time, "discovered_ode": f_analytic}, file)

def error_display_every_seed(input_params):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dim_x = input_params["dim_x"]
    n_train = input_params["n_train"]
    symbolic_model = input_params["symbolic_model"]
    x_id = 0

    if symbolic_model == "dcode":
        path_base = os.path.join(dir_path, "results_vi/{}/{}/sample-{}/dim-{}/".format(input_params["model_name"], input_params["data_filename"], n_train, x_id))
    else:
        path_base = os.path.join(dir_path, "results/{}/{}/sample-{}/dim-{}/".format(input_params["model_name"], input_params["data_filename"], n_train, dim_x))

    data_path = os.path.join(dir_path, "../NeuralODE/Data", input_params["model_name"], (input_params ["data_filename"] + ".pkl"))
    seed_start = 0
    seed_end = input_params["n_seed"]

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
    raw_fitness_list = [x["model"].fitness_ for x in res_list]

    plot_type = input_params["plot_type"]
    fig_title = input_params["figure_title"]
    Nplots = input_params["n_trajectories_plots"]

    n_sample, n_patients, n_dim = dg.xt.shape

    dt = 1 / (n_sample - 1)
    time_list = np.arange(0, 1 + dt, dt)

    complete_simulated_value = np.zeros(((n_sample) * n_patients, 1))  
    complete_simulated_time = np.zeros(((n_sample) * n_patients, 1))  
    complete_original  = np.zeros((n_sample * n_patients, 1))  
    complete_original_time  = np.zeros((n_sample * n_patients, 1))  

    fig1, ax1 = plt.subplots(1, 1, sharex=False)
    ax1.plot(fitness_list, c = "b", label = "oob fitness")
    ax1.plot(raw_fitness_list, c = "r", label = "fitness")
    ax1.set_xlabel("SEED")
    ax1.set_ylabel("Fitness")
    fig1.suptitle("Fitness over every seed")
    set_unique_legend(ax1)
    fig1.savefig(dir_path + "/results/figures/" + input_params["data_filename"] + "_fitness.pdf")

    for seed in range(seed_start, seed_end):
        print("\n\n********************************************")
        print( "SEED #" + str(seed+1) + " out of " + str(seed_end))
        print("********************************************")
        for i in range(n_patients):
            print(str(i+1) + " / " + str(n_patients) + " patients.")
            complete_original[i*n_sample:(i+1)*n_sample,0] = dg.xt[:,i,0]
            complete_original_time[i*n_sample:(i+1)*n_sample,0] = time_list + time_list[-1] * i
        
            if plot_type == "best_fit":
                [x_pred, f_analytic] = get_values_and_analytical_form(dg, f_hat_list[seed], i)

                complete_simulated_time[i*(n_sample):(i+1)*(n_sample), 0] = x_pred.t + x_pred.t[-1] * i
                complete_simulated_value[i*(n_sample):(i+1)*(n_sample),0] = x_pred.y[0]

        fig1, ax1 = plt.subplots(1, 1, sharex=False)
        ax1.plot(complete_original_time, complete_original, c = "b",label = "original")
        ax1.plot(complete_simulated_time, complete_simulated_value, c = "r", label = "reconstructed")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Trajectory")
        fig1.suptitle("SEED #" + str(seed) + "\n" + f_analytic + "\n Fitness: {:.6f}".format(fitness_list[seed]))
        set_unique_legend(ax1)
        fig1.savefig(dir_path + "/results/figures/" + input_params["data_filename"] + "_seed_" + str(seed) +  ".pdf")

        output_path = pathlib.Path(dir_path + "/results/solved_ode/", input_params["data_filename"]  + "_seed_" + str(seed) + '_solved.pkl')

        if not output_path.parent.exists():
            os.mkdir(output_path.parent)

        with open(output_path, 'wb') as file:
            pickle.dump({"original": complete_original, "computed": complete_simulated_value, "time_array": complete_original_time, "discovered_ode": f_analytic}, file)


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
    # error_display_every_seed(input_params)
    plt.show()