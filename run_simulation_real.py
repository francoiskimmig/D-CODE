import argparse
import numpy as np

import data
from gp_utils import run_gp_real
from interpolate import num_diff
import pickle
import os
import time
import sys
import json
import matplotlib.pyplot as plt


def run(dim_x, x_id, n_sample, alg, seed, n_seed, data_path, ode_name, data_filename, config):
    np.random.seed(999)
    dg = data.DataGeneratorFromFile(dim_x, n_sample, data_path)

    yt = dg.generate_data()

    dxdt_hat = num_diff(yt, dg, alg)
    print('Numerical differentiation: Done.')
    # print("dxdt_hat: ", dxdt_hat.shape)

    X_train = yt[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    # print("X_train: ", X_train.shape)
    # print("y_train: ", y_train.shape)
    # fig1, ax1 = plt.subplots(1, 1)
    # ax1.plot(dg.solver.t[:-1], dxdt_hat[:, 18, x_id], label="smoothed dxdt")
    # ax1.plot(dg.solver.t[:-1], yt[:-1, 18, x_id], label="raw")
    # fig1.legend()
    # plt.show()
    assert X_train.shape[0] == y_train.shape[0]

    # TODO Add to input file algorithm choice
    if alg == 'tv':
        path_base = 'results/{}/{}/sample-{}/dim-{}/'.format(ode_name, data_filename, n_sample, dim_x)
    else:
        path_base = 'results_spline/{}/sample-{}/dim-{}/'.format(ode_name, n_sample, dim_x)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    for s in range(seed, seed+n_seed):
        print(' ')
        print('Running with seed {}'.format(s))
        start = time.time()

        f_hat, est_gp = run_gp_real(X_train, y_train, config, s)

        if x_id == 0:
            path = path_base + 'grad_seed_{}.pkl'.format(s)
        else:
            path = path_base + 'grad_x_{}_seed_{}.pkl'.format(x_id, s)
        end = time.time()

        with open(path, 'wb') as f:
            pickle.dump({
                'model': est_gp._program,
                'X_train': X_train,
                'y_train': y_train,
                'seed': s,
                'f_hat': f_hat,
                'dg': dg,
                'time': end-start,
            }, f)

def main(input_args): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data file path", type=str, required=True)
    parser.add_argument("--dim_x", help="number of dimensions", type=int, default=2)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='tv', choices=['tv', 'spline'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)
    parser.add_argument("--ode_name", help="ode name for output folder", type=str, default="real")
    parser.add_argument("--config", help="config for the symbolic regression", type=dict, required=True)
    
    # read options
    if isinstance(input_args, dict):
        run(dim_x = input_args["dim_x"], x_id=0,  n_sample = input_args["n_sample"],  alg="tv" ,seed=0, n_seed = input_args["n_seed"] , data_path = input_args["data_path"], ode_name= input_args["ode_name"],data_filename=input_args["data_filename"], config= input_args["config"])
        # args_key = list(input_args.keys())
        # for key in args_key:
        #     input_args["--" + key] = input_args.pop(key)

        # args_str = json.dumps(input_args)
        # input_args = args_str.replace(":",",").replace("'","").replace('"',"").replace(" ","")[1:-1].split(',')

    # if isinstance(input_args, list) and (all(isinstance(s, str) for s in input_args)):
    #     args = parser.parse_args(input_args)
    #     print('Running with: ', args)

    #     run(args.dim_x, args.x_id, args.n_sample, args.alg, seed=args.seed, n_seed=args.n_seed, data_path=args.data_path, ode_name=args.ode_name, config=args.config)
        
    # else:
    #     raise TypeError("""invalid argument type 
    #         -> the argument must be a list of string or a dictionary""")


    # print(input_args)


if __name__ == '__main__':
    main(sys.argv[1:]) 
