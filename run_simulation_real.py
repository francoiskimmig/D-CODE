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

def run(dim_x, x_id, n_sample, alg, seed, n_seed, data_path, ode_name):
    np.random.seed(999)

    dg = data.DataGeneratorFromFile(dim_x, n_sample, data_path)

    yt = dg.generate_data()
    # print(dg.xt.shape)
    # print(dg.solver.t.shape)

    dxdt_hat = num_diff(yt, dg, alg)
    print('Numerical differentiation: Done.')

    X_train = yt[:-1, :, :]
    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])

    y_train = dxdt_hat[:, :, x_id].flatten()
    assert X_train.shape[0] == y_train.shape[0]

    # TODO Add to input file algorithm choice
    if alg == 'tv':
        path_base = 'results/{}/sample-{}/dim-{}/'.format(ode_name, n_sample, dim_x)
    else:
        path_base = 'results_spline/{}/sample-{}/dim-{}/'.format(ode_name, n_sample, dim_x)

    if not os.path.isdir(path_base):
        os.makedirs(path_base)

    for s in range(seed, seed+n_seed):
        print(' ')
        print('Running with seed {}'.format(s))
        start = time.time()
        f_hat, est_gp = run_gp_real(X_train, y_train, x_id, s)
        print(f_hat)

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

        print(f_hat)

def main(input_args): 
    
    # read options
    if isinstance(input_args, dict):
        args_key = list(input_args.keys())
        for key in args_key:
            input_args["--" + key] = input_args.pop(key)

        args_str = json.dumps(input_args)
        input_args = args_str.replace(":",",").replace("'","").replace('"',"").replace(" ","")[1:-1].split(',')

    if isinstance(input_args, list) and (all(isinstance(s, str) for s in input_args)):
        pass    
        
    else:
        raise TypeError("""invalid argument type 
            -> the argument must be a list of string or a dictionary""")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data file path", type=str, required=True)
    parser.add_argument("--dim_x", help="number of dimensions", type=int, default=2)
    parser.add_argument("--x_id", help="ID of the equation to be learned", type=int, default=0)
    parser.add_argument("--n_sample", help="number of trajectories", type=int, default=100)
    parser.add_argument("--alg", help="name of the benchmark", type=str, default='tv', choices=['tv', 'spline'])
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--n_seed", help="random seed", type=int, default=10)
    parser.add_argument("--ode_name", help="ode name for output folder", type=str, default="real")

    args = parser.parse_args(input_args)

    print('Running with: ', args)

    run(args.dim_x, args.x_id, args.n_sample, args.alg, seed=args.seed, n_seed=args.n_seed, data_path=args.data_path, ode_name=args.ode_name)

if __name__ == '__main__':
    main(sys.argv[1:]) 
