import run_simulation_real as run
import json

# # option 1 -----------------------------------------------------------------------------------------
# data_path = "/Users/francois/Codes/NeuralODE/Data/first_order/data_default.pkl"
# dim_x = 1
# n_train = 3
# n_seed = 5

# # option 2 -----------------------------------------------------------------------------------------
# data_path = "/Users/francois/Codes/NeuralODE/Data/first_order/data_noise_level_0_1.pkl"
# dim_x = 1
# n_train = 80
# n_seed = 5

# # option 3 -----------------------------------------------------------------------------------------
# data_path = "/Users/francois/Codes/NeuralODE/Data/first_order/data_noise_level_0_05_init_2.pkl"
# dim_x = 1
# n_train = 40
# n_seed = 5

# # option 4 -----------------------------------------------------------------------------------------
# data_path = "/Users/francois/Codes/NeuralODE/Data/first_order/data_noise_level_0_01_init_1_alpha_3.pkl"
# dim_x = 1
# n_train = 80
# n_seed = 5

# option 5 -----------------------------------------------------------------------------------------
data_path = "/Users/francois/Codes/NeuralODE/Data/first_order_NL/data_noise_level_0_param_m0_9_init_1.pkl"
dim_x = 1
n_train = 80
n_seed = 5

# script -------------------------------------------------------------------------------------------
args = {"data_path": data_path, "dim_x": dim_x, "n_sample": n_train, "n_seed": n_seed}

run.main(args)
