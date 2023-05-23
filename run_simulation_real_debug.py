import os 
import run_simulation_real as run
import json
import argparse

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

# # option 5 -----------------------------------------------------------------------------------------
# data_path = "/Users/francois/Codes/NeuralODE/Data/first_order_NL/data_noise_level_0_param_m0_9_init_1.pkl"
# dim_x = 1
# n_train = 80
# n_seed = 5

if __name__ == "__main__":
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

    config = input_params["config"]
    config["const_range"] = tuple(config["const_range"])
    config["init_depth"] = tuple(config["init_depth"])

    data_path = os.path.join(dir_path, "../NeuralODE/Data", input_params["model_name"], (input_params ["data_filename"] + ".pkl"))
    args = {"data_path": data_path, "dim_x": input_params["dim_x"], "n_sample": input_params["n_train"], "n_seed": input_params["n_seed"],
         "ode_name": input_params["model_name"], "data_filename": input_params["data_filename"], "config": config}
    run.main(args)