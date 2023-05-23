import os 
import run_simulation_real_vi as run
import json
import argparse

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
        json_file = os.path.join(dir_path, "input_files/real_data_no_time.json")

    if not json_file.endswith(".json"):
        json_file += ".json"
    
    with open(json_file, 'r') as file:
        input_params = json.load(file)

    config = input_params["config"]
    config["const_range"] = tuple(config["const_range"])
    config["init_depth"] = tuple(config["init_depth"])

    data_path = os.path.join(dir_path, "../NeuralODE/Data", input_params["model_name"], (input_params ["data_filename"] + ".pkl"))
    if input_params["target_dimension"]:
        x_id = input_params["target_dimension"] - 1
    else:
        x_id = 0

    args = {"data_path": data_path, "dim_x": input_params["dim_x"], "target_dimension": x_id, "n_sample": input_params["n_train"], "n_seed": input_params["n_seed"],
         "ode_name": input_params["model_name"], "data_filename": input_params["data_filename"], "config": config}
    run.main(args)