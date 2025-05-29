import itertools
import os
import argparse
import json
import math


def generate_param_combinations(params_dict):
    """
    Generate all combinations of parameters based on provided dictionary values.
    """
    keys = params_dict.keys()
    values = (params_dict[key] if isinstance(params_dict[key], list) else [params_dict[key]] for key in keys)
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def create_input_files(base_output_file, param_combinations, num_files):
    """
    Create .txt input files for parallelize_main.sh, distributing parameter combinations across multiple files.
    """
    output_dir = os.path.dirname(base_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(base_output_file))[0]  # Extract base filename without extension

    sep = ';'

    # Split param_combinations into num_files chunks
    num_files = min(num_files, len(param_combinations))  # Prevent empty files
    batch_size = math.ceil(len(param_combinations) / num_files)  # Ensure all items are distributed
    batches = [param_combinations[i * batch_size:(i + 1) * batch_size] for i in range(num_files)]

    for i, batch in enumerate(batches, start=1):
        if num_files == 1:
            output_file = os.path.join(output_dir, f"{base_filename}.txt")  # No index if only one file
            file_label = base_filename
        else:
            output_file = os.path.join(output_dir, f"{base_filename}_{i}.txt")  # Append index if multiple files
            file_label = f"{base_filename}_{i}"  # Label to append at the end of each line

        with open(output_file, "w") as f:
            for params in batch:
                params['run_id'] = file_label
            for params in batch:
                f.write(sep.join(str(key) + sep + str(params[key]) for key in sorted(list(params.keys()))) + '\n')

        print(f"Generated input file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate .txt files for parallelize_main.sh")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing parameters.")
    parser.add_argument("--num_files", type=int, default=1, help="Number of output files to generate (must be >=1).")
    args = parser.parse_args()

    if args.num_files < 1:
        raise ValueError(f"--num_files must be at least 1, found {args.num_files}.")

    # Derive base output filename from json_path
    json_dir, json_filename = os.path.split(args.json_path)
    base_output_file = os.path.join(json_dir, os.path.splitext(json_filename)[0] + ".txt")

    # Load parameters from JSON file
    with open(args.json_path, "r") as f:
        params = json.load(f)

    param_combinations = generate_param_combinations(params)
    create_input_files(base_output_file, param_combinations, args.num_files)


if __name__ == "__main__":
    main()

