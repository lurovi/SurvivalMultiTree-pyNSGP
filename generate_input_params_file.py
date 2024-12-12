# Define the datasets
datasets = [
    "pbc2",
    "support2",
    "framingham",
    "breast_cancer_metabric",
    "breast_cancer_metabric_relapse",
]

# Define the methods
methods = ["coxnet", "nsgp"]

# Number of repetitions
repetitions = 50

# Run Id
run_id = 1

# Verbose
verbose = 1

# Test size
test_size = 0.3

# File generation
def generate_block():
    lines = []
    for method in methods:
        for rep in range(1, repetitions + 1):
            for dataset in datasets:
                for normalize in [1, 0]:
                    line = f"{method},{rep},{dataset},{normalize},{test_size},config_{method}.yaml,{run_id},{verbose}"
                    lines.append(line)
    return lines

# Write to a file
output_file = "generated_block.txt"
with open(output_file, "w") as file:
    file.write("\n".join(generate_block()))

print(f"File '{output_file}' has been generated successfully.")

