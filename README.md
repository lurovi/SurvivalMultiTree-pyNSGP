# SurvivalMultiTree-pyNSGP

This repository contains the code for Multi-Objective Genetic Programming with Multi-trees as individuals for Survival Regression.

# Setup

This code runs on Ubuntu and it uses Python as coding language (specifically Python 3.10.15 version).
Open the terminal and type the following commands:

```bash
git clone -b v1.6.2 https://github.com/giorgia-nadizar/genepro.git
git clone https://github.com/lurovi/SurvivalMultiTree-pyNSGP.git
cd SurvivalMultiTree-pyNSGP
conda env create -f environment.yml
conda activate genesurv-mt
cd ..
cd genepro
pip3 install -U .
cd ..
cd SurvivalMultiTree-pyNSGP
pip3 install -U .
```

# Run

Place yourself with the terminal within the folder 'SurvivalMultiTree-pyNSGP'.
This version enables you to address a Survival Regression problem by either using coxnet or NSGP with Multi-trees as individuals.
The python file that runs the experiment is the 'main.py', which in turns calls one of the functions contained in 'methods.py' (one function for each method).
The 'main.py' can be executed by using './scripts/run\_main.sh' and providing the required parameters: method, seed, dataset name, test size in percentage w.r.t. the provided dataset, path to the configuration file with method-specific hyper-parameters, a run id that identifies the run (employed to log the eventual successful completion of the run in the results folder).
Moreover, parallel runs can be executed by using './scripts/parallelize\_main.sh' (it requires parallel GNU to be installed on the system: 'sudo apt install parallel').
The parallel .sh script requires, as parameters, the path to a .txt file containing, for each line, the ;-separated sequence of parameters for a given run (parameter\_name1;parameter\_value1;parameter\_name2;parameter\_value2;parameter\_name3;parameter\_value3;...).
Therefore, if this .txt file contains N lines, the parallel script will execute N runs by parallelizing on the provided number of cores (the second parameter).

Results are saved in the results folder, and a .txt file named 'completed\_<run\_id>.txt' is created within this folder containing the sequences of parameters of the successful runs sharing the same run id. Runs that raised an exception are tracked in the run\_with\_exceptions folder.

Data and parameters can be found in [SurvivalMultiTree-pyNSGP-DATA](https://github.com/lurovi/SurvivalMultiTree-pyNSGP-DATA).
You can clone this repository and the data repository within the same folder and have a structure like this for better access to configuration files, etc.:

```bash
.
└── projects_folder
    ├── SurvivalMultiTree-pyNSGP
    └── SurvivalMultiTree-pyNSGP-DATA
```
Remember to place yourself with the terminal within the folder 'SurvivalMultiTree-pyNSGP'.

The 'datasets' folder contains examples of datasets in .csv format. The 'params' folder contains examples of parameters files for run execution. In particular, the .json files contains the possible parameters for a set of runs. By executing the 'generate\_parallelize\_input\_files.py' script by providing the path to one of these .json files, a .txt file with the same name and in the same folder will be created containing the lines detailing all the possible combinations of parameters described in the corresponding .json file. This .txt file can be fed as input the the 'scripts/parallelize_main.sh' script to parallelize the 'main.py'.

# Example of execution

Execute the 'main.py' with command-line arguments for a single run that stores the results in the 'results' folder and exceptions in the 'run\_with\_exceptions' folder:

```bash
python3 main.py --method nsgp --seed 42 --dataset support2 --normalize 1 --test_size 0.3 --config ../SurvivalMultiTree-pyNSGP-DATA/config_nsgp.yaml 
```

By additionally using '--verbose' as option you can print progress of the execution in the terminal.
By additionally using '--profile' as option you can print at the end a table detailing information about speed execution of all the primitives and functions employed.

You can run the main (without using verbose or profile) also by doing (must specify run_id, this script is designed to be executed in parallel):

```bash
./scripts/run_main.sh method nsgp seed 42 dataset support2 normalize 1 test_size 0.3 config ../SurvivalMultiTree-pyNSGP-DATA/config_nsgp.yaml run_id test
```

In order to parallelize multiple runs, you must firstly use the 'generate\_parallelize\_input\_files.py' script to generate the .txt file containint the parallel runs to execute.
Suppose you have a parameters .json file detailing the combinations of parameters to be executed as runs:

```json
{
	"method": ["nsgp"],
	"seed": [1, 2, 3],
	"dataset": ["pbc2", "support2"],
	"normalize": [0, 1],
	"test_size": [0.3],
	"config": ["../SurvivalMultiTree-pyNSGP-DATA/config_nsgp.yaml"]
}
```

By executing:

```bash
python3 generate_parallelize_input_files.py --json_path ../SurvivalMultiTree-pyNSGP-DATA/params/example_nsgp.json 
```

A 'exaple_nsgp.txt' file will be created within the '../SurvivalMultiTree-pyNSGP-DATA/params/' folder. Each line of this file represents a specific run with a given set of parameters. This .txt file contains the runs derived by the cartesion product of all the parameters in the provided .json file.

By doing:

```bash
./scripts/parallelize_main.sh ../SurvivalMultiTree-pyNSGP-DATA/params/example_nsgp.txt 6 
```
The runs in the .txt file will be executed in parallel by using 6 cores from your computer, one core for each run.
The lines corresponding to completed and successful runs are logged within a .txt file in the results folder by using as run_id the name of the parallelizable .txt file.

# Sklearn estimator

This library offers a sklearn-based estimator that can be directly used within Python code. An example of Python script is shown in the 'example.py'.

