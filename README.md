# SurvivalMultiTree-pyNSGP

This repository contains the code for Multi-Objective Genetic Programming with Multi-trees as individuals for Survival Regression.

# Setup

This code runs on Ubuntu and it uses Python as coding language.
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

This version enables you to address a Survival Regression problem by either using coxnet or NSGP with Multi-trees as individuals.
The python file that runs the experiment is the 'main.py', which in turns calls one of the functions contained in 'methods.py' (one function for each method).
The 'main.py' can be executed by using 'run\_single\_main.sh' and providing the required parameters: method, seed, dataset name, test size in percentage w.r.t. the provided dataset, path to the configuration file with method-specific hyper-parameters, a run id that identifies the run (employed to log the eventual successful completion of the run in the results folder), an integer (1 or 0) indicating whether or not enabling verbose output.
Moreover, parallel runs can be executed by using 'run\_parallel\_main.sh' (it requires parallel GNU to be installed on the system: 'sudo apt install parallel').
The parallel .sh script requires, as parameters, the path to a .txt file containing, for each line, the comma-separated sequence of parameters for a given run.
Therefore, if this .txt file contains N lines, the parallel script will execute N runs by parallelizing on the provided number of cores (the second parameter).

Results are saved in the results folder, and a .txt file named 'completed\_run<run\_id>.txt' is created within this folder containing the sequences of parameters of the successful runs sharing the same run id. Runs that raised an exception are tracked in the run\_with\_exceptions folder.

