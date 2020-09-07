# Deep active inference model of the rubber-hand illusion

This repository contains the full code used in "[A deep active inference model of the rubber-hand illusion](https://arxiv.org/abs/2008.07408)". This repository consists of two parts: The environment modelled in Unity and a Python repository containing the agent code. These components interact by means of the [ML-Agents Toolkit for Unity](https://github.com/Unity-Technologies/ml-agents).


## Setup
To use the environment and Python code, clone this repository or download its contents directly from the [latest release](#releases). 
```sh
git clone https://github.com/thomasroodnl/active-inference-rhi.git
```
### Installing the necessary packages
The fastest way to get the Python environment up and running is by creating a new conda environment (for the installation of conda see: [link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). The repository contains a file named 'conda-environment.yml', which can be used to automatically create a conda environment containing the right packages. To create an environment from this file:
1. Open a terminal/Anaconda prompt and navigate to the root of the repository
2. Execute the following command to download and install the necessary packages in a new environment:
```sh
conda env create -f conda-environment.yml
```
After all the packages have been downloaded and installed, the environment can be accessed under the name **active-inference-env**:
```sh
conda activate active-inference-env
```
To easily run and edit the scripts in the Python repository, open the **active-inference-agent** folder as a project in an IDE of choice and make sure to set the project interpreter to the `python.exe` of the **active-inference-env** environment. If you have trouble finding the location of the environment folder containing this file, you can use the following command to show the file system paths for all environments:
```sh
conda info --envs
```


## Running the agent
The Python project contains three main folders:
* **model_operation**: contains the code necessary to run the agent, including the agent itself
* **model_training**: contains the code concerned with model training, including data generation algorithms and the visual neural networks
* **post_processing**: contains the code used to plot the results for the paper


The model_operation folder contains a file `main_inference.py` which can be used to run the agent in the environment. There are two ways to run the agent:
1. Run the Python code with `editor_mode = False`, which will automatically launch a prebuild version of the environment (located at /Unity Environment/build/).
2. Open the Unity project in the Unity editor, run the Python code with `editor_mode = True` and click the play button in the Unity editor when prompted in the Python console.
> Note: the build included is a universal Windows platform build (x86 for backward compatability). For Mac/Linux based systems, it should still be possible to access the environment through the Unity editor. Please contact us if you run into any issues such that we can update the instructions and/or include a build for every platform.  

## Releases



