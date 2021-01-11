# Deep active inference model of the rubber-hand illusion

This repository contains the full code used in "A deep active inference model of the rubber-hand illusion"<sup>1</sup>. This repository consists of two parts: The environment modelled in Unity and a Python repository containing the agent code. These components interact by means of the [ML-Agents Toolkit for Unity](https://github.com/Unity-Technologies/ml-agents).


## Setup
To use the environment and Python code, clone this repository or download its contents directly from the [latest release](https://github.com/thomasroodnl/active-inference-rhi/releases/tag/v1.0). 
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
To easily run and edit the scripts in the Python repository, open the **active-inference-agent** folder as a project in an IDE of choice. This is important as the current state of relative imports only functions correctly with respect to the **active-inference-agent** folder as source folder. In addition, make sure to set the project interpreter to the `python.exe` of the **active-inference-env** environment. If you have trouble finding the location of the environment folder containing this file, you can use the following command to show the file system paths for all environments:
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
2. Open the Unity project in the Unity editor, run the Python code with `editor_mode = True` and click the play button in the Unity editor when prompted in the Python console. If you are using Unity Hub, you can load the project by clicking the **ADD** button in the 'Projects' overview and selecting the **Unity Environment** folder. *Note: If the environment is empty when opening the project for the first time, navigate to the 'Scenes' folder in the file navigator in the bottom left of the UI. Double-clicking the `DefaultScene.unity` file will open the agent environment.*
> Note: the build included is a universal Windows platform build (x86 for backward compatability). For Mac/Linux based systems, it should still be possible to access the environment through the Unity editor. Please contact us if you run into any issues such that we can update the instructions and/or include a build for every platform.  

## Citing
If you use this software in your research, please cite [our work](https://doi.org/10.1007/978-3-030-64919-7_10) using the following BibTex entry:
```
@InProceedings{rood2020deep,
author="Rood, Thomas
and van Gerven, Marcel
and Lanillos, Pablo",
editor="Verbelen, Tim
and Lanillos, Pablo
and Buckley, Christopher L.
and De Boom, Cedric",
title="A Deep Active Inference Model of the Rubber-Hand Illusion",
booktitle="Active Inference",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="84--91",
abstract="Understanding how perception and action deal with sensorimotor conflicts, such as the rubber-hand illusion (RHI), is essential to understand how the body adapts to uncertain situations. Recent results in humans have shown that the RHI not only produces a change in the perceived arm location, but also causes involuntary forces. Here, we describe a deep active inference agent in a virtual environment, which we subjected to the RHI, that is able to account for these results. We show that our model, which deals with visual high-dimensional inputs, produces similar perceptual and force patterns to those found in humans.",
isbn="978-3-030-64919-7"
}
```


<sup>1</sup> Thomas Rood, Marcel van Gerven, and Pablo Lanillos. [A deep active inference model of the rubber-hand illusion](https://doi.org/10.1007/978-3-030-64919-7_10). In Tim Verbelen, Pablo Lanillos,  Christopher L. Buckley, and Cedric De Boom, editors, *Active Inference*, pages 84–91, Cham, 2020. Springer International Publishing.
