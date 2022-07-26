# How to view/ Run these Notebooks

These tutorials are written as [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/)

## Read-only

If you want to take a quick look PDFs or markdowns have been generated for each notebook requiring no software requirements

1. [ConvolutionalNeuralNetworks](https://github.com/cemac/LIFD_ConvolutionalNeuralNetworks/blob/main/CNN_Volcanic_deformation.md)
2. [Physics_Informed_Neural_Networks](https://github.com/cemac/LIFD_Physics_Informed_Neural_Networks/blob/main/PINNs_1DHeatEquationExample.pdf)
3. [GaussianProcesses](https://github.com/cemac/LIFD_GaussianProcesses/blob/main/Gaussian_Processes.md)
4. [RandomForests](https://github.com/cemac/LIFD_RandomForests/blob/main/README.md)

## Running on your laptop or server you have access to.

To run on your machine first clone this repo via your preferred method e.g. [cloning via terminal](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) or [cloning via gitbash (windows)](https://www.gitkraken.com/blog/what-is-git-bash)

**Some notebooks require additional code from other git repos**

**In your terminal (Mac or Linux) or your gitbash terminal run**

```bash
git clone --recursive git@github.com:cemac/LIFD_ConvolutionalNeuralNetworks.git
```
 or if you have already cloned but forgotten to get the submodules:
```bash
cd LIFD_ENV_ML_NOTEBOOKS
git submodule update --init --recursive
```
## Cloning individual tutorials

1. `git clone --recursive git@github.com:cemac/LIFD_ConvolutionalNeuralNetworks.git`
2. `git clone --recursive git@github.com:cemac/LIFD_RandomForests.git`
3. `git clone --recursive git@github.com:cemac/LIFD_GaussianProcesses.git`
4. `git clone --recursive git@github.com:cemac/LIFD_Physics_Informed_Neural_Networks.git`


## How to Run

These notebooks can run with the resources provided and the anaconda environment setup. If you are familiar with anaconda, jupyter notebooks and GitHub. Simply clone this repository and run it within your Jupyter Notebook setup.

## Requirements

**Python**

It is recommended you use [anaconda](https://medium.com/pankajmathur/what-is-anaconda-and-why-should-i-bother-about-it-4744915bf3e6) to manage the python packages required. Sore Machine learning libraries are large and if you only wish to run one notebook consider installing the environment provided for that specific notebook. Otherwise, you can install all required packages running the following commands.  

```bash
conda env create -f <env-file>.yml
conda activate <env-name>
# save yourself some space with one extra command
conda clean -a
jupyter-notebook # launches the notebook server
```

further information can be found in the [jupyter_notebooks](jupyter_notebooks.md) guide. For set up and Troubleshooting tips.

## Binder

Notebooks that are [binder](https://mybinder.readthedocs.io/en/latest/index.html#what-is-binder) compatible have a binder launch button in their readme.  
