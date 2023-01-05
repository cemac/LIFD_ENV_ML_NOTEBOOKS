<div align="center">
<img src="https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/images/LIFDlogo.png"></a>
<a href="https://www.cemac.leeds.ac.uk/">
  <img src="https://github.com/cemac/cemac_generic/blob/master/Images/cemac.png"></a>
  <br>
</div>

# Leeds Institute for Fluid Dynamics Machine Learning For Earth Sciences #
## Jupyter Notebooks ##

 [![GitHub release](https://img.shields.io/github/release/cemac/LIFD_ENV_ML_NOTEBOOKS.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/releases) [![GitHub top language](https://img.shields.io/github/languages/top/cemac/LIFD_ENV_ML_NOTEBOOKS.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS) [![GitHub issues](https://img.shields.io/github/issues/cemac/LIFD_ENV_ML_NOTEBOOKS.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/issues) [![GitHub last commit](https://img.shields.io/github/last-commit/cemac/LIFD_ENV_ML_NOTEBOOKS.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/commits/master) [![GitHub All Releases](https://img.shields.io/github/downloads/cemac/LIFD_ENV_ML_NOTEBOOKS/total.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/releases) ![GitHub](https://img.shields.io/github/license/cemac/LIFD_ENV_ML_NOTEBOOKS.svg)[![DOI](https://zenodo.org/badge/366734586.svg)](https://zenodo.org/badge/latestdoi/366734586)



[![Twitter Follow](https://img.shields.io/twitter/follow/FluidsLeeds.svg?style=social&label=Follow)](https://twitter.com/FluidsLeeds)

Leeds Institute for Fluid Dynamics (LIFD) has teamed up with the Center for Environmental Modelling and Computation (CEMAC) team to create 4 Jupyter notebook tutorials on the following topics.

1. [ConvolutionalNeuralNetworks](https://github.com/cemac/LIFD_ConvolutionalNeuralNetworks)
2. [Physics_Informed_Neural_Networks](https://github.com/cemac/LIFD_Physics_Informed_Neural_Networks)
3. [GaussianProcesses](https://github.com/cemac/LIFD_GaussianProcesses)
4. [RandomForests](https://github.com/cemac/LIFD_RandomForests)
5. [GenerativeAdversarialNetworks](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks)

**PLEASE NOTE YOU MUST CLONE RECURSIVELY (SEE BELOW)**

These notebooks require very little previous knowledge on a topic and will include links to further reading where necessary. Each Notebook should take about 2 hours to run through and should run out of the box home installations of Jupyter notebooks. These notebooks are designed with automatic checking of python environment files to remain easy to set up into the future.

As this resource grows in order to not make the repository unwieldy this repository is made up of submodules this means you can clone

## How do I get started ??

Some tutorials are so lightweight you can run them on [binder](https://mybinder.readthedocs.io/en/latest/#what-is-binder) the others we recommend running on your local machine to get started either clone this repository (**LARGE SIZE**) or select a tutorial to clone and run each tutorial separately.

### Binder enabled tutorials

1. [GaussianProcesses](https://github.com/cemac/LIFD_GaussianProcesses)
2. [RandomForests](https://github.com/cemac/LIFD_RandomForests)
3. [GenerativeAdversarialNetworks](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks

### Cloning the whole repository

``bash
git clone --recursive git@github.com:cemac/LIFD_ENV_ML_NOTEBOOKS.git
``

then follow the Individual README.md instructions

### Cloning individual tutorials

1. `git clone --recursive git@github.com:cemac/LIFD_ConvolutionalNeuralNetworks.git`
2. `git clone --recursive git@github.com:cemac/LIFD_RandomForests.git`
3. `git clone --recursive git@github.com:cemac/LIFD_GaussianProcesses.git`
4. `git clone --recursive git@github.com:cemac/LIFD_Physics_Informed_Neural_Networks.git`
5. `git clone --recursive git@github.com:cemac/LIFD_GenerativeAdversarialNetworks.git`


## How to Run

These notebooks can run with the resources provided and the anaconda environment setup. If you are familiar with anaconda, jupyter notebooks and GitHub. Simply clone this repository and run it within your Jupyter Notebook setup. Otherwise please read the [how to run](howtorun.md) guide. Induvidual notebooks have bespoke instructions


```bash
git clone --recursive git@github.com:cemac/LIFD_ENV_ML_NOTEBOOKS.git
cd LIFD_ENV_ML_NOTEBOOKS
```

## Requirements

**Python**

It is recommended you use [anaconda](https://medium.com/pankajmathur/what-is-anaconda-and-why-should-i-bother-about-it-4744915bf3e6) to manage the python packages required. Sore Machine learning libraries are large and if you only wish to run one notebook consider installing the environment provided for that specific notebook. Otherwise, you can install all required packages running the following commands.  

```bash
conda env create -f <env-file>.yml
conda activate <env-name>
# save yourself some space with one extra command
conda clean -a
```

**What if I forgot to clone recursively?**

Not to worry in your cloned folder simply run:

```bash
git submodule init
git submodule update --init --recursive
```

**Hardware**

These notebooks are designed to run on a personal computer. Although please note the techniques demonstrated can be very computationally intensive so there may be options to skip steps depending on the hardware available. e.g. use pre-trained models.

**Knowledge**

No background knowledge is required of the environmental Science concepts or machine learning concepts. We have assumed some foundational knowledge but links are provided to in-depth information on the fundamentals of each concept  

## Contributions

We hope that this resource can be built upon to provide a wealth of training material for Earth Science Machine Learning topics at Leeds

# Licence information #

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">LIFD_ENV_ML_NOTEBOOKS</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://cemac.leeds.ac.uk/" property="cc:attributionName" rel="cc:attributionURL">cemac</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Acknowledgements

*Leeds Institute of Fluid Dynamics*, *CEMAC*, *Matthew Gaddes*, *Oliver Pollard*, *Chetan Deva*, *Fergus Shone*, *Michael MacRaild*, *Phil Livermore*, *Giulia Fedrizzi*, *Eszter Kovacs*, *Ana Reyna Flores*, *Francesca Morris*, *Emma Pearce*, *Maeve Murphy Quinlan*, *Sara Osman*, *Jonathan Coney*, *Eilish O'grady*, *Leif Denby*, *Sandra Piazolo*, *Caitlin Howath*, *Claire Bartholomew*, *Anna Hogg* and *Ali Gooya*
