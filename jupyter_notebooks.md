# Jupyter Notebooks

These tutorials are written as interactive [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) if you're new to notebooks we've included some information on
how to get these running on your computer.

After you have [installed the necessary python packages](#Installation)

All instructions are for running in your terminal (mac or linux) or you

launch the notebook server

```bash
jupyter notebook --generate-config
jupyter-notebook
```

The notebook should launch in your browsers if not go to the address given in the terminal e.g.

[http://localhost:8888](http://localhost:8888)

## Tips for running notebooks

* A quick overview of jupyter notebooks can be found [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
* If you want a clean copy and one to "play with" select `file`--> `make a copy` and rename your copy as a working copy.

## Troubleshooting

* most things will be fixed by a fresh install
* stange widget errors can be fixed with `jupyter nbextension enable --py --sys-prefix widgetsnbextension`


If you wish to run these notebooks on a remote machine with accelerated hardware then please follow [these instructions](https://github.com/cemac/cemac_generic/wiki/Jupyter-Notebooks-Via-SSH-Tunnelling)  


# Installation [Anaconda](https://medium.com/pankajmathur/what-is-anaconda-and-why-should-i-bother-about-it-4744915bf3e6) (recomended instaltation method)

1. Install anaconda or miniconda on your computer  via the appropriate installer found [here](https://conda.io/en/latest/miniconda.html)

2. Machine Learning libraries are large so if you're only interested in one notebook then go to the folder required and install the tutorial specific yml file using [conda](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) or [mamba](#mamba)

### All Notebooks

```bash
conda env create -f allnotebooks.yml
conda activate LIFD
```

### Random Forests

```bash
cd RandomForests
conda env create -f RF.yml
conda activate RF
```
### Convolutional Neural Networks

```bash
cd ConvolutionalNeuralNetworks
conda env create -f CNN.yml
conda activate CNN
```

### Gaussian Processes

```bash
cd GaussianProcesses
conda env create -f GP.yml
conda activate
```

### Physics Informed Neural Networks

```bash
cd Physics_Informed_Neural_Networks
conda env create -f PINN.yml
conda activate PINNs
```

*this is the non version specific environment - the most likely to install easily but in future may need some tweaking may be required the full list of versions of every package in the python environment are given in `absolute_enviroments` folder these can be used instead to create a more rigid environment or to track down version differences*

# mamba

[Mamba](https://mamba.readthedocs.io/en/latest/) is a faster version of conda installed by running `conda install -c conda-forge mamba` and then used in place of conda in commands e.g. `conda install` becomes `mamba install` `conda env create` becomes `mamba env create` and so on.

This may be useful if the conda environment is taking a long time to solve.
