# Physics Informed Neural Networks

These set of notebooks explore Physics Informed Neural Networks to explore Partial Differential Equations

This tutorial has been split into 3 tutorials.

## Recommended Background Reading

If you are unfamiliar with some of the concept covered in this tutorial it's recommended to read through the background reading below either as you go through the notebook or beforehand. These links are also contained with in the notebooks

* [Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
* [Physics Guided Neural Networks](https://towardsdatascience.com/physics-guided-neural-networks-pgnns-8fe9dbad9414)
* [Physics-Informed Neural Networks:  A Deep LearningFramework for Solving Forward and Inverse ProblemsInvolving Nonlinear Partial Differential Equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

## Quick look

If you want a quick look at the contents inside the notebook before deciding to run it please view the md files genrated (*note some html code not fully rendered*)

* [PNNs_1DHeatEquationExample_nonML](PNNs_1DHeatEquation_nonML.md)
* [PNNs_1DHeatEquationExample](PNNs_1DHeatEquationExample.md)
* [PNNs_NavierStokesEquationExample_nonML](PNNs_NavierStokesEquationExample.md)


## Installation and Requirements

This notebook is designed to run on a laptop  with no special hardware required therefore recommended to do a local installation as outline in the repository [howtorun](../howtorun.md) and [jupyter_notebooks](../jupyter_notebooks.md) sections.


These notebooks require some additional data from the [PINNs](https://github.com/maziarraissi/PINNs) repository

If you have not already then in your gitbash or terminal please run the following code in the LIFD_ENV_ML_NOTEBOOKS directory via the terminal(mac or linux)  or git bash (windows)

```bash
git submodule init
git submodule update --init --recursive
```

**If this does not work please clone the [PINNs](https://github.com/maziarraissi/PINNs) repository into your Physics_Informed_Neural_Networks folder on your computer**

### Quick start

If you're already familiar with git, anaconda and virtual environments the environment you need to create is found in CNN.yml and the code below to install activate and launch the notebook

```bash
conda env create -f PINN.yml
conda activate PINN
jupyter-notebook
```
