# Random Forests

[![LIFD_ENV_ML_NOTEBOOKS](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/actions/workflows/python-package-conda-RF.yml/badge.svg)](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/actions/workflows/python-package-conda-RF.yml) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cemac/LIFD_RandomForests/HEAD?labpath=RandomForests.ipynb)

This notebook explores Random Forests to find out what variables control leaf temperature

## Recommended Background Reading

If you are unfamiliar with some of the concepts covered in this tutorial it's recommended to read through the background reading below either as you go through the notebook or beforehand.

* [Decision Tree Introductory Video](https://www.youtube.com/embed/kakLu2is3ds)
* [Random Forests Introductory Video](https://www.youtube.com/embed/v6VJ2RO66Ag)
* [Random Forest overview linked with python](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)

## Quick look

If you want a quick look at the contents inside the notebook before deciding to run it please view the [md file](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/RandomForests/RandomForests.md) generated (*note some HTML code not fully rendered*)

## Installation and Requirements

This notebook is designed to run on a laptop with no special hardware required therefore recommended to do a local installation as outlined in the repository [howtorun](../howtorun.md) and [jupyter_notebooks](../jupyter_notebooks.md) sections.

### Quick start

If you're already familiar with git, anaconda and virtual environments the environment you need to create is found in GP.yml and the code below to install activate and launch the notebook

```bash
conda env create -f RF.yml
conda activate RF
jupyter-notebook
```
