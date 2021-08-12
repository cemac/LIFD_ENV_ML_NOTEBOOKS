# Convolutional Neural Networks

This notebook explores Convolutional Neural Networks to dectect and categorise Volcanic deformation.

## Recommended Background Reading

If you are unfamiliar with some of the concept covered in this tutorial it's recommended to read through the background reading below either as you go through the notebook or beforehand.

* [The very basics in a Victor Zhou Blog](https://victorzhou.com/blog/intro-to-cnns-part-1/)
* [A deep dive into CNNs in towards data science](https://towardsdatascience.com/deep-dive-into-convolutional-networks-48db75969fdf)

## Quick look

If you want a quick look at the contents inside the notebook before deciding to run it please view the [pdf](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/ConvolutionalNeuralNetworks/CNN_Volcanic_deformation.pdf) generated (*note some html code note rendered*)

## Installation and Requirements

This notebook is designed to run on a laptop  with no special hardware required therefore recommended to do a local installation as outline in the repository [howtorun](../howtorun.md) and [jupyter_notebooks](../jupyter_notebooks.md) sections.


These notebooks require some additional data from the [VolcNet](https://github.com/matthew-gaddes/VolcNet) repository

If you have not already then in your gitbash or terminal please run the following code in the LIFD_ENV_ML_NOTEBOOKS directory via the terminal(mac or linux)  or git bash (windows)

```bash
git submodule init
git submodule update --init --recursive
```

**If this does not work please clone the [VolcNet](https://github.com/matthew-gaddes/VolcNet) repository into your ConvolutionalNeuralNetworks folder on your computer**

### Quick start

If you're already familiar with git, anaconda and virtual environments the environment you need to create is found in CNN.yml and the code below to install activate and launch the notebook

```bash
conda env create -f CNN.yml
conda activate CNN
jupyter-notebook
```
