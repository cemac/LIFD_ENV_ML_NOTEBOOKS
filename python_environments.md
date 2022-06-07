# Python Environments for Machine Learning

Machine learning python environments can be a bit of more of a headache than standard python environments.

In these tutorials a corresponding `.yml` file has been provided for each tutorial. As well as **conda recipes** in the README.mds. These have been combined into `Kerras.yml`, `Tensorflow.yml`, `GPflow.yml`, 'PyTorch.yml' and 'SKLearn.yml` so environments are not replicated and you can use this environment in the future for relevant projects.

There is also an overall `.yml` and conda recipe which will install all libraries, the best option will depend on how you're running these allnotebooks

## Using allnotebooks.yml file or allnotebooks_condarecipe.sh

This is the best option if:

* You intend to run all the notebooks on the same machine
* You have a spare few GB storage for the environment
* You're familiar with anaconda

Pros
1. One command and you're set up for all the notebooks
2. Provides an environment you can run most machine learning libraries from for your own work

Cons:

1. This will take a long time to install the environment
2. This will take up a lot of space
3. This is most likely option to go wrong/ need tweaking

## Using the individual .yml files or condarecipe.sh in each tutorial folder

This is the best option if

* You only intend to run a few notebooks
* You're running notebooks on different machines, e.g. your laptop, university machines
* You're running on a machine with old NVIDIA drivers e.g. foe-linux

## OS differences

**Tensorflow**

Windows latest Tensorflow version is behind Linux and Mac, we've tried to make the notebooks compatible with the version available on both.

**PyTorch**

Depending on your OS the NVIDIA driver, CUDA and PyTorch version compatibility will be different

https://docs.nvidia.com/deploy/cuda-compatibility/index.html
