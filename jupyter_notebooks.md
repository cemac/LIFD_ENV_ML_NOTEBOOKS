# Jupyter Notebooks

These tutorials are written as interactive [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) if you're new to notebooks we've included some information on how to get these running on your computer.

If you wish to run these notebooks on a remote machine with accelerated hardware then please follow [these instructions](https://github.com/cemac/cemac_generic/wiki/Jupyter-Notebooks-Via-SSH-Tunnelling)  

# Anaconda

1. Install anaconda or miniconda on your computer  via the appropriate installer found [here](https://conda.io/en/latest/miniconda.html)

2. Go to the folder required and install the yml file using conda e.g.
```
cd ConvolutionalNeuralNetworks
conda env create -f CNN.yml
conda activate CNN
```

*this is the non version specific environment - the most likely to install easily but in future may need some tweaking may be required the full list of versions of every package in the python environment are given in `absolute_enviroments` folder these can be used instead to create a more rigid environment or to track down version differences*

3. launch the notebook server
jupyter notebook --generate-config
jupyter-notebook
```

The notebook should launch in your browsers if not go to the address given in the terminal e.g.

`http://localhost:5555`

## Tips for running notebooks

* A quick overview of jupyter notebooks can be found [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
* If you want a clean copy and one to "play with" select `file`--> `make a copy` and rename your copy as a working copy.

## Toubleshooting

* most things will be fixed by a fresh install
* stange widget errors can be fixed with `jupyter nbextension enable --py --sys-prefix widgetsnbextension`
