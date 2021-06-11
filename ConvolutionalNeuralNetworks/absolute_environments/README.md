# Python environments

These two files [nn_win.yml](nn_win.yml) and [nn_linux.yml](nn_linux.yml) contain the full running environments used to run on those OS.  

If the environment [CCN.yml](../CCN.yml) doesn't successfully run the notebook there might be a newer version of a library installed that is not compatible. You will need to compare your versions specifically the following packages:

* python=3.8
* keras
* tensorflow=2.3
* pydot
* graphviz
* ipdb
* matplotlib=3.0
* basemap-data-hires
* geopy

you can do this by running

`conda list | grep <package name>`
`conda install <package name>=<version_number>`
