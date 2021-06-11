# Rough notes

These notes are not for final notes

## Science and plan:

Github repo: [https://github.com/matthew-gaddes/VUDLNet_21](https://github.com/matthew-gaddes/VUDLNet_21)

There's a preprint of the paper on EarthArxiv - [https://eartharxiv.org/repository/view/1969/](https://eartharxiv.org/repository/view/1969/)  Figures 4 and 6 might be useful.  

Keras blog post: [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html](https://eartharxiv.org/repository/view/1969/)

In terms of structure, the main file (detect_locate_cnn.py) is broken into steps and may provide enough structure for a notebook:
  1.  Get a DEM for a selection of volcanoes.  
  2.  Make synthetic interferogrms for volcanoes.  
  3.  Import real data (from VolcNet).  
  4.  Merge and rescale real and synthetic data.  
  5.  Compute bottleneck features.  
  6.  Train fully connected network.  
  7.  Fine tune the fully connected network and the 5th convolutional block.  

## Background knowledge assumed:

know what CNN’s are but never used it. Matthew suggests not using anything simpler than starting with his data set.

Bottleneck files need to train network if they want to be running in the 2 hours. Minimum working example doesn’t show great results, could have the option to include extra data to see full results. Pre trained model is huge. 48 GB file bottle neck features 4GB to get full results but could be reduced.  

## Introductory Material

* [Towards Data science](https://towardsdatascience.com/deep-dive-into-convolutional-networks-48db75969fdf)
* [Victor Zhou's blog](https://victorzhou.com/blog/intro-to-cnns-part-1/)
* [PyImageSearch](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)
* [Machine Learning mastery](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)

The best resource, however, is probably [Francois Chollet's book on the subject](https://github.com/fchollet).   
There's a questionable (morally) [pdf of it on the net](http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf)
