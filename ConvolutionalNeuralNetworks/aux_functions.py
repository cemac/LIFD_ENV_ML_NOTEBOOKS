#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:43:13 2020

@author: matthew
"""
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Maths and 
import numpy.ma as ma 
# Premade data is provided as pickles
import pickle
# Plotting utilies
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%

def centre_to_box(centre_width):
    """ A function to convert from centre and width notation to start and stop notation (in both x and y directions).   
    Inputs:
        centre_wdith
    Returns:
    History:
        2020_10_28 | MEG | Wrote the docs.  
        """
    x_start = centre_width[0] - centre_width[2]
    x_stop = centre_width[0] + centre_width[2]
    y_start = centre_width[1] - centre_width[3]
    y_stop = centre_width[1] + centre_width[3]
    return [x_start, x_stop, y_start, y_stop]
     
  
#%%

def add_square_plot(x_start, x_stop, y_start, y_stop, ax, colour = 'k'):
    """Draw localization square around an area of interest, x_start etc are in pixels, so (0,0) is top left.
    Inputs:
        x_start | int | start of box
        x_stop | int | etc.
        y_start | int |
        y_ stop | int |
        ax | axes object | axes on which to draw
        colour | string | colour of bounding box.  Useful to change when plotting labels, and predictions from a model.  
    Returns:
        box on figure
    History:
        2019/??/?? | MEG | Written
        2020/04/20 | MEG | Document, copy to from small_plot_functions to LiCSAlert_aux_functions
    """
    ax.plot((x_start, x_start), (y_start, y_stop), c= colour)           # left hand side
    ax.plot((x_start, x_stop), (y_stop, y_stop), c= colour)             # bottom
    ax.plot((x_stop, x_stop), (y_stop, y_start), c= colour)             # righ hand side
    ax.plot((x_stop, x_start), (y_start, y_start), c= colour)             # top
   

#%%

def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range (i.e. truncate the colormap so that it isn't
    compressed on the shorter side) . Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin))
          Only got this to work with:
              1 - vmin/(vmax + abs(vmin))
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))

      2017/??/?? | taken from stack exchange
      2017/10/11 | update so that crops shorter side of colorbar (so if data are in range [-1 100],
                   100 will be dark red, and -1 slightly blue (and not dark blue))
      '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    if midpoint > 0.5:                                      # crop the top or bottom of the colourscale so it's not asymetric.
        stop=(0.5 + (1-midpoint))
    else:
        start=(0.5 - midpoint)


    cdict = { 'red': [], 'green': [], 'blue': [], 'alpha': []  }
    # regular index to compute the colors
    reg_index = np.hstack([np.linspace(start, 0.5, 128, endpoint=False),  np.linspace(0.5, stop, 129)])

    # shifted index to match the data
    shift_index = np.hstack([ np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap


#%%

def normalise_m1_1(r2_array):
    """ Rescale a rank 2 array so that it lies within the range[-1, 1]
    """
    import numpy as np
    r2_array = r2_array - np.min(r2_array)
    r2_array = 2 * (r2_array/np.max(r2_array))
    r2_array -= 1
    return r2_array

def open_VolcNet_file(file_path, defo_sources):
    """A file to open a single VolcNet file and extrast the deformation source into a one hot encoded numpy array, 
    and the deforamtion location as a n_ifg x 4 array.  
    Ifgs are masked arrays, in m, with up as positive.  
    
    Inputs:
        file_path | string or Path | path to fie
        defo_sources | list of strings | names of deformation sources, should the same as the names used in VolcNet
    
    Returns:
        X | r4 masked array | ifgs, as above. ? x y x x n_channels  
        Y_class | r2 array | class labels, ? x n_classes
        Y_loc | r2 array | locations of signals, ? x 4 (as x,y, width, heigh)
    
    History:
        2020_01_11 | MEG | Written
    """
    import pickle
    import numpy as np
     
    # 0: Open the file    
    with open(file_path, 'rb') as f:       # open the real data file
        ifgs = pickle.load(f)              # this is a masked array of ifgs
        ifgs_dates = pickle.load(f)        # list of strings, YYYYMMDD that ifgs span
        pixel_lons = pickle.load(f)        # numpy array of lons of lower left pixel of ifgs
        pixel_lats = pickle.load(f)        # numpy array of lats of lower left pixel of ifgs
        all_labels = pickle.load(f)        # list of dicts of labels associated with the data.  e.g. deformation type etc.  
    f.close()        
    
    # 1: Initiate arrays
    n_ifgs = ifgs.shape[0]                                                                      # get number of ifgs in file
    X = ifgs                                                                                    # soft copy to rename
    Y_class = np.zeros((n_ifgs, len(defo_sources)))                                             # initiate
    Y_loc = np.zeros((n_ifgs, 4))                                                               # initaite

    # 2: Convert the deformation classes to a one hot encoded array and the locations to an array
    for n_ifg in range(n_ifgs):                                                                 # loop through the ifgs
        current_defo_source = all_labels[n_ifg]['deformation_source']                           # get the current ifgs deformation type/class label
        arg_n = defo_sources.index(current_defo_source)                                         # get which number in the defo_sources list this is
        Y_class[n_ifg, arg_n] = 1                                                               # write it into the correct position to make a one hot encoded list.  
        Y_loc[n_ifg, :] = all_labels[n_ifg]['deformation_location']                             # get the location of deformation.  
        
    return X, Y_class, Y_loc

def plot_data_class_loc(data, plot_args, classes=None, locs=None, classes_predicted=None,
                        locs_predicted=None, source_names = None, point_size=5, 
                        figsize = (12,8), window_title = None):
    """A figure to plot some data, add the class and predicted class, and add the location and predicted location
    Inputs: 
        X_m | rank 4 masked array |ifgs in metres, with water and incoherence masked 
              rank 4 array |        can be CNN input with no masking and any range too.                        
        plot_args | rank 1 array | Which data to plot (e.g.: array([ 0,  1,  2,  3,  4]))
        classes             | array | one hot encoding
        locs                | rank 2 array | nx4,  locations of deforamtion, columns are x, y, x half width, y half width
        classes_predicted   | array | as per classes
        locs_predicted      | rank 2 array | as per locs
        source_names        |
        point_size          | int | size of the dot at the centre of the deformation
        figsize             | tuple | Size of figure in inches.  
        window_title        | None or string | Sets the title of the window, if not None
    Returns:
        Figure
    History:
        2019/??/?? | MEG | Written
        2020/10/21 | MEG | Moved to the detect_locate github repo.  
        2020/10/21 | MEG | Update so that no colorbars don't change the units (previously *100 to convert m to cm)
        2020/10/28 | MEG | Update docs    """    

    cmap_noscaling = plt.get_cmap('coolwarm')
    label_fs = 14
    n_rows = 3
    n_cols = 5
    n_plots = int(len(plot_args))
    f1, axes = plt.subplots(n_rows, n_cols, figsize = figsize)
    if window_title is not None:
        f1.canvas.set_window_title(window_title)
    for n_plot in range(n_plots):    # loop through each plot arg
        # convert axes to a rank 1 so that it's easy to index them as we loop through the plots
        axe = np.ravel(axes)[n_plot] 
        #1: Draw the ifg (each ifg has its own colourscale)
        ifg_min = ma.min(data[plot_args[n_plot], :,:,0])   # min of ifg being plotted
        ifg_max = ma.max(data[plot_args[n_plot], :,:,0])   # max
        ifg_mid = 1 - ifg_max/(ifg_max + abs(ifg_min))     # mid point
        cmap_scaled = remappedColorMap(cmap_noscaling, start=0, midpoint=ifg_mid, stop=1.0, 
                                       name='cmap_noscaling')    # rescale cmap to be the correct size
        axe.imshow(data[plot_args[n_plot], :,:,0], cmap = cmap_scaled) # plot ifg with camp
        
        #2: Draw a colorbar
        # colourbar is plotted within the subplot axes
        axe_cbar = inset_axes(axe, width="40%", height="3%", loc=8, borderpad=0.3) 
        norm2 = mpl.colors.Normalize(vmin=ifg_min, vmax=ifg_max)                                                        # No change to whatever the units in X are.
        cb2 = mpl.colorbar.ColorbarBase(axe_cbar, cmap=cmap_scaled, norm = norm2, orientation = 'horizontal')
        cb2.ax.xaxis.set_ticks_position('top')
        if np.abs(ifg_max) > np.abs(ifg_min):
            cb2.ax.xaxis.set_ticks([np.round(0,0), np.round(ifg_max, 3)])
        else:
            cb2.ax.xaxis.set_ticks([np.round(ifg_min,3), np.round(0,0)])       
        
        #3: Add labels/locations
        if locs is not None:
            # covert from centre width notation to start stop notation, # [x_start, x_stop, Y_start, Y_stop]
            start_stop_locs = centre_to_box(locs[plot_args[n_plot]])  
            add_square_plot(start_stop_locs[0], start_stop_locs[1], 
                            start_stop_locs[2], start_stop_locs[3], axe, colour='k')  # box around deformation
            axe.scatter(locs[plot_args[n_plot], 0], locs[plot_args[n_plot], 1], s = point_size, c = 'k')
            
        if locs_predicted is not None:
            # covert from centre width notation to start stop notation, # [x_start, x_stop, Y_start, Y_stop]
            start_stop_locs_pred = centre_to_box(locs_predicted[plot_args[n_plot]])    
            add_square_plot(start_stop_locs_pred[0], start_stop_locs_pred[1], 
                            start_stop_locs_pred[2], start_stop_locs_pred[3], 
                            axe, colour='r')    # box around deformation
            axe.scatter(locs_predicted[plot_args[n_plot], 0],
                        locs_predicted[plot_args[n_plot], 1], s = point_size, c = 'r')
        
        if classes is not None and classes_predicted is None:    # if only have labels and not predicted lables
            label = np.argmax(classes[plot_args[n_plot], :])     # labels from the cnn
            axe.set_title(f'Ifg: {plot_args[n_plot]}, Label: {source_names[label]}' , fontsize=label_fs)
            
        elif classes_predicted is not None and classes is None:  # if only have predicted labels and not labels
            label_model = np.argmax(classes_predicted[plot_args[n_plot]])   # original label
            value_model = str(np.round(np.max(classes_predicted[plot_args[n_plot]]),
                                       2))# max value from that row
            axe.set_title(f'Ifg: {plot_args[n_plot]},  CNN label: {source_names[label_model]} ({value_model})',
                          fontsize=label_fs)
        else:                                                      # if we have both predicted labels and lables
            label = np.argmax(classes[plot_args[n_plot], :])      # labels from the cnn
            label_model = np.argmax(classes_predicted[plot_args[n_plot]])   # original label
            # max value from that row
            value_model = str(np.round(np.max(classes_predicted[plot_args[n_plot]]), 2))     
            axe.set_title(f'Ifg: {plot_args[n_plot]}, Label: {source_names[label]}\nCNN label: {source_names[label_model]} ({value_model})',
                          fontsize=label_fs)
        
        
        axe.set_ylim(top = 0, bottom = data.shape[1])
        axe.set_xlim(left = 0, right= data.shape[2])
        axe.set_yticks([])
        axe.set_xticks([])
        
        if n_plots < 15:       # remove any left over/unused subplots
            axes_to_del = np.ravel(axes)[(n_plots):]
            for axe_to_del in axes_to_del:
                axe_to_del.set_visible(False)
                
def plot_data_class_loc_caller(X_m, classes=None, locs=None, classes_predicted=None, locs_predicted=None, 
                               source_names = None, point_size=5, figsize = (24,16), window_title = None):
    """ A function to call plot_data_class_loc to plot more than 15 ifgs.  
    
    Inputs as per 'plot_data_class_loc'   
    History:
        2019/??/?? | MEG | Written
        2020/11/11 | MEG | Fix bug in how window_title is handled when there is only one plot
    """    
    n_data = X_m.shape[0]
    n_plots = int(np.ceil(n_data/15))
    all_args = np.arange(0,n_data)
    
    if n_plots == 1:
        plot_data_class_loc(X_m, all_args, classes, locs, classes_predicted, locs_predicted,
                            source_names, point_size, figsize, window_title)
    
    else:
        for n_plot in np.arange(n_plots-1):
            plot_args = np.arange(n_plot*15, (n_plot*15)+15)
            plot_data_class_loc(X_m, plot_args, classes, locs, classes_predicted, locs_predicted,
                                source_names, point_size, figsize, window_title)
        
        plot_args = np.arange((n_plot+1)*15, n_data)     # plot the last one that might have some blank spaces
        plot_data_class_loc(X_m, plot_args, classes, locs, classes_predicted, locs_predicted,
                            source_names, point_size, figsize, window_title)

    
def open_datafile_and_plot(file_path, n_data = 15, rad_to_m_convert = False,
                           window_title = None):
    """ A function to open a .pkl file of ifgs and quickly plot the first n_data ifgs.  
    Inputs:
        pkl_path | path or string | path to pkl file to be opened
        n_data | int | the first n_data data in the pkl will be plotted.  
        rad_to_m_convert | boolean | If True, data are in sentinel-1 rads and are convereted to .  
        window_title        | None or string | Sets the title of the window, if not None
    Returns:
        figure
    History:
        2020/10/28 | MEG | Written
        2020/11/11 | MEG | Update to handle either .pkl or .npz
    """  
    s1_wav = 0.055465763              # in metres
    
    if file_path.split('.')[-1] == 'pkl':  # if it's a .pkl
        with open(file_path, 'rb') as f:   # open the file
            X = pickle.load(f)             # and extract data (X) and labels (Y)
            Y_class = pickle.load(f)
            Y_loc = pickle.load(f)
        f.close()
    elif file_path.split('.')[-1] == 'npz': # if it's a npz
        data = np.load(file_path)           # load it
        X = data['X']                       # and extract data (X) and labels (Y)
        Y_class = data['Y_class']
        Y_loc = data['Y_loc']
    else:                                   # no other file types are currently supported
        raise Exception(f"Error!  File was not understood as either a .pkl or a .npz so exiting.  ")
    
    if type(X) is dict:
        print('The variable X contained in this dictionary is dictionary, which usually means it contains the same data stored in various formats.  '
              'Taking the uuu phase and converting it to metres.  ')
        X = X['uuu']
        rad_to_m_convert = True
    
    if rad_to_m_convert:
        X = X *(s1_wav / (4 * np.pi))                                                                                              # convert from unwrapped phase to metres (for Sentinel-1)
        
    plot_data_class_loc_caller(X[:n_data,], Y_class[:n_data,], Y_loc[:n_data,], source_names = ['dyke', 'sill', 'no def'],
                               window_title = window_title)    

def shuffle_arrays(data_dict):
    """A function to shuffle a selection of arrays along their first axis.  
    The arrays are all shuffled in the same way (so good for data and labels)
    Inputs:
        data_dict | dictionary | containing e.g. X, X_m, Y_class, Y_loc
    Returns:
        data_dict_shuffled | dictionary | containing e.g. X, X_m, Y_class, Y_loc, shuffled along first axis
    History:
        2019/??/?? | MEG | Written
        2020/10/28 | MEG | Comment.  
            
    """
    import numpy as np
    
    data_dict_shuffled = {}                                                         # initiate
    args = np.arange(0, data_dict['X'].shape[0])                                    # get the numbers along the first dim (which is the number of data)
    np.random.shuffle(args)                                                         # shuffle this
    
    for data_label_name in data_dict:                                               # loop through each array in dictionary 
        data_dict_shuffled[data_label_name] = data_dict[data_label_name][args,:]    # and then copy to the new dict in the shuffled order (set by args)
        
    return data_dict_shuffled

def augment_flip(X, Y_loc, flip):
    """A function to flip data horizontally or vertically 
    and apply the same transformation to the location label
    Inputs:
        X | r4 array | samples x height x width x channels
        Y_loc | r2 array | samples X 4
        flip | string | determines which way to flip.  
    """
    import numpy as np
    
    Y_loc_flip = np.copy(Y_loc)                              # make a copy of the location labels
    
    if flip is 'up_down':                                       # conver the string input to a value
        X_flip = X[:,::-1,:,:]                              # reverse in dim 2 which is y
        Y_loc_flip[:,1] = X.shape[1] - Y_loc_flip[:,1]
    elif flip is 'left_right':
        X_flip = X[:,:,::-1,:]                              # reverse in dim 3 which is x
        Y_loc_flip[:,0] = X.shape[2] - Y_loc_flip[:,0]      # flipping horizontally
    elif flip is 'both':
        X_flip = X[:,::-1,::-1,:]                              # reverse in dim 2 (y) and dim 3 (x)
        Y_loc_flip[:,1] = X.shape[1] - Y_loc_flip[:,1]
        Y_loc_flip[:,0] = X.shape[2] - Y_loc_flip[:,0]      # flipping horizontally
    else:
        raise Exception("'flip' must be either 'up_down', 'left_right', or 'both'.  ")
    
    return X_flip, Y_loc_flip





def augment_rotate(X, Y_loc):
    """ Rotate data and the label.  Angles are random in range [0 360], and different for each sample.  
    Note: Location labels aren't rotated!  Assumed to be roughly square.  
    Inputs:
        X | r4 array | samples x height x width x channels
        Y_loc | r2 array | samples X 4
    Returns:
        
    """
    import numpy as np
    import numpy.ma as ma
    
    def rot(image, xy, angle):
        """Taken from stack exchange """
        from scipy.ndimage import rotate
        im_rot = rotate(image,angle, reshape = False, mode = 'nearest') 
        org_center = (np.array(image.shape[:2][::-1])-1)/2.
        rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        return im_rot, new+rot_center

    X_rotate = ma.copy(X)
    Y_loc_rotate = ma.copy(Y_loc)
    rotate_angles_deg = np.random.randint(0, 360, X.shape[0])

    for n_ifg, ifg in enumerate(X):                                                                     #loop through each ifg
        ifg_rot, xy_rot = rot(ifg, Y_loc_rotate[n_ifg,:2], rotate_angles_deg[n_ifg])
        X_rotate[n_ifg,:,:,:] = ifg_rot
        Y_loc_rotate[n_ifg, :2] = xy_rot

    return X_rotate, Y_loc_rotate

def augment_translate(X, Y_loc, max_translate = (20,20)):
    """
    Inputs:
        max_translate | tuple | max x translation, max y translation
    """
    import numpy as np
    import numpy.ma as ma
    
    n_pixs = X.shape[1]                                                             # normally 224

    X_translate = ma.copy(X)
    Y_loc_translate = ma.copy(Y_loc)
    x_translations = np.random.randint(0, 2*max_translate[0], X.shape[0])           # translations could be + or - max translation, but everything is positive when indexing arrays so double the max translation
    y_translations = np.random.randint(0, 2*max_translate[1], X.shape[0])
    
    Y_loc_translate[:,0] -= x_translations - max_translate[0]                                  # these are the x centres
    Y_loc_translate[:,1] -= y_translations - max_translate[1]                                # these are the y centres
    
    
    for n_ifg, ifg in enumerate(X):                                                                                                                         #loop through each ifg, but ma doesn't have a pad (ie can't pad masked arrays)
        ifg_large_data = np.pad(ma.getdata(ifg), ((max_translate[1],max_translate[1]),(max_translate[0], max_translate[0]), (0,0)), mode = 'edge')          # padding the data  (y then x then channels)
        ifg_large_mask = np.pad(ma.getmask(ifg), ((max_translate[1],max_translate[1]),(max_translate[0], max_translate[0]), (0,0)), mode = 'edge')          # padding the mask (y then x then channels)
        ifg_large = ma.array(ifg_large_data, mask=ifg_large_mask)                                                                                           # recombining the padded mask and data to make an enlarged masked array
        ifg_crop = ifg_large[y_translations[n_ifg]:y_translations[n_ifg]+n_pixs,x_translations[n_ifg]:x_translations[n_ifg]+n_pixs, :]                      # crop from the large ifg back to the original resolution
        X_translate[n_ifg,:,:,:] = ifg_crop                                                                                                                 # append result to big rank 4 of ifgs

    return X_translate, Y_loc_translate

def choose_for_augmentation(X, Y_class, Y_loc, n_per_class):
    """A function to randomly select only some of the data, but in  fashion that ensures that the classes 
    are balanced (i.e. there are equal numbers of each class).  Particularly useful if working with real 
    data, and the classes are usually very unbalanced (lots of no_def lables, normally).  
    Inputs:
        X           | rank 4 array | data.  
        Y_class     | rank 2 array | One hot encoding of class labels
        Y_loc       | rank 2 array | locations of deformation
        n_per_class | int | number of data per class. e.g. 3
    Returns:
        X_sample           | rank 4 array | data.  
        Y_class_sample     | rank 2 array | One hot encoding of class labels
        Y_loc_sample       | rank 2 array | locations of deformation
    History:
        2019/??/?? | MEG | Written
        2019/10/28 | MEG | Update to handle dicts
        2020/10/29 | MEG | Write the docs.  
        2020/10/30 | MEG | Fix bug that was causing Y_class and Y_loc to become masked arrays.  
        
    """
    
    n_classes = Y_class.shape[1]        # only works if one hot encoding is used
    X_sample = []
    Y_class_sample = []
    Y_loc_sample = []
    
    for i in range(n_classes):                                                              # loop through each class
        args_class = np.ravel(np.argwhere(Y_class[:,i] != 0))                               # get the args of the data of this label
        args_sample = args_class[np.random.randint(0, len(args_class), n_per_class)]        # choose n_per_class of these (ie so we always choose the same number from each label)
        X_sample.append(X[args_sample, :,:,:])                                              # choose the data, and keep adding to a list (each item in the list is n_per_class_label x ny x nx n chanels)
        Y_class_sample.append(Y_class[args_sample, :])                                      # and class labels
        Y_loc_sample.append(Y_loc[args_sample, :])                                          # and location labels
    
    X_sample = ma.vstack(X_sample)                                                          # maskd array, merge along the first axis, so now have n_class x n_per_class of data
    Y_class_sample = np.vstack(Y_class_sample)                                              # normal numpy array, note that these would be in order of the class (ie calss 0 first, then class 1 etc.  )
    Y_loc_sample = np.vstack(Y_loc_sample)                                                  # also normal numpy array
    
    data_dict = {'X'       : X_sample,                                                      # package the data and labels together into a dict
                 'Y_class' : Y_class_sample,
                 'Y_loc'   : Y_loc_sample}
    
    data_dict_shuffled = shuffle_arrays(data_dict)                                          # shuffle (so that these aren't in the order of the class labels)
    
    X_sample = data_dict_shuffled['X']                                                      # and unpack as this function doesn't use dictionaries
    Y_class_sample = data_dict_shuffled['Y_class']
    Y_loc_sample = data_dict_shuffled['Y_loc']
    
    return X_sample, Y_class_sample, Y_loc_sample

def augment_data(X, Y_class, Y_loc, n_data = 500):
    """ A function to augment data and presserve the location label for any deformation.  
    Note that n_data is not particularly intelligent as many more data may be generated,
    and only n_data returned, so even if n_data is low, the function can still be slow.  
    Inputs:
        X           | rank 4 array | data.  
        Y_class     | rank 2 array | One hot encoding of class labels
        Y_loc       | rank 2 array | locations of deformation
        n_data      | int |
    Returns:
        X_aug           | rank 4 array | data.  
        Y_class_aug     | rank 2 array | One hot encoding of class labels
        Y_loc_aug       | rank 2 array | locations of deformation
    History:
        2019/??/?? | MEG | Written
        2020/10/29 | MEG | Write the docs.  
        2020_01_11 | MEG | Major rewrite to speed things up.  
    """

    # the three possible types of flip
    flips = ['none', 'up_down', 'left_right', 'both']     

    # 0: get the correct nunber of data    
    n_ifgs = X.shape[0]
    # package the data and labels together into a dict
    data_dict = {'X'       : X,                                                    
                 'Y_class' : Y_class,
                 'Y_loc'   : Y_loc}
    # if we have fewer ifgs than we need, repeat them
    if n_ifgs < n_data:      
        n_repeat = int(np.ceil(n_data / n_ifgs)) 
        # get the number of repeats needed (round up and make an int)
        data_dict['X'] = ma.repeat(data_dict['X'], axis = 0, repeats = n_repeat)
        data_dict['Y_class'] = np.repeat(data_dict['Y_class'], axis = 0, repeats = n_repeat)
        data_dict['Y_loc'] = np.repeat(data_dict['Y_loc'], axis = 0, repeats = n_repeat)
    # shuffle (so that these aren't in the order of the class labels)
    data_dict = shuffle_arrays(data_dict)        
    for key in data_dict:   
        # then crop them to the correct number
        data_dict[key] = data_dict[key][:n_data,]
    # and unpack as this function doesn't use dictionaries            
    X_aug = data_dict['X']    
    Y_class_aug = data_dict['Y_class']
    Y_loc_aug = data_dict['Y_loc']

    # 1: do the flips        
    for data_n in range(n_data):
        flip = flips[np.random.randint(0, len(flips))]  # choose a flip at random
        if flip != 'none':
            X_aug[data_n:data_n+1,], Y_loc_aug[data_n:data_n+1,] = augment_flip(X_aug[data_n:data_n+1,], Y_loc_aug[data_n:data_n+1,], flip)          # do the augmentaiton via one of the flips.  
        
    # 2: do the rotations
    X_aug, Y_loc_aug = augment_rotate(X_aug, Y_loc_aug) # rotate 
    
    # 3: Do the translations
    X_aug, Y_loc_aug = augment_translate(X_aug,  Y_loc_aug, max_translate = (20,20)) 
        
    return X_aug, Y_class_aug, Y_loc_aug    
