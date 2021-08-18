<div style="background-color: #ccffcc; padding: 10px;">
    <h1> Tutorial 2 </h1> 
    <h2> Physics Informed Neural Networks Part 3</h2>
    <h2> PINN Navier Stokes Example </h2>
</div>    

# Overview

This notebook is based on two papers: *[Physics-Informed Neural Networks:  A Deep LearningFramework for Solving Forward and Inverse ProblemsInvolving Nonlinear Partial Differential Equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)* and *[Hidden Physics Models:  Machine Learning of NonlinearPartial Differential Equations](https://www.sciencedirect.com/science/article/pii/S0021999117309014)* with the help of  Fergus Shone and Michael Macraild.

These tutorials will go through solving Partial Differential Equations using Physics Informed Neuaral Networks focusing on the Burgers Equation and a more complex example using the Navier Stokes Equation

**This introduction section is replicated in all PINN tutorial notebooks (please skip if you've already been through)** 

<div style="background-color: #ccffcc; padding: 10px;">
If you have not already then in your repositoy directory please run the following code. Via the terminal (mac or linux) or gitbash (windows)
    
```bash
git submodule init
git submodule update --init --recursive
```
**If this does not work please clone the [PINNs](https://github.com/maziarraissi/PINNs) repository into your Physics_Informed_Neural_Networks folder**
    
</div>

<div style="background-color: #ccffcc; padding: 10px;">

<h1>Physics Informed Neural Networks</h1>

For a typical Neural Network using algorithims like gradient descent to look for a hypothesis, data is the only guide, however if the data is noisy or sparse and we already have governing physical models we can use the knowledge we already know to optamize and inform the algoithms. This can be done via [feature enginnering]() or by adding a physicall inconsistency term to the loss function.
<a href="https://towardsdatascience.com/physics-guided-neural-networks-pgnns-8fe9dbad9414">
<img src="https://miro.medium.com/max/700/1*uM2Qh4PFQLWLLI_KHbgaVw.png">
</a>   
  
 
## The very basics

If you know nothing about neural networks there is a [toy neural network python code example](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/tree/main/ToyNeuralNetwork) included in the [LIFD ENV ML Notebooks Repository]( https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS). Creating a 2 layer neural network to illustrate the fundamentals of how Neural Networks work and the equivlent code using the python machine learning library [tensorflow](https://keras.io/). 

    
## Recommended reading 
    
The in-depth theory behind neural networks will not be covered here as this tutorial is focusing on application of machine learning methods. If you wish to learn more here are some great starting points.   

* [All you need to know on Neural networks](https://towardsdatascience.com/nns-aynk-c34efe37f15a) 
* [Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
* [Physics Guided Neural Networks](https://towardsdatascience.com/physics-guided-neural-networks-pgnns-8fe9dbad9414)
* [Maziar Rassi's Physics informed GitHub web Page](https://maziarraissi.github.io/PINNs/)

</div>


<hr>


<div style="background-color: #e6ccff; padding: 10px;">
    
<h1> Machine Learning Theory </h1>
<a href="https://victorzhou.com/series/neural-networks-from-scratch/">
<img src="https://victorzhou.com/media/nn-series/network.svg">
</a>

    
## Physics informed Neural Networks

Neural networks work by using lots of data to calculate weights and biases from data alone to minimise the loss function enabling them to act as universal fuction approximators. However these loose their robustness when data is limited. However by using know physical laws or empirical validated relationships the solutions from neural networks can be sufficiently constrianed by disregardins no realistic solutions.
    
A Physics Informed Nueral Network considers a parameterized and nonlinear partial differential equation in the genral form;

\begin{align}
u_t + \mathcal{N}[u;  \lambda] = 0,   x \in \Omega, t \in [0,T],\\
\end{align}


where $\mathcal{u(t,x)}$ denores the hidden solution, $\mathcal{N}$ is a nonlinear differential operator acting on $u$, $\mathcal{\lambda}$ and $\Omega$ is a subset of $\mathbb{R}^D$ (the perscribed data). This set up an encapuslate a wide range of problems such as diffusion processes, conservation laws,  advection-diffusion-reaction  systems,  and  kinetic  equations and conservation laws. 

Here we will go though this for the 1D Heat equation and Navier stokes equations


</div>    

<div style="background-color: #cce5ff; padding: 10px;">

<h1> Python </h1>

    
## Tensorflow 
    
There are many machine learning python libraries available, [TensorFlow](https://www.tensorflow.org/) a is one such library. If you have GPUs on the machine you are using TensorFlow will automatically use them and run the code even faster!

## Further Reading

* [Running Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/running.html#running)
* [Tensorflow optimizers](https://www.tutorialspoint.com/tensorflow/tensorflow_optimizers.htm)

</div>
    
<hr>

<div style="background-color: #ffffcc; padding: 10px;">
    
<h1> Requirements </h1>

These notebooks should run with the following requirements satisfied

<h2> Python Packages: </h2>

* Python 3
* tensorflow > 2
* numpy 
* matplotlib
* scipy

<h2> Data Requirements</h2>
    
This notebook referes to some data included in the git hub repositroy
    
</div>


**Contents:**

1. [1D Heat Equation Non ML Example](PINNs_1DHeatEquations_nonML.ipynb)
2. [1D Heat Equation PINN Example](PINNs_1DEquationExample.ipynb)
3. **[Navier-Stokes PINNs discovery of PDE’s](PINNs_Navier_Stokes_example.ipynb)**
4. [Navier-Stokes PINNs Hidden Fluid Mechanics](PINNs_NavierStokes_HFM.ipynb)


<hr>

<div style="background-color: #cce5ff; padding: 10px;">
Load in all required modules (includig some auxillary code) and turn off warnings. 
</div>


```python
# For readability: disable warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
import sys
sys.path.insert(0, 'PINNs/Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from time import time
import scipy.sparse as sp
import scipy.sparse.linalg as la
```

<hr>
<div style="background-color: #ccffcc; padding: 10px;">

<h1> Navier-Stokes inverse data driven discovery of PDE’s </h1>

Navier-Stokes equations describe the physics of many phenomena of scientific and engineering interest. They may be used to model the weather, ocean currents, water flow in a pipe and air flow around a wing. The Navier-Stokes equations in their full and simplified forms help with the design of aircraft and cars, the study of blood flow, the design of power stations, the analysis of the dispersion of pollutants, and many other applications. Let us consider the Navier-Stokes equations in two dimensions (2D) given explicitly by

\begin{equation}    
u_t + \lambda_1 (u u_x + v u_y) = -p_x + \lambda_2(u_{xx} + u_{yy}),\\
v_t + \lambda_1 (u v_x + v v_y) = -p_y + \lambda_2(v_{xx} + v_{yy}),
\end{equation}
   
where $u(t, x, y)$ denotes the $x$-component of the velocity field, $v(t, x, y)$ the $y$-component, and $p(t, x, y)$ the pressure. Here, $\lambda = (\lambda_1, \lambda_2)$ are the unknown parameters. Solutions to the Navier-Stokes equations are searched in the set of divergence-free functions; i.e.,

\begin{equation} 
u_x + v_y = 0.
\end{equation}
       
This extra equation is the continuity equation for incompressible fluids that describes the conservation of mass of the fluid. We make the assumption that

\begin{equation}    
u = \psi_y,\ \ \ v = -\psi_x,
\end{equation}
</div>

<div style="background-color: #ccffcc; padding: 10px;">


for some latent function $\psi(t,x,y)$. Under this assumption, the continuity equation will be automatically satisfied. Given noisy measurements

\begin{equation}
\{t^i, x^i, y^i, u^i, v^i\}_{i=1}^{N}
\end{equation}
    
of the velocity field, we are interested in learning the parameters $\lambda$ as well as the pressure $p(t,x,y)$. We define $f(t,x,y)$ and $g(t,x,y)$ to be given by

\begin{equation}
\begin{array}{c}
f := u_t + \lambda_1 (u u_x + v u_y) + p_x - \lambda_2(u_{xx} + u_{yy}),\\
g := v_t + \lambda_1 (u v_x + v v_y) + p_y - \lambda_2(v_{xx} + v_{yy}),
\end{array}
\end{equation}

and proceed by jointly approximating 

\begin{equation}
\begin{bmatrix}
\psi(t,x,y) & p(t,x,y)
\end{bmatrix}
\end{equation}
    
using a single neural network with two outputs. This prior assumption results into a [physics informed neural network](https://arxiv.org/abs/1711.10566) 
    
\begin{equation}
\begin{bmatrix}
f(t,x,y) & g(t,x,y)
\end{bmatrix}.
\end{equation}
    
The parameters $\lambda$ of the Navier-Stokes operator as well as the parameters of the neural networks 

\begin{equation}
\begin{bmatrix}
\psi(t,x,y) & p(t,x,y)
\end{bmatrix}
\end{equation}
and 
    
\begin{equation}
\begin{bmatrix}
f(t,x,y) & g(t,x,y)
\end{bmatrix}
\end{equation}
    
can be trained by minimizing the mean squared error loss$

\begin{equation}
\begin{array}{rl}
MSE :=& \frac{1}{N}\sum_{i=1}^{N} \left(|u(t^i,x^i,y^i) - u^i|^2 + |v(t^i,x^i,y^i) - v^i|^2\right) \\
    +& \frac{1}{N}\sum_{i=1}^{N} \left(|f(t^i,x^i,y^i)|^2 + |g(t^i,x^i,y^i)|^2\right).
\end{array}
\end{equation}
    
</div>


```python
def xavier_init( size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

```


```python
def initialize_NN( layers):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
      
```

<div style="background-color: #ccffcc; padding: 10px;">

# Initalise the neural network 
    
`init` is called passing in the training data `x_train`, `y_train`, `t_train`, `u_train` and  `v_train` with information about the neural network layers
    
# Extract vars
    
`init` reformats some of the data and outputs model features that we need to pass into the training function `train`

</div>

<div style="background-color: #cce5ff; padding: 10px;">

# Advanced 
    
    
Once you have run through the notebook once you may wish to alter the optamizer used in the `init()` function to see the large effect optamizer choice may have. 
    
We've highlighted in the comments a number of possible optamizers to use from the [tf.compat.v1.train](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train) module. 
*This method was chosen to limit tensorflow version modifications required from the original source code*
    
You can learn more about different optamizers [here](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6)
    
</div>

# init


```python
def init(x, y, t, u, v, layers):
        # This line of code is required to prevent some tensorflow errors arrising from the
        # inclusion of some tensorflw v 1 code 
        tf.compat.v1.disable_eager_execution()
        X = np.concatenate([x, y, t], 1)
        # lb and ub denote lower and upper bounds on the inputs to the network
        # these bounds are used to normalise the network variables
        lb = X.min(0)
        ub = X.max(0)
                
        X = X
        
        x = X[:,0:1]
        y = X[:,1:2]
        t = X[:,2:3]
        
        u = u
        v = v
        
        layers = layers
        
        # Initialize NN
        weights, biases = initialize_NN(layers)        
        
        # Initialize parameters
        lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        
        
        
        # tf placeholders and graph
        ## This converts the data into a Tensorflow format
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
        y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, y.shape[1]])
        t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, t.shape[1]])

        u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, u.shape[1]])
        v_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, v.shape[1]])
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = net_NS(x_tf, y_tf, t_tf,lambda_1, lambda_2, weights, biases,lb, ub)

        loss = tf.reduce_sum(tf.square(u_tf - u_pred)) + \
            tf.reduce_sum(tf.square(v_tf - v_pred)) + \
            tf.reduce_sum(tf.square(f_u_pred)) + \
            tf.reduce_sum(tf.square(f_v_pred))

        
        ##############################################################################################
        #                                                                                            #
        ## the optimizer is something that can be tuned to different requirements                    #
        ## we have not investigated using different optimizers, the orignal code uses L-BFGS-B which # 
        ## is not tensorflow 2 compatible                                                            #
        #                                                                                            #
        #  SELECT OPTAMIZER BY UNCOMMENTING OUT one of the below lines AND RERUNNING CODE            #
        #  You can alsoe edit the learning rate to see the effect of that                            #
        #                                                                                            #
        ##############################################################################################
        
        learning_rate = 0.001
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9)
        # optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate) # 8 %
        # optimizer = tf.compat.v1.train.ProximalGradientDescentOptimizer(learning_rate)  
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate) 
        # optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate) # yeilds poor results
        # ptimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate) 
        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        
        # LEAVE THESE OPIMISERS ALONE
        optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        train_op_Adam = optimizer_Adam.minimize(loss)                    

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        xvars=[X,lb,ub,x,y,t,u,v]
        NNvars=[layers, weights, biases, lambda_1, lambda_2]
        tfvars=[sess, x_tf,y_tf, t_tf ,u_tf,v_tf]
        preds=[u_pred,v_pred, p_pred, f_u_pred, f_v_pred]
        optvars=[loss, optimizer,optimizer_Adam,train_op_Adam]
        return xvars,NNvars,tfvars,preds,optvars
```

<div style="background-color: #ccffcc; padding: 10px;">

`neural_net()` constructs the network Y where X is a matrix containing the input and output coordinates, i.e. x,t,u and X is normalised so that all values lie between -1 and 1, this improves training

`net_NS()` is where the PDE is encoded:
    
</div>


```python
def neural_net( X, weights, biases,lb, ub):
    
    num_layers = len(weights) + 1

    H = 2.0*(X - lb)/(ub - lb) - 1.0
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y



```


```python
def net_NS( x, y, t,lambda_1, lambda_2, weights, biases,lb,ub):
    
    psi_and_p = neural_net(tf.concat([x,y,t], 1), weights, biases, lb,ub)
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]

    u = tf.gradients(psi, y)[0]
    v = -tf.gradients(psi, x)[0]  

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]

    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)

    return u, v, p, f_u, f_v

```

<div style="background-color: #ccffcc; padding: 10px;">

# Load data and set input parameters 
   
</div>


<div style="background-color: #cce5ff; padding: 10px;">

Once you have run through the notebook once you may wish to alter any the following 
    
- number of data training points `N_train`
- number of layers in the network `layers`
- number of neurons per layer `layers`
    
to see the impact on the results

</div>


```python
N_train = 5000
# structure of network: 
# 8 fully connected layers with 20 nodes per layer
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
```


```python
# Load Data
data = scipy.io.loadmat('PINNs/main/Data/cylinder_nektar_wake.mat')

U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    
## We downsample the boundary data to leave N_train randomly distributed points
## This makes the training more difficult - 
## if we used all the points then there is not much for the network to do!
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]
```

<div style="background-color: #cce5ff; padding: 10px;">

If this fails you may need to restarted the notebook with a flag:
```bash


jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

```
</div>

<div style="background-color: #ccffcc; padding: 10px;">

# Initalise the nerual network 
    
`init` is called passing in the training data `x_train`,  `y_train`, `u_train` and `v_train` with information about the neural network layers. The bound information `lb` `ub` is included in the `init()` function
    
# Extract vars
    
`init` reformats some of the data and outputs model features that we need to pass into the training function `train`
    
</div>


```python
xvars, NNvars, tfvars, preds, optvars = init(x_train, y_train, t_train, u_train, v_train, layers)
X, lb, ub, x, y, t, u, v = xvars
layers, weights, biases, lambda_1, lambda_2 = NNvars
sess, x_tf,y_tf, t_tf ,u_tf,v_tf = tfvars
u_pred,v_pred, p_pred, f_u_pred, f_v_pred = preds
loss, optimizer, optimizer_Adam, train_op_Adam = optvars
```


```python
def train(sess, nIter,x_tf, y_tf, t_tf,u_tf, v_tf,x, y, t,u, v, loss, train_op_Adam, optimizer): 
    tf_dict = {x_tf: x, y_tf: y, t_tf: t,
             u_tf: u, v_tf: v}

    start_time = time()
    for it in range(nIter):
        sess.run(train_op_Adam, tf_dict)

        # Print
        if it % 50 == 0:
            elapsed = time() - start_time
            loss_value = sess.run(loss, tf_dict)
            lambda_1_value = sess.run(lambda_1)
            lambda_2_value = sess.run(lambda_2)
            print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
            start_time = time()

    return optimizer.minimize(loss)
    
```


```python
def predict( x_star, y_star, t_star, u_pred, v_pred, p_pred):

    tf_dict = {x_tf: x_star, y_tf: y_star, t_tf: t_star}

    u_star = sess.run(u_pred, tf_dict)
    v_star = sess.run(v_pred, tf_dict)
    p_star = sess.run(p_pred, tf_dict)

    return u_star, v_star, p_star

def plot_solution(X_star, u_star, index):
  
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
  
  
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
      
      
```


```python
# set random seeds
np.random.seed(1234)
tf.random.set_seed(1234)

```

<div style="background-color: #cce5ff; padding: 10px;">

**Training might take a long time depending on value of Train_iterations**

If you set Train_iterations too low the end results will be garbage. 20000 was used to achieve excellent results in the original papers but this value is too high to run on a laptop. 

* If you are using a machine with GPUs please set `Train_iterations=20000` to achieve the best results
* If you are using a well spec'ed laptop/computer and can leave this setting `Train_iterations=10000` should suffice (may take a while)
* If you are using a low spec'ed laptop/computer or cannont leave the code running `Train_interations=5000` is the reccomended values (high errors will remain)
    
</div>


```python
# Training
Train_iterations = 20000
```


```python
train(sess, Train_iterations,x_tf, y_tf, t_tf, u_tf, v_tf,x, y, t,u_train, v_train, loss, train_op_Adam, optimizer_Adam)
```

<div style="background-color: #ccffcc; padding: 10px;">

# Use trained model to predict from data sample
    
`predict` will predict `u`, `v` and `p` using the trained model

</div>


```python
# Test Data
snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

# Prediction
u_pred,v_pred, p_pred, f_u_pred, f_v_pred=preds
u_pred, v_pred, p_pred = predict(x_star, y_star, t_star, u_pred, v_pred, p_pred)
lambda_1_value = sess.run(lambda_1)
lambda_2_value = sess.run(lambda_2)
```

<div style="background-color: #ccffcc; padding: 10px;">

# Calculate Errors
    
if you have set the number of training iterations large enough the errors should be small.

</div>


```python
# Error
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
```


```python
print('Error u: %e' % (error_u))    
print('Error v: %e' % (error_v))    
print('Error p: %e' % (error_p))    
print('Error l1: %.5f%%' % (error_lambda_1))                             
print('Error l2: %.5f%%' % (error_lambda_2))        
```


```python
# Predict for plotting
lb = X_star.min(0)
ub = X_star.max(0)
nn = 200
x = np.linspace(lb[0], ub[0], nn)
y = np.linspace(lb[1], ub[1], nn)
X, Y = np.meshgrid(x,y)

UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
```

<div style="background-color: #ccffcc; padding: 10px;">

# Using Noisy Data
    
We're now going to repeat the previous steps but include some noise in our data to see the effect of that on our results

</div>


```python
######################################################################
########################### Noisy Data ###############################
######################################################################
noise = 0.01        
u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

```


```python
xvars, NNvars, tfvars, preds, optvars = init(x_train, y_train, t_train, u_train, v_train, layers)
X, lb, ub, x, y, t, u, v = xvars
layers, weights, biases, lambda_1, lambda_2 = NNvars
sess, x_tf, y_tf, t_tf ,u_tf,v_tf = tfvars
u_pred, v_pred, p_pred, f_u_pred, f_v_pred = preds
loss, optimizer, optimizer_Adam, train_op_Adam = optvars
```

<div style="background-color: #cce5ff; padding: 10px;">

**Training might take a while depending on value of Train_iterations**

If you set Train_iterations too low the end results will be garbage. 20000 was used to achieve excellent results. 

* If you are using a machine with [GPUs](https://towardsdatascience.com/what-is-a-gpu-and-do-you-need-one-in-deep-learning-718b9597aa0d) please set `Train_iterations` to 20000 and this will run in a few mins
* If you are using a well spec'ed laptop/computer then setting `Train_iterations=10000` but it will take a little while
* If you are using a low spec'ed laptop/computer or cannont leave the code running `Train_iterations=5000` is the reccomended value (this solution may not be accurate)
    
</div>


```python
# Training
train(sess, 20000, x_tf, y_tf, t_tf, u_tf, v_tf, x, y, t, u_train, v_train, loss, train_op_Adam, optimizer_Adam)
```


```python
lambda_1_value_noisy = sess.run(lambda_1)
lambda_2_value_noisy = sess.run(lambda_2)

error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100

print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
print('Error l2: %.5f%%' % (error_lambda_2_noisy))     
```


```python
######################################################################
############################# Plotting ###############################
######################################################################    
# Load Data
data_vort = scipy.io.loadmat('PINNs/main/Data/cylinder_nektar_t0_vorticity.mat')

x_vort = data_vort['x'] 
y_vort = data_vort['y'] 
w_vort = data_vort['w'] 
modes = np.asscalar(data_vort['modes'])
nel = np.asscalar(data_vort['nel'])    

xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')

box_lb = np.array([1.0, -2.0])
box_ub = np.array([8.0, 2.0])

fig, ax = plt.subplots()
ax.axis('off')
plt.figure(figsize=(16, 8))
####### Row 0: Vorticity ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
ax = plt.subplot(gs0[:, :])

for i in range(0, nel):
    h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)

ax.set_aspect('equal', 'box')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Vorticity', fontsize = 10)
plt.tight_layout()
```


```python
####### Row 1: Training data ##################
########      u(t,x,y)     ###################  
plt.figure(figsize=(20, 8))
gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
ax = plt.subplot(gs1[:, 0],  projection='3d')
ax.axis('off')

r1 = [x_star.min(), x_star.max()]
r2 = [data['t'].min(), data['t'].max()]       
r3 = [y_star.min(), y_star.max()]

for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
    if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

ax.scatter(x_train, t_train, y_train, s = 0.1)
# Predict for plotting
lb = X_star.min(0)
ub = X_star.max(0)
nn = 200
x = np.linspace(lb[0], ub[0], nn)
y = np.linspace(lb[1], ub[1], nn)
X, Y = np.meshgrid(x,y)
ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)

ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
ax.set_xlim3d(r1)
ax.set_ylim3d(r2)
ax.set_zlim3d(r3)
axisEqual3D(ax)

########      v(t,x,y)     ###################        
ax = plt.subplot(gs1[:, 1],  projection='3d')
ax.axis('off')

r1 = [x_star.min(), x_star.max()]
r2 = [data['t'].min(), data['t'].max()]       
r3 = [y_star.min(), y_star.max()]

for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
    if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

ax.scatter(x_train, t_train, y_train, s = 0.1)
ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)

ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
ax.set_xlim3d(r1)
ax.set_ylim3d(r2)
ax.set_zlim3d(r3)
axisEqual3D(ax)


```


```python
fig, ax = plt.subplots()
plt.figure(figsize=(20, 8))
ax.axis('off')

######## Row 2: Pressure #######################
########      Predicted p(t,x,y)     ########### 
gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
ax = plt.subplot(gs2[:, 0])
h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
            extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', 'box')
ax.set_title('Predicted pressure', fontsize = 10)

########     Exact p(t,x,y)     ########### 
ax = plt.subplot(gs2[:, 1])
h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
            extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', 'box')
ax.set_title('Exact pressure', fontsize = 12)
```

<div style="background-color: #ccffcc; padding: 10px;">
    
Predicted versus exact instantaneous pressure field at a representative time instant. By definition, the pressure can be recovered up to a constant, hence justifying the different magnitude between the two plots. This remarkable qualitative agreement highlights the ability of physics-informed neural networks to identify the entire pressure field, despite the fact that no data on the pressure are used during model training. 

**NB** train must be set to approx 10000 to achieve the desired results.
</div>


```python
######## Row 3: Table #######################
gs3 = gridspec.GridSpec(1, 2)
gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
ax = plt.subplot(gs3[:, :])
ax.axis('off')
plt.rc('text', usetex=False)
s=''
s = s + "Correct PDE \n "
s = s + "$u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})$"
s = s + "\n"
s = s + "$v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})$"
s = s + "\n \n \n"
s = s + r'Identified PDE (clean data) '
s = s + "\n"
s = s + '$u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})$' % (lambda_1_value, lambda_2_value)
s = s + "\n"
s = s + '$v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})$' % (lambda_1_value, lambda_2_value)
s = s + "\n\n \n"

s = s + r'Identified PDE (1% noise) & '
s = s + "\n"
s = s + '$u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})$' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + "\n"
s = s + '$v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})$' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + "\n"

plt.rc('font',family='serif')

plt.rc('font',size=16)
ax.text(0,0,s,fontsize=18)

# savefig('./figures/NavierStokes_prediction') 
```

<div style="background-color: #ccffcc; padding: 10px;"> 
    
if you have not been able to run enough training iterations the figures produced running 10000 iterations can be found:
    
* [Solution with network trained over 10000 iterations](figures/PINNS_NS_10000_PDE.png)
* [Figure comparing predicted vs exact with network trained over 10000 iterations](figures/PINNS_NS_10000_predict_vs_exact.png)

**Further Work**

Congratulations, you have now trained your another physics-informed neural network!

This network contains a number of hyper-parameters that could be tuned to give better results. Various hyper-parameters include:
- number of data training points `N_train`
- number of `layers` in the network
- number of neurons per layer
- optimisation 

It is also possible to use different sampling techniques for training data points. We randomly select $N_u$ data points, but alternative methods could be choosing only boundary points or choosing more points near the $t=0$ boundary.

return [here](#init) to alter optimization method used
    
</div>

<hr>

<div style="background-color: #e6ccff; padding: 10px;">

## Next steps

Now we've demonstrated using PINNs for more complex equations we can take a breif look at Hidden Fluid Mechanics (*this final notebook is beyond the scope of these tutorials but provided to give a breif example of the methodology*)
    
[Navier-Stokes PINNs Hidden Fluid Mechanics](PINNs_NavierStokes_HFM.ipynb)
</div>


```python

```
