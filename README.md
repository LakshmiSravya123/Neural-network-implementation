# Neural-network-implementation
NN implementation without SKlearn and built in packages


## Description
This program consists of the implementation of a feedforward neural network
(or multi-layer perceptron, MLP), with an arbitrary number of layers, for
multiclass classification. 

The output layers of the neural networks considered here are made of m
neurons that are fully connected to the previously hidden layers. They are
equipped with a softmax non-linearity.
The output of the jth neuron of the output layer gives a score for the class
j, which is interpreted as the probability of x being of class j.



## Installation
Download the NN.py

## Dataset

The CIFAR-10 dataset consists of 60000 32×32 color images in 10 classes,
with 6000 images per class. We are going to use 49000 images for training,
1000 images for validation and 10000 images for testing. Each image,
originally represented with 3 × 32 × 32 matrices, has been flattened, meaning
that each example is represented with an array of size 3072
 
## Import
import numpy as np

## Parameters

1) hidden_dims: a tuple containing the number of neurons in each hidden
layer. The tuple size is the number of hidden layers of your
network (meaning that if there are L − 1 hidden layer, there should
be L+1 layers in total if we count the input and output layers).

2) datapath: a string containing the path to the dataset. The code to
load the dataset and split it into training, validation, and test sets
is given to you in the __init__ function.

3) n_classes: the number of classes in the classification problem (also
the number of neurons of the output layer).

4) epsilon: a number " 2 (0, 1) that is used to clip tiny probabilities
and substantial probabilities when evaluating the cross-entropy
loss. You don’t need to change its value from the default 10−6.

5) LR: the learning rate used to train the neural network with minibatch
stochastic gradient descent (SGD). 

6) batch_size: the batch size for minibatch SGD.

7) seed: the random seed that ensures that two runs with the same seed
yield the same results. 

8) activation: a string describing the activation function. It can be
"relu," "sigmoid," or "tanh." We remind you of the three activation
functions:

## Functions
1) NN.initialize_weights function: This function sets
the random seed and creates a dictionary self.weights (that also
contains the biases). This function's input is a list dims of size 2, containing the input dimension and the number of classes.
As you know, it is necessary to initialize the parameters randomly
your neural network (trying to avoid symmetry and saturating neurons,
and ideally, so that the pre-activation lies in the bending region
of the activation function so that the overall networks act as a non
linear function). You have to sample the weights of a layer from a
uniform distribution

2) NN.sigmoid function that returns for an input x:
sigma(x) when grad is set to False, and sigma'(x) otherwise.

3) NN.relu function, that returns for an input x:
RELU(x) when grad is set to False, and RELU'(x) otherwise.

4) NN.tanh function, that returns for an input x:
tanh(x) when grad is set to False, and tanh'(x) otherwise.

5) NN.activation function: that returns the right activation function,
evaluated at its input x (with the variable grad taken into
account). Remember that the activation function used is stored
in the self.activation_str variable.

6) NN.softmax function:
The function takes as input a NumPy array x and should return an
array (of the same size as x) corresponding to softmax(x). This
the function should work for any input representing a minibatch
of examples, i.e., a batch_size × n_dimensions matrix.

7) NN.forward
function: This function propagates its input x through the layers of
the neural network. The results of the forward propagation should
be stored in the cache dictionary. The keys of the dictionary should
be called "Z0", "A1", "Z1", ..., "AL," "ZL," where L − 1 is the number
of hidden layers. For simplicity, you can use Python’s f-Strings to access the weights easily (e.g., cache[f"Z{layer_n}"]) if layer_n is
a variable used to loop through the number of layers.
Cache ["Ai"] should store the pre-activation at layer I (i.e., before
applying the activation function). Cache ["Zi"] should store the activation
at layer i. Following this logic, cache["ZL"] should be a vector
of probabilities.

8) NN.backward
function: This function takes as input the cache evaluated using the
NN.forward function on a mini-batch, and the labels of the minibatch examples as a matrix of size batch_size×n_classes. Note that
labels are the one-hot encodings of the labels (c.f. question 7). The
function should populate the grads dictionary and return it. grads
contain the gradients of the loss (evaluated on the current mini-batch)
for the network parameters and the activations and preactivation.

9) NN.update function:
This function updates the network parameters W and b using the gradients
stored in the input grads. You will need to loop through all the elements
of the self. Weights dictionary to update them. This function's
grads input contains the gradients of the loss function
evaluated on the current mini-batch with the NN.backward function.

10) NN.one_hot function. This function  transforms
an array y of size batch_size containing values from 0 to
n_classes−1 into a matrix of size batch_size×n_classes containing
the one-hot encodings of the true labels y.

11) NN.loss function: This function takes as input a matrix
predictions of size batch_size×n_classes containing the probabilities
that the NN.forward function evaluate, and an array labels
of size batch_size×n_classes containing the one-hot encodings
of the true labels. The function should return the cross-entropy
loss.


## Language used
Python
