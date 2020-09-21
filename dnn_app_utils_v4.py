import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

from dnn_activation_functions import *
from dnn_initialization import *
from dnn_forward_prop import *
from dnn_backward_prop import *

def compute_cost(AL, Y, parameters, lambd=0):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    lambd -- regularization hyperparameter, scalar

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    n_layers = len(parameters) // 2   # number of layers in the neural network

    cross_entropy_cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))  # This gives you the cross-entropy part of the cost
    
    # L2 regularization cost
    L2_regularization_cost = 0
    for l in range(1, n_layers):
        L2_regularization_cost +=  np.sum(np.square(parameters['W' + str(l)])) 
    L2_regularization_cost = 1/m * lambd/2 * L2_regularization_cost
    
    cost = cross_entropy_cost + L2_regularization_cost

    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 1001, lambd=0, keep_prob=1, \
                    print_cost=True, print_cost_plot=True, parameters=None, initialization="he"):                                #lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    lambd -- regularization hyperparameter, scalar
    keep_prob -- probability of keeping a neuron active during drop-out, scalar 
    print_cost -- if True, it prints the cost every 100 steps
    print_cost_plot -- if True, it prints the plot of the cost function at each iteration 
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tic = time.time()

    np.random.seed(1)
    n_layers = len(layers_dims)        # number of layers in the neural network
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    if parameters == None:
        if initialization == "zeros":
            parameters = initialize_parameters_zeros(layers_dims)
        elif initialization == "random":
            parameters = initialize_parameters_random(layers_dims)
        elif initialization == "he":
            parameters = initialize_parameters_he(layers_dims)
            print("Initialization He")
        else: 
            parameters = initialize_parameters_deep(layers_dims)
            print("Initialization deep")
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, keep_prob)
        
        # Compute cost.
        cost = compute_cost(AL, Y, parameters, lambd)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambd, keep_prob)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            

    toc = time.time()
    if print_cost == True:
        print("The total training time is", toc-tic, "seconds")

    # plot the cost
    if print_cost_plot == True:
        plt.figure(figsize=(9,5))
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
