
# coding: utf-8

# Gradient Descent using numpy vector operations:

import numpy as np


# Load the data points:

def load_data():

    #Loading the dataset.
    dataset = np.genfromtxt('data_points.csv', delimiter=",")

    X = dataset[:, 0]
    Y = dataset[:, 1]

    return (X,Y)


# Initialise the parameters to zeros (ideally a random point to ease computation):

def initialise_param(dim):
    w = np.zeros(dim)
    b = 0
    return [w, b]


# Compute the activation.
# In this example we are not using an activation functiona as such but are returing the z value as Y_step

def activation(w, b, X):
    return np.add(np.multiply(w,X), b)


# Calculate the cost, gradients for a single Iteration:

def propagate_step(w,b,X,Y):

    #Number of Examples
    m = float(X.shape[0])

    #Activation (prediction for the step)
    A = activation(w, b, X)

    #Cost
    cost = (1/m) * (np.sum(np.square(Y-A)))

    #Gradients
    dw = (-2/m)*(np.dot(X,(Y-A).T))
    db = (-2/m)*np.sum(Y-A)

    grads = {'dw':dw,
            'db':db}

    return grads, cost


# Iterate to optimise the costs:

def optimise(w, b, X, Y, num_iterations=2000, learning_rate=0.0001):

    costs =[]

    for i in range(num_iterations):

        grads, cost = propagate_step(w, b, X, Y)

        if i%100==0:
            costs.append(cost)
            print "Iteration {itr} : Weights w: {w}, b: {b}, Cost: {cst}".format(itr =i, w=w, b=b, cst=cost)

        w = w - learning_rate*grads['dw']
        b = b - learning_rate*grads['db']

    params = {'w': w,
             'b':b}

    return params, grads, costs


# placeholder for the predict.

def prediction (w, b, X):

    return activation(w, b, X)

# Model() function to bind all the operations together:

def model():

    X, Y = load_data()

    w, b = initialise_param((1,))

    params, _, _ = optimise(w, b , X, Y, num_iterations=2000, learning_rate=0.0001)

    X_required = np.array([22.00,])

    return prediction(params['w'], params['b'], X_required)

# In[320]:

if __name__ == '__main__':

    print model()
