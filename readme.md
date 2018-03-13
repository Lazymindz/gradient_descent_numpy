
### Gradient Descent example  (Linear Regression) using Numpy Vector Operations

##### Below is a simple framework for building a basic Neural Network:

Define the model features

Prepare the Dataset

Initialize the model's parameters
Loop:
Calculate current loss (forward propagation)
Calculate current gradient (backward propagation)
Update parameters (gradient descent)

Use the parameters for output predictions


```python
import numpy as np
```


```python
def load_data():
    
    #Loading the dataset.
    dataset = np.genfromtxt('data_points.csv', delimiter=",")
    
    X = dataset[:, 0]
    Y = dataset[:, 1]
    
    return (X,Y)
```


```python
def initialise_param(dim):
    w = np.zeros(dim)
    b = 0
    return [w, b]
```


```python
def activation(w, b, X):
    return np.add(np.multiply(w,X), b)
```

References to understand the Cost, Gradients:

http://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html

https://github.com/mattnedrich/GradientDescentExample

https://www.youtube.com/watch?v=XdM6ER7zTLk&index=2&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3


```python
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
```


```python
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
```


```python
def prediction (w, b, X):
    
    return activation(w, b, X)
```


```python
if __name__ == '__main__':
    
    X, Y = load_data()
        
    w, b = initialise_param((1,))
    
    params, _, _ = optimise(w, b , X, Y, num_iterations=2000, learning_rate=0.0001)
    
    X_required = np.array([22.00,])
    
    print prediction(params['w'], params['b'], X_required)
```

    Iteration 0 : Weights w: [0.], b: 0, Cost: 5565.10783449
    Iteration 100 : Weights w: [1.47880272], b: 0.0350749705952, Cost: 112.647056628
    Iteration 200 : Weights w: [1.47868474], b: 0.0410776713033, Cost: 112.643452008
    Iteration 300 : Weights w: [1.47856684], b: 0.0470758430193, Cost: 112.639852825
    Iteration 400 : Weights w: [1.47844904], b: 0.0530694891604, Cost: 112.636259071
    Iteration 500 : Weights w: [1.47833133], b: 0.0590586131412, Cost: 112.632670738
    Iteration 600 : Weights w: [1.4782137], b: 0.0650432183735, Cost: 112.629087818
    Iteration 700 : Weights w: [1.47809616], b: 0.0710233082667, Cost: 112.625510303
    Iteration 800 : Weights w: [1.47797872], b: 0.0769988862276, Cost: 112.621938183
    Iteration 900 : Weights w: [1.47786136], b: 0.0829699556604, Cost: 112.618371452
    Iteration 1000 : Weights w: [1.47774409], b: 0.0889365199668, Cost: 112.614810101
    Iteration 1100 : Weights w: [1.4776269], b: 0.0948985825459, Cost: 112.611254122
    Iteration 1200 : Weights w: [1.47750981], b: 0.100856146794, Cost: 112.607703507
    Iteration 1300 : Weights w: [1.4773928], b: 0.106809216105, Cost: 112.604158248
    Iteration 1400 : Weights w: [1.47727589], b: 0.112757793871, Cost: 112.600618336
    Iteration 1500 : Weights w: [1.47715906], b: 0.11870188348, Cost: 112.597083765
    Iteration 1600 : Weights w: [1.47704231], b: 0.124641488319, Cost: 112.593554524
    Iteration 1700 : Weights w: [1.47692566], b: 0.130576611771, Cost: 112.590030608
    Iteration 1800 : Weights w: [1.4768091], b: 0.136507257218, Cost: 112.586512006
    Iteration 1900 : Weights w: [1.47669262], b: 0.142433428038, Cost: 112.582998713
    [32.6330322]

