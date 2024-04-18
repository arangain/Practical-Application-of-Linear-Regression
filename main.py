# Learning how to implement linear regression with one variable to predict profits for a restaurant franchise!

# first, importing packages that I need
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
# remember you need to pip install numpy 
# numpy is a fundamental package for working with matrices in python
# matplotlib is a famous library to plot graphs in python
# utils.py contains helper functions for this assignment.


# Problem Statement:
"""
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet
- you would like to expand your business to cities that may give your restaurant higher profits
- the chain already has restaurants in various cities and you have data for profits and populations from the cities
- you also have data on cities that are candidates for a new restaurant.
    - for these cities, you have the city population 

Can you use the data to help you idenfity which cities may potentially give your businesses higher profits?


"""

#Dataset 
"""
Start by loading the data for the task
The load_data() function shown below loads the data into variables X_train and y_train
- x_train is the population of the city
- y_train is the profit of a restaurant in that city. A negative value indicates a loss
- both x_train and y_train are numpy arrays

"""

# load the dataset
x_train, y_train = load_data()


#View the variables
"""
Before starting on any task, its useful to get more familiar with your dataset
- A good place to start is to just print out each variable and see what it contains

The code below prints the variable x_train and the type of the varaible

"""

# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

#  x_train is a numpy array that contains decimal values that are all greater than 0,
# - these values represent the city population times 10,000

#print y_train
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

# y_train is a numpy array that contains decimal values, some negative and some positive
# - represents the restaurant's monthly profits, in units of $10,000


# check dimensions of variables
# - print the shape of x_train, y_train and see how many training examples we have in the dataset

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

# city population array x_train has 97 data points, and monthly average profits also has 97 data points
# so they are NumPy 1D arrays



#Visualise your data!!
"""
it is useful to understand data by visualising it
we'll use a scatter plot for this data since it only has two properties for plot (profit and population)
When you have more than two properties, you can still use a scatter plot to see the relationship between each pair of properties.

"""

# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()


# OUR GOAL: is to build a linear regression model to fit this data
# - with this model, you can input a new city's population, and have the model estimate your restaurant's potential monthly profits for that city


# Linear Regression
"""
In this practice problem, you will fit linear regression parameters (w,b) to your dataset
fwb(x) = wx+b, where the function maps x(city population) to y(restaurant's monthly profit for that city)

the choice of (w,b) that fits your data the best is one that has the smallest cost J(w,b)
to find values of (w,b) that gets the smallest possible cost J(w,b), you can use a method called gradient descent

The trained linear regression model can then take the input feature x (city population) and output a prediction 
for predicted monthly profit for a restaurant in that city


"""

# Compute Cost!!
"""
Implement a function to calculate J(w,b) so that you can check the progress of your gradient descent implementation

m is the number of training examples in the dataset for m x n matrix

"""

# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        total_cost = total_cost + (f_wb_i-y[i])**2
    total_cost = total_cost/(2*m)
    

    return total_cost


# Check implementation with following test code

# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests # this is a solution given by the instructor, won't have this normally
from public_tests import *
compute_cost_test(compute_cost)



# Gradient Descent

# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ### 

    for i in range(m):                             
        err = (np.dot(x[i], w) + b) - y[i]                        
        dj_dw = dj_dw + err * x[i]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw/m                                
    dj_db = dj_db/m        
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db



# check implementation of gradient descent computation
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient) # this test usually won't be given in real life problems


# Running gradient descent algorithm implemented above in our dataset
# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


#Learning parameters using batch gradient descent 
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)



# Now use parameters gained from gradient descent to plot the linear fit
# first, loop through the training examples and calculate the prediction for each example
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

    
# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

# we can use this to make predictions on profits; lets predict what the profit would be in areas of 35k and 70k people
# since the model takes in population of a city in 10,000's as input, 
# 35,000 people can be translated into np.array([3.5])

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

