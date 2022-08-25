# -*- coding: utf-8 -*-
#Import required libraries
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat

#Load the data from file
if exists("ex4data1.mat"):
    input_data = loadmat("ex4data1.mat")
else:
    print('File does not exist!')
    
x_data,y_data = input_data["X"],input_data["y"].ravel()

#Load the weights from file
if exists("ex4weights.mat"):
    weights = loadmat("ex4weights.mat")
else:
    print('File does not exist!')
    
theta_1,theta_2 = weights["Theta1"],weights["Theta2"]

plt.figure()
rng = np.random.RandomState(0)
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_data[rng.randint(x_data.shape[0])].reshape((20, 20)).T,cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
plt.show()

def Sigmoid(z):
    val = 1/(1+np.exp(-z))
    return val

def SigmoidGradient(z):
    grad_val = np.multiply(Sigmoid(z),1-Sigmoid(z))
    return grad_val

def RandomInitialization(Lin,Lout):
    #epsilon_initial = np.sqrt(6)/(np.sqrt(Lin+Lout))
    epsilon_initial = 0.12
    rng = np.random.RandomState(0)
    weights_random = rng.rand(Lout,Lin+1)* 2 * epsilon_initial - epsilon_initial
    return weights_random

def costFunction(nn_params,input_layers,hidden_layers,num_labels,x,y,lambda_val):
    theta_1_weights = nn_params[:hidden_layers*(input_layers+1)]
    theta_1_weights = theta_1_weights.reshape((hidden_layers,input_layers+1))
    theta_2_weights = nn_params[hidden_layers*(input_layers+1):]
    theta_2_weights = theta_2_weights.reshape((num_labels,hidden_layers+1))
    
    x_training_dataset = np.hstack((np.ones((x.shape[0],1)),x))
    a2 = Sigmoid(np.dot(x_training_dataset, theta_1_weights.T))
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = Sigmoid(np.dot(a2,theta_2_weights.T))
    y_training_dataset = np.zeros((x.shape[0],num_labels))
    y_training_dataset[np.arange(x.shape[0]),y-1] = 1
    
    J = -np.multiply(y_training_dataset,np.log(a3))-np.multiply(1-y_training_dataset,np.log(1-a3))
    J = np.sum(J)/x.shape[0]
    regularization = np.sum(np.power(theta_1_weights[:,1:],2)) + np.sum(np.power(theta_2_weights[:,1:],2))
    J = J+((regularization*lambda_val)/(2*x.shape[0]))
    return J

def gradientCalculation(nn_params,input_layers,hidden_layers,num_labels,x,y,lambda_val):
    theta_1_weights = nn_params[:hidden_layers*(input_layers+1)]
    theta_1_weights = theta_1_weights.reshape((hidden_layers,input_layers+1))
    theta_2_weights = nn_params[hidden_layers*(input_layers+1):]
    theta_2_weights = theta_2_weights.reshape((num_labels,hidden_layers+1))
    
    x_training_dataset = np.hstack((np.ones((x.shape[0],1)),x))
    a2 = Sigmoid(np.dot(x_training_dataset, theta_1_weights.T))
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = Sigmoid(np.dot(a2,theta_2_weights.T))
    y_training_dataset = np.zeros((x.shape[0],num_labels))
    y_training_dataset[np.arange(x.shape[0]),y-1] = 1
    
    theta_1_gradient = np.zeros(theta_1_weights.shape)
    theta_2_gradient = np.zeros(theta_2_weights.shape)
    delta_3 = a3 - y_training_dataset
    delta_2 = np.multiply(np.dot(delta_3,theta_2_weights),np.multiply(a2,1-a2))
    
    for i in range(x.shape[0]):
        theta_1_gradient = theta_1_gradient + np.dot(delta_2[i,1:][:,np.newaxis],np.atleast_2d(x_training_dataset[i]))
        theta_2_gradient = theta_2_gradient + np.dot(delta_3[i][:,np.newaxis], np.atleast_2d(a2[i]))
        
    
    theta_2_gradient = theta_2_gradient/x.shape[0]
    theta_1_gradient = theta_1_gradient/x.shape[0]
    
    theta_1_gradient[:,1:] = theta_1_gradient[:,1:] + ((lambda_val*theta_1_gradient[:,1:])/x.shape[0])
    theta_2_gradient[:,1:] = theta_2_gradient[:,1:] + ((lambda_val*theta_2_gradient[:,1:])/x.shape[0])
    
    gradient_value = np.hstack((theta_1_gradient.ravel(),theta_2_gradient.ravel()))
    return gradient_value

def gradient(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lam):
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 = Theta1.reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape((num_labels, hidden_layer_size + 1))

    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    A2 = Sigmoid(np.dot(X_train, Theta1.T))
    A2 = np.hstack((np.ones((A2.shape[0], 1)), A2))
    A3 = Sigmoid(np.dot(A2, Theta2.T))
    y_train = np.zeros((X.shape[0], num_labels))
    y_train[np.arange(X.shape[0]), y - 1] = 1

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    delta3 = A3 - y_train
    delta2 = np.multiply(np.dot(delta3, Theta2), np.multiply(A2, 1 - A2))
    for i in range(X.shape[0]):
        Theta1_grad += np.dot(delta2[i, 1:][:, np.newaxis], np.atleast_2d(X_train[i]))
        Theta2_grad += np.dot(delta3[i][:, np.newaxis], np.atleast_2d(A2[i]))
    Theta1_grad /= X.shape[0]
    Theta2_grad /= X.shape[0]

    Theta1_grad[:, 1:] += lam * Theta1[:, 1:] / X.shape[0]
    Theta2_grad[:, 1:] += lam * Theta2[:, 1:] / X.shape[0]
    grad = np.hstack((Theta1_grad.ravel(), Theta2_grad.ravel()))
    return grad


input_layers = 400
hidden_layers = 50
num_labels = 10

lambda_val = 1

theta_1_initial = RandomInitialization(input_layers, hidden_layers)
theta_2_initial = RandomInitialization(hidden_layers, num_labels)

nn_params = np.hstack((theta_1_initial.ravel(), theta_2_initial.ravel()))
grad_val= gradientCalculation(nn_params, input_layers, hidden_layers, num_labels, x_data, y_data, lambda_val)
residual = optimize.minimize(fun=costFunction, x0=nn_params,args =(input_layers,hidden_layers,num_labels,x_data,y_data,lambda_val),method="CG",jac=gradientCalculation,options={"maxiter": 50})

theta_1_optimized = residual.x[:hidden_layers*(input_layers+1)]
theta_1_optimized = theta_1_optimized.reshape((hidden_layers,input_layers+1))
theta_2_optimized = residual.x[hidden_layers*(input_layers+1):]
theta_2_optimized = theta_2_optimized.reshape((num_labels,hidden_layers+1))

plt.figure()
for i in range(hidden_layers):
    plt.subplot(10, 5, i + 1)
    plt.imshow(theta_1_optimized[i,1:].reshape((20, 20)).T,cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
plt.show()
