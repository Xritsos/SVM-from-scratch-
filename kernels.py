import numpy as np


def poly(coef=None, degree=None, gamma=None):
    
    if coef == None:
        coef = 1.0
            
    if degree == None:
        degree = 1
            
        
    def poly(x1, x2):
        
        dot = np.dot(x1, x2.T)
        
        result = np.power(dot + coef, degree)
        
        return result
    
    return poly


def rbf(gamma, **kwargs):
    
    if gamma == None:
        gamma = 0.5
        
    if gamma <= 0.0:
        raise ValueError("gamma should be positive !")
    
    def rbf(x1, x2):
        
        n = x1.shape[0]
        m = x2.shape[0]
        
        xx1 = np.dot(np.sum(np.power(x1, 2), 1).reshape(n, 1), np.ones((1, m)))
        xx2 = np.dot(np.sum(np.power(x2, 2), 1).reshape(m, 1), np.ones((1, n))) 
        
        result = np.exp(-(xx1 + xx2.T - 2 * np.dot(x1, x2.T)) * gamma) 
        
        return result
    
    return rbf


def linear(**kwargs):
    
    def linear(x1, x2):
        
        result = np.dot(x1, x2.T)
    
        return result
    
    return linear


def sigmoid(coef=None, gamma=None, **kwargs):
    
    if coef == None:
        coef = 1.0
    
    if gamma == None:
        gamma = 1.0
        
    if gamma <= 0.0:
        raise ValueError("gamma should be positive !")
        
    def sigmoid(x1, x2):

        result = np.dot(x1, x2.T)
        result *= gamma 
        result += coef
        
        np.tanh(result, result) 
        
        return result
    
    return sigmoid
