import numpy as np
import numexpr as ne



def poly(coef=None, degree=None, gamma=None):
    
    if coef == None:
        coef = 1.0
            
    if degree == None:
        degree = 1
            
    if gamma == None:
        gamma = 1.0
            
    if gamma <= 0:
        raise ValueError("gamma should be positive !")
        
    def f(x1, x2):
        dot = np.dot(x1, x2.T)
        dot = gamma * dot
        
        result = np.power(dot + coef, degree)
        
        return result
    
    return f

# def rbf(x, z, gamma):
#     n = x.shape[0]
#     m = z.shape[0]
#     xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
#     zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))   
      
#     return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) * gamma)

def rbf(gamma, **kwargs):
    
    if gamma == None:
        gamma = 0.5
        
    if gamma <= 0:
        raise ValueError("gamma should be positive !")
    
    def f(x1, x2):
        n = x1.shape[0]
        m = x2.shape[0]
        xx1 = np.dot(np.sum(np.power(x1, 2), 1).reshape(n, 1), np.ones((1, m)))
        xx2 = np.dot(np.sum(np.power(x2, 2), 1).reshape(m, 1), np.ones((1, n))) 
        
        return np.exp(-(xx1 + xx2.T - 2 * np.dot(x1, x2.T)) * gamma) 
    
    return f


def linear(**kwargs):
    
    def f(x1, x2):
        
        result = np.dot(x1, x2.T)
    
        return result
    
    return f


def sigmoid(coef=None, gamma=None):
    
    if coef == None:
        coef = 1.0
    
    if gamma == None:
        gamma = 1.0
        
    def f(x1, x2):
    
        dot = np.dot(x1.T, x2)
        dot = gamma * dot + coef
    
        result = np.tanh(dot)
    
        return result
    
    return f
