import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0.0  

    def update(self, theta, grad):
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return theta - self.lr * self.v


import math

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_history = []
        self.m = 0.0
        self.v = 0.0
        self.t = 0  
        

    def update(self, theta, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - (self.beta1 ** self.t))
        v_hat = self.v / (1 - (self.beta2 ** self.t))

        # Append the bias-corrected variance!
        self.v_history.append(np.copy(v_hat))

        theta_next = theta - (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        return theta_next
    


class AMSGrad:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_history = []
        self.m = 0.0
        self.v = 0.0
        self.v_hat_max = 0.0  

    def update(self, theta, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # np.maximum calculates the element-wise max across the arrays
        self.v_hat_max = np.maximum(self.v_hat_max, self.v)
        
        # Append AFTER the max is calculated, using the correct variable name
        self.v_history.append(np.copy(self.v_hat_max))

        theta_next = theta - (self.lr / (np.sqrt(self.v_hat_max) + self.epsilon)) * self.m
        return theta_next
    

