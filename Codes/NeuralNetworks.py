import numpy as np
import preprocess

def sigmoid(s, deriv=False):
        if (deriv == True):
            return s * (1.0 - s)
        return 1.0/(1+ np.exp(-s))

def LReLU( signal, deriv=False, leak = 0.01 ):
        if deriv:
            return np.clip(signal > 0, leakage, 1.0)
        else:
            output = np.copy( signal )
            output[ output < 0 ] *= leakage
            return output

class NeuralNetwork(object):
    def __init__(self, x, y):
        #parameters
        np.random.seed(0)
        self.input  = x
        self.size   = x.shape[1]
        self.w1     = 0.1 * np.random.rand(int(x.shape[1]), 89) 
        self.w2     = 0.1 * np.random.rand(89, 1)
        self.y      = y
        self.output = np.zeros(y.shape)
        
    def feedForward(self):
        #forward propogation through the network
        self.l1 = np.dot(self.input, self.w1)
        self.layer1 = sigmoid(self.l1) #Layer 1
        self.output = sigmoid(np.dot(self.layer1, self.w2)) #Output
        
        
    def backprop(self):
        self.output_error = self.y - self.output # error in output
        self.output_delta = self.output_error * sigmoid(self.output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.w2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * sigmoid(self.layer1, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.w1 += self.input.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.w2 += self.layer1.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights

        
    def train(self, epochs = 1000):
        self.loss = []
        for _ in range(epochs):
            self.feedForward()
            self.backprop()


    def pred(self, x):
        self.input = x
        self.feedForward()




