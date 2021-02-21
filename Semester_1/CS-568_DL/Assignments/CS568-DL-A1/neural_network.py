import numpy as np
import matplotlib.pyplot as plt
from f_utils import *
import copy
from f_check_gradient import *
from sklearn.utils import shuffle


class NeuralNetwork():
  
    def __init__(self, num_neurons, activations_func, learning_rate, num_epochs, mini_batch_size):     
        self.num_neurons = num_neurons
        self.activations_func = activations_func
        self.learning_rate = learning_rate
        self.num_iterations = num_epochs
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(self.num_neurons) - 1
        self.parameters = dict()
        self.net = dict()
        self.grads = dict()

        
    def initialize_parameters(self):
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):
            if self.activations_func[l] == 'relu': #xavier intialization method
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l-1]/2.)
            else:                
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l - 1])
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("weights and biases initialization",self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)

          
    def fprop(self, batch_input): 
        ## add code here 
        a = self.parameters["W1"]*batch_input + self.parameters["b1"]
        z  = sigmoid(a)
        y = np.dot(self.parameters["W2"],z) + self.parameters["b2"]
        self.net["y"] = y
        self.net["z"] = z
        self.net["a"] = a
        self.net["input"] = batch_input
                   
    def calculate_loss(self, batch_target):       
        ## add code here
        loss = 0.5*np.sum((self.net["y"] - batch_target)**2)
        return loss
        
    def update_parameters(self,epoch):
        ## add code here
        for l in range(1, self.num_layers + 1):
            for x in range(len(self.grads['dW%s'%l])):
                t = self.learning_rate*self.grads['dW%s'%l][x]
                if l==1:
                    t = t.T
                self.parameters['W%s'%l]-=t
    
    def bprop(self, batch_target):
        ## add code here 
        self.grads["dW3"] = (self.net["y"] - batch_target)
        self.grads["dW2"] = list()
        self.grads["dW1"] = list()
        for x in range(len(self.grads["dW3"][0])):
            self.grads["dW2"].append(self.grads["dW3"][0][x]*self.parameters["W2"])

        for x in range(len(self.grads["dW2"])):
            self.grads["dW1"].append(self.grads["dW2"][x]*self.net["input"][0][x]) 
        self.grads["dW2"] = np.array(self.grads["dW2"])
        self.grads["dW1"] = np.array(self.grads["dW1"])
        

    def plot_loss(self,loss,val_loss):        
        plt.figure()
        fig = plt.gcf()
        plt.plot(loss, linewidth=3, label="train")
        plt.plot(val_loss, linewidth=3, label="val")
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('learning rate =%s, hidden layers=%s' % (self.learning_rate, self.num_layers-1))
        plt.grid()
        plt.legend()
        plt.show()
        fig.savefig('plot_loss.png')
        
    
    def plot_gradients(self):
        avg_l_g = []
        grad = copy.deepcopy(self.grads)
        for l in range(1, self.num_layers+1):
#             print("layer %s"%l)
             weights_grad = grad['dW%s' % l]  
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d])
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
        layers = ['layer %s'%l for l in range(self.num_layers+1)]
        weights_grad_mag = avg_l_g
        fig = plt.gcf()
        plt.xticks(range(len(layers)), layers)
        plt.xlabel('layers')
        plt.ylabel('average gradients magnitude')
        plt.title('')
        plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
        plt.show() 
        fig.savefig('plot_gradients.png')
    

    def train(self, train_x, train_y, val_x, val_y):
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        self.initialize_parameters()        
        train_loss = []
        val_loss = []  
        num_samples = train_y.shape[1]       
        check_grad = True
        grad_ok = 0
        

        for i in range(0, self.num_iterations):
            for idx in range(0, num_samples, self.mini_batch_size):
                minibatch_input =  train_x[:, idx:idx + self.mini_batch_size]
                minibatch_target =  train_y[:, idx:idx + self.mini_batch_size]
                
                if check_grad == True:
                    grad_ok = check_gradients(self, minibatch_input, minibatch_target)               
                    if grad_ok == 0:                           
                        print("gradients are not ok!\n")                           
                            
                   
                if grad_ok == 1:
                    check_grad = False
                    self.fprop(minibatch_input)
                    loss = self.calculate_loss(minibatch_target)
                    self.bprop(minibatch_target)           
                    self.update_parameters(i)
                   
            train_loss.append(loss) 
            self.fprop(val_x)
            va_loss = self.calculate_loss(val_y)
            val_loss.append(va_loss)              
            print("Epoch %i: training loss %f, validation loss %f" % (i, loss,va_loss))
        self.plot_loss(train_loss,val_loss)      
        self.plot_gradients()
       
    
    def test(self,test_x,test_t):
        
        self.fprop(test_x)
        loss = self.calculate_loss(test_t)
        return loss, self.net["y"]
