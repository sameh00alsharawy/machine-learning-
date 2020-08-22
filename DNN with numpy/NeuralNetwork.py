import numpy as np 

"""
this file contain 2 classes
N_N   : classification
N_N_L : Regression
"""


class N_N(object):
    def __init__(self, n_h=4, lr=0.001):
        """
        n_h .. number of neurons in the hidden layer
        lr  .. the learning rate
        """
        self.n_h = n_h 
        self.lr = lr
    
    
    def find_dim(self, X, y):
        """
        Arguments :
            
            X .. the input matrix, each row correspond to on feature, each column
                 corresponds to one instance (#features, #examples(m))
            y .. the class vector of shape (sizeof output layer, m) 
        
        returns:
        
            n_x .. number of features
            n_h .. number of neurons in the hidden layer
            n_y .. number of output neurons
        
        """
        n_x = X.shape[0]
        n_h = self.n_h
        n_y = y.shape[0]
        
        return n_x, n_h, n_y
        
    def initialize (self, n_x, n_h, n_y):
        """
        Arguments :
            
            n_x .. number of features
            n_h .. number of neurons in the hidden layer
            n_y .. number of output neurons
        Return :
            parameters .. a dictionary containing W1, b1, W2, b2
            W1  .. the weights from input to hidden layer, a matrix of shape (n_h, n_x)
            b1  .. the bias of the hidden layer, a matrix of shape(n_h, 1)
            W2  .. the weights from the hidden layer to the output layer, (n_y, n_h)
            b2  .. the bias of the output layer (n_y, 1)
        """
        W1 = np.random.randn(n_h, n_x)*0.0001
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.0001
        b2 = np.zeros((n_y, 1))
        parameters= {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
        
        return parameters

    def sigmoid(self, A):
        """
        Arguments:
            A .. A matrix
        returns:
            B .. sigmoid of A 1/(1+e^-A)
        """
        B = 1/(1+(np.exp(A)))
        return B
    
    def forward_propagation (self, X, parameters):
        """
        Arguments:
            X .. input features for size m dataset, a matrix of size (nx, m)
            parameters .. a dictionary containing W1, b1, W2, b2
        returns:
            A2 .. The sigmoid output of the second activation
            cache .. a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        
        cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
        return A2, cache
    
    def compute_cost(self, A2, y):
        """
        Arguments:
            A2 .. the sigmoid output of the second activation
            y  .. the true classes 
            
        return :
            cost .. cost function of logistic regression over m examples
        """
        m = y.shape[1]
        n_y = y.shape[0]
        cost = np.sum(-y*np.log(A2)-(1-y)*np.log(1-A2), axis =1, keepdims=True )/m
        cost = np.sum(cost)/n_y
        return cost
    
    def backward_propagation(self, parameters, cache, X, y):
        """
        Arguments:
            parameters .. a dictionary containing W1, b1, W2, b2
            cache .. a dictionary containing "Z1", "A1", "Z2" and "A2"
            X .. input features for size m dataset, a matrix of size (nx, m)
            y .. the class vector of shape (sizeof output layer, m)
        returns:
            grads .. a dictionary containing dW1, db1, dW2, db2
        """
        m = X.shape[1]
        
        W2 = parameters['W2']
        
        A1 = cache['A1']
        A2 = cache['A2']      
        
        dZ2 = A2 -y
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis = 1 , keepdims=True)/m
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m
        
        grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
 
        return grads

    def update_parameters(self,parameters, grads):
        """
        Arguments:
        parameters .. python dictionary containing your parameters 
        grads .. python dictionary containing your gradients 
        
        Returns:
        parameters .. python dictionary containing your updated parameters 
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        W1 = W1-self.lr*dW1 
        b1 = b1-self.lr*db1
        W2 = W2-self.lr*dW2
        b2 = b2-self.lr*db2
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
        
        
    def train(self, X, y, n_iteration=1500, print_cost=0):
        """
        Arguments :
            X .. input features for size m dataset, a matrix of size (nx, m)
            y .. the class vector of shape (sizeof output layer, m)
        Returns:
            parameters .. python dictionary containing your updated parameters    
            costs .. a list containing the cost after each iteration
        """
        costs = []
        n_x, n_h, n_y = self.find_dim( X, y)
        parameters = self.initialize(n_x, n_h, n_y)
        
        for i in range(n_iteration):
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, y)
            costs.append(cost)
            grads = self.backward_propagation(parameters, cache, X, y)
            parameters = self.update_parameters(parameters, grads)
            if print_cost and i % print_cost == 0 :
                print(f'the cost at the loop number {i} is ' , cost)
        return parameters, costs
    
    def predict (self, parameters, X):
        """
        Arguments:
            parameters .. python dictionary containing your updated parameters
            X .. input data
        return 
            yhat .. prediction
        """
        A2, cache = self.forward_propagation(X, parameters)
        return A2
    
# cLass N_N_L is a regression network


class N_N_L(object):
    def __init__(self, n_h=4, lr=0.001):
        """
        n_h .. number of neurons in the hidden layer
        lr  .. the learning rate
        """
        self.n_h = n_h 
        self.lr = lr
    
    
    def find_dim(self, X, y):
        """
        Arguments :
            
            X .. the input matrix, each row correspond to on feature, each column
                 corresponds to one instance (#features, #examples(m))
            y .. the class vector of shape (sizeof output layer, m) 
        
        returns:
        
            n_x .. number of features
            n_h .. number of neurons in the hidden layer
            n_y .. number of output neurons
        
        """
        n_x = X.shape[0]
        n_h = self.n_h
        n_y = y.shape[0]
        
        return n_x, n_h, n_y
        
    def initialize (self, n_x, n_h, n_y):
        """
        Arguments :
            
            n_x .. number of features
            n_h .. number of neurons in the hidden layer
            n_y .. number of output neurons
        Return :
            parameters .. a dictionary containing W1, b1, W2, b2
            W1  .. the weights from input to hidden layer, a matrix of shape (n_h, n_x)
            b1  .. the bias of the hidden layer, a matrix of shape(n_h, 1)
            W2  .. the weights from the hidden layer to the output layer, (n_y, n_h)
            b2  .. the bias of the output layer (n_y, 1)
        """
        W1 = np.random.randn(n_h, n_x)* np.sqrt(2.0/n_x)
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)* np.sqrt(2.0/n_h)
        b2 = np.zeros((n_y, 1))
        parameters= {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
        
        return parameters

    def sigmoid(self, A):
        """
        Arguments:
            A .. A matrix
        returns:
            B .. sigmoid of A 1/(1+e^-A)
        """
        B = 1/(1+(np.exp(A)))
        return B
    
    def forward_propagation (self, X, parameters):
        """
        Arguments:
            X .. input features for size m dataset, a matrix of size (nx, m)
            parameters .. a dictionary containing W1, b1, W2, b2
        returns:
            A2 .. The sigmoid output of the second activation
            cache .. a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = Z2
        
        cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
        return A2, cache
    
    def compute_cost(self, A2, y):
        """
        Arguments:
            A2 .. the sigmoid output of the second activation
            y  .. the true classes 
            
        return :
            cost .. cost function of logistic regression over m examples
        """
        m = y.shape[1]
        n_y = y.shape[0]
        cost = (1/2*m)*(np.sum((A2-y)**2))
        cost = np.sum(cost)/n_y
        return cost
    
    def backward_propagation(self, parameters, cache, X, y):
        """
        Arguments:
            parameters .. a dictionary containing W1, b1, W2, b2
            cache .. a dictionary containing "Z1", "A1", "Z2" and "A2"
            X .. input features for size m dataset, a matrix of size (nx, m)
            y .. the class vector of shape (sizeof output layer, m)
        returns:
            grads .. a dictionary containing dW1, db1, dW2, db2
        """
        m = X.shape[1]
        
        W2 = parameters['W2']
        
        A1 = cache['A1']
        A2 = cache['A2']      
        
        dZ2 = (1/m)*(A2-y)
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis = 1 , keepdims=True)/m
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m
        
        grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
 
        return grads

    def update_parameters(self,parameters, grads):
        """
        Arguments:
        parameters .. python dictionary containing your parameters 
        grads .. python dictionary containing your gradients 
        
        Returns:
        parameters .. python dictionary containing your updated parameters 
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        W1 = W1-self.lr*dW1 
        b1 = b1-self.lr*db1
        W2 = W2-self.lr*dW2
        b2 = b2-self.lr*db2
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
        
        
    def train(self, X, y, n_iteration=1500, print_cost=0):
        """
        Arguments :
            X .. input features for size m dataset, a matrix of size (nx, m)
            y .. the class vector of shape (sizeof output layer, m)
        Returns:
            parameters .. python dictionary containing your updated parameters    
            costs .. a list containing the cost after each iteration
        """
        costs = []
        n_x, n_h, n_y = self.find_dim( X, y)
        parameters = self.initialize(n_x, n_h, n_y)
        
        for i in range(n_iteration):
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, y)
            costs.append(cost)
            grads = self.backward_propagation(parameters, cache, X, y)
            parameters = self.update_parameters(parameters, grads)
            if print_cost and i % print_cost == 0 :
                print(f'the cost at the loop number {i} is ' , cost)
                
        return parameters, costs
    
    def predict (self, parameters, X):
        """
        Arguments:
            parameters .. python dictionary containing your updated parameters
            X .. input data
        return 
            yhat .. prediction
        """
        A2, cache = self.forward_propagation(X, parameters)
        return A2        
        
        
        
        
        
        
        
        
        
        
        
        