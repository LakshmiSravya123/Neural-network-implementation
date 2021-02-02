import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights={}
        
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]

        for layer_n in range(1, self.n_hidden + 2):
            
            sqrt=np.sqrt(all_dims[layer_n-1])
            self.weights[f"W{layer_n}"] = np.random.uniform(-1/sqrt, 1/sqrt, [all_dims[layer_n-1],all_dims[layer_n]])
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
           
                
           

    def relu(self, x, grad=False):
        if grad==False:
            return np.maximum(0,x)
        else:

            return np.maximum(0,x>0)
           

    def sigmoid(self, x, grad=False):
        if grad == False:
            return (1 / (1 + np.exp(-x)))
        else:
            s = 1 / (1 + np.exp(-x))
            
            return (s*(1-s))
          

    def tanh(self, x, grad=False):
        if grad == False:
           return np.tanh(x)
        else:
           a= np.tanh(x)
           return (1 - a**2)
           
    

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
           return self.relu(x,grad)

        elif self.activation_str == "sigmoid":
           return self.sigmoid(x,grad)
           
        elif self.activation_str == "tanh":
           return self.tanh(x,grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        e = np.exp(x - np.max(x))
        
        if e.ndim == 1:
           return e / np.sum(e, axis=0)
        else:  
           return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
       
        #e_x = np.exp(x - np.max(x))
        #return e_x / e_x.sum(axis=0)
        

    def forward(self, x):
        cache = {"Z0": x}
        
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
       
        for layer_n in range(1, self.n_hidden + 2):
            
            if layer_n == (self.n_hidden + 1):
                cache[f"A{layer_n}"]=np.dot(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"])+self.weights[f"b{layer_n}"]
                cache[f"Z{layer_n}"]=self.softmax(cache[f"A{layer_n}"])
            else:
                cache[f"A{layer_n}"]=np.dot(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"])+self.weights[f"b{layer_n}"]
                cache[f"Z{layer_n}"]=self.activation(cache[f"A{layer_n}"])
                
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}       
        m=self.batch_size
        
        grads[f"dA{self.n_hidden+1}"] = output-labels
        grads[f"db{self.n_hidden+1}"] = (1 / m) * np.sum(grads[f"dA{self.n_hidden+1}"], axis=0,keepdims=True)
        grads[f"dW{self.n_hidden+1}"] = (1 / m) * np.dot( cache[f"Z{self.n_hidden+1-1}"].T,grads[f"dA{self.n_hidden+1}"])
        grads[f"dZ{self.n_hidden+1-1}"]=  np.dot(grads[f"dA{self.n_hidden+1}"],self.weights[f"W{self.n_hidden+1}"].T)
        for layer in reversed(range(1,self.n_hidden + 1)):

                grads[f"dA{layer}"]=   grads[f"dZ{layer}"]*self.activation(cache[f"A{layer}"],grad=True)    
                grads[f"db{layer}"] = (1 / m) * np.sum(grads[f"dA{layer}"], axis=0,keepdims=True)             
                grads[f"dW{layer}"] = (1 / m) * np.dot( cache[f"Z{layer-1}"].T,grads[f"dA{layer}"])
                if(layer>1) :
                   grads[f"dZ{layer-1}"]=  np.dot(grads[f"dA{layer}"],self.weights[f"W{layer}"].T)
                
              
              
        
        
        return grads
    
    def update(self, grads):
        
        for layer in range(1, self.n_hidden + 2):
          
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]        
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]

        return self.weights

    def one_hot(self, y):
        targets = np.array(y)
        ohm = np.zeros((targets.shape[0], self.n_classes),dtype=int)
        ohm[np.arange(targets.shape[0]), targets] = 1
        return ohm

    def loss(self, prediction, labels):
        
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        #logprobs = np.multiply(np.log(prediction), labels) + np.multiply((1 - labels), np.log(1 - prediction))
        yl=np.multiply(prediction,labels)
        yl=yl[yl!=0]
        yl=-np.log(yl)
        cost=np.mean(yl)
        cost = np.squeeze(cost) 
        return cost

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache=self.forward(minibatchX)
                grad=self.backward(cache,minibatchY)
                self.update(grad)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss,test_accuracy
    

