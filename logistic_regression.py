import numpy as np 

class Logistic_Regression:
    def __init__(self, max_iters, gamma, train_set, test_set):
        self.max_iters = max_iters
        self.gamma = gamma
        self.x_train, self.y_train = train_set
        self.x_test, self.y_test = test_set
        self.initial_weights = np.zeros((np.shape(self.x_train)[1]))
        self.weights = None
        self.train_loss = None
        self.test_loss = None
        
    def replace(self, data, value, new_value):
        data[data == value] = new_value
        return data
        
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def calculate_grad(self, y, tx, w):
        return (1 / y.shape[0] * (self.sigmoid(tx.dot(w)) - y).T.dot(tx)).T
    
    def calculate_loss(self, y, tx, w):
        return np.sum(-(1 / y.shape[0]) * (y.T.dot(np.log(self.sigmoid(tx.dot(w))))+ (1 - y).T.dot(np.log(1 - self.sigmoid(tx.dot(w))))))
    
    def learning_by_gradient_descent(self, y, tx, w, gamma):
        loss = self.calculate_loss(y, tx, w)
        w = w - gamma * (self.calculate_grad(y, tx, w))
        return loss, w
    
    def predict(self, result):
        return np.array([1 if x > 0.5 else 0 for x in result])
    
    def logistic_regression(self):
        self.y_train = self.replace(self.y_train, -1, 0)
        self.weights = self.initial_weights
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        for n_iter in range(self.max_iters):
            self.train_loss, self.weights = self.learning_by_gradient_descent(self.y_train, self.x_train, self.weights, self.gamma)
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        print(self.train_loss)
        predictions = self.sigmoid(self.x_train.dot(self.weights))
        train_predictions = self.predict(predictions)
        print(predictions)
        correct = np.sum(train_predictions == self.y_train)
        train_accuracy = correct / np.shape(self.x_train)[0]
        print(train_accuracy)
    
    def test(self):
        output = self.sigmoid(self.x_test.dot(self.weights))
        self.test_predictions = self.predict(output)
        correct = np.sum(self.test_predictions == self.y_test)
        test_accuracy = correct / np.shape(self.x_test)[0]
        print(test_accuracy)
