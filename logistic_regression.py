import numpy as np 
import csv
import pickle

class Logistic_Regression:
    def __init__(self, max_iters, gamma, train_set, test_set, id):
        self.max_iters = max_iters
        self.gamma = gamma
        self.x_train, self.y_train = train_set
        self.x_test, self.y_test = test_set
        self.id = id
        self.initial_weights = np.zeros((np.shape(self.x_train)[1]))
        self.weights = None
        self.train_loss = None
        self.test_loss = None
        self.y_values = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.test_predictions = None
        self.prepare_data()
        
    def replace(self, data, value, new_value):
        data[data == value] = new_value
        return data
    
    def prepare_data(self):
        self.y_values = np.unique(self.y_train)
        if abs(self.y_values[0] - self.y_values[1]) > 1:
            self.y_train = self.replace(self.y_train, self.y_values[0], 0)
            self.y_train = self.replace(self.y_train, self.y_values[1], 1)
            self.y_test = self.replace(self.y_test, self.y_values[0], 0)
            self.y_test = self.replace(self.y_test, self.y_values[1], 1)
        
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
    
    def train(self):
        self.weights = self.initial_weights
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        for n_iter in range(self.max_iters):
            self.train_loss, self.weights = self.learning_by_gradient_descent(self.y_train, self.x_train, self.weights, self.gamma)
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        predictions = self.sigmoid(self.x_train.dot(self.weights))
        train_predictions = self.predict(predictions)
        correct = np.sum(train_predictions == self.y_train)
        self.train_accuracy = correct / self.x_train.shape[0]
    
    def test(self):
        self.test_loss = self.calculate_loss(self.y_test, self.x_test, self.weights)
        predictions = self.sigmoid(self.x_test.dot(self.weights))
        self.test_predictions = self.predict(predictions)
        correct = np.sum(self.test_predictions == self.y_test)
        self.test_accuracy = correct / self.x_test.shape[0]
        print(self.test_accuracy)
          
    def submission(self):
        self.test_predictions = self.replace(self.test_predictions, 0, self.y_values[0])
        self.test_predictions = self.replace(self.test_predictions, 1, self.y_values[1])
        with open('data/submission_1.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in range(len(self.id)):
                writer.writerow([self.id[i], self.test_predictions[i]])
                

class Reg_Logistic_Regression:
    def __init__(self, max_iters, gamma, lambda_, train_set, test_set, id):
        self.max_iters = max_iters
        self.lambda_ = lambda_
        self.gamma = gamma
        self.x_train, self.y_train = train_set
        self.x_test, self.y_test = test_set
        self.id = id
        self.initial_weights = np.zeros((np.shape(self.x_train)[1]))
        self.weights = None
        self.train_loss = None
        self.test_loss = None
        self.y_values = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.test_predictions = None
        self.prepare_data()
        
    def replace(self, data, value, new_value):
        data[data == value] = new_value
        return data
    
    def prepare_data(self):
        self.y_values = np.unique(self.y_train)
        if abs(self.y_values[0] - self.y_values[1]) > 1:
            self.y_train = self.replace(self.y_train, self.y_values[0], 0)
            self.y_train = self.replace(self.y_train, self.y_values[1], 1)
            self.y_test = self.replace(self.y_test, self.y_values[0], 0)
            self.y_test = self.replace(self.y_test, self.y_values[1], 1)
        
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def calculate_grad(self, y, tx, w):
        return (1 / y.shape[0] * (self.sigmoid(tx.dot(w)) - y).T.dot(tx)).T
    
    def calculate_loss(self, y, tx, w):
        return np.sum(-(1 / y.shape[0]) * (y.T.dot(np.log(self.sigmoid(tx.dot(w))))+ (1 - y).T.dot(np.log(1 - self.sigmoid(tx.dot(w))))))
    
    def learning_by_gradient_descent(self, y, tx, w, gamma, lambda_):
        loss = self.calculate_loss(y, tx, w) + lambda_ * np.sum(np.square(w))
        w = w - gamma * (self.calculate_grad(y, tx, w) + 2 * lambda_ * w)
        return loss, w
    
    def predict(self, result):
        return np.array([1 if x > 0.5 else 0 for x in result])
    
    def train(self):
        self.weights = self.initial_weights
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        for n_iter in range(self.max_iters):
            self.train_loss, self.weights = self.learning_by_gradient_descent(self.y_train, self.x_train, self.weights, self.gamma, self.lambda_)
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights) + self.lambda_ * np.sum(np.square(self.weights))
        predictions = self.sigmoid(self.x_train.dot(self.weights))
        train_predictions = self.predict(predictions)
        correct = np.sum(train_predictions == self.y_train)
        self.train_accuracy = correct / self.x_train.shape[0]
    
    def test(self):
        self.test_loss = self.calculate_loss(self.y_test, self.x_test, self.weights)
        predictions = self.sigmoid(self.x_test.dot(self.weights))
        self.test_predictions = self.predict(predictions)
        correct = np.sum(self.test_predictions == self.y_test)
        self.test_accuracy = correct / self.x_test.shape[0]
        print(self.test_accuracy)
          
    def submission(self):
        self.test_predictions = self.replace(self.test_predictions, 0, self.y_values[0])
        self.test_predictions = self.replace(self.test_predictions, 1, self.y_values[1])
        with open('data/submission_1.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in range(len(self.id)):
                writer.writerow([self.id[i], self.test_predictions[i]])