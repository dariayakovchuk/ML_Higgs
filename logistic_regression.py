import numpy as np 

class Logistic_Regression:
    def __init__(self, gamma):
        self.max_iters = 40000
        self.gamma = gamma
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.train_loss, self.test_loss = None, None
        self.train_accuracy, self.test_accuracy = None, None
        self.weights = None
        self.test_predictions = None
    
    def least_squares(self, y, tx):
        w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
        return w
        
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
    
    def train(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.weights = np.zeros((np.shape(self.x_train)[1]))
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        loss_prev = 0
        threshold = 1e-8
        for n_iter in range(self.max_iters):
            if n_iter % 10000 == 0:
                self.gamma = self.gamma/2
            self.train_loss, self.weights = self.learning_by_gradient_descent(self.y_train, self.x_train, self.weights, self.gamma)
            if abs(loss_prev - self.train_loss) < threshold:
                break
            loss_prev = self.train_loss
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        predictions = self.sigmoid(self.x_train.dot(self.weights))
        train_predictions = self.predict(predictions)
        correct = np.sum(train_predictions == self.y_train)
        self.train_accuracy = correct / self.x_train.shape[0]
    
    def test(self, x_test, y_test):
        self.x_test, self.y_test = x_test, y_test
        self.test_loss = self.calculate_loss(self.y_test, self.x_test, self.weights)
        predictions = self.sigmoid(self.x_test.dot(self.weights))
        self.test_predictions = self.predict(predictions)
        correct = np.sum(self.test_predictions == self.y_test)
        self.test_accuracy = correct / self.x_test.shape[0]
                

class Reg_Logistic_Regression:
    def __init__(self, gamma, lambda_):
        self.max_iters = 4000
        self.gamma = gamma
        self.lambda_ = lambda_
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.train_loss, self.test_loss = None, None
        self.train_accuracy, self.test_accuracy = None, None
        self.weights = None
        self.test_predictions = None
            
    def least_squares(self, y, tx):
        w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
        return w
        
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
    
    def train(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.weights = np.zeros((np.shape(self.x_train)[1]))
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights)
        threshold = 1e-8
        loss_prev = 0
        for n_iter in range(self.max_iters):
            if n_iter % 1000 == 0:
                self.gamma = self.gamma/5
            self.train_loss, self.weights = self.learning_by_gradient_descent(self.y_train, self.x_train, self.weights, self.gamma, self.lambda_)
            if abs(loss_prev - self.train_loss) < threshold:
                break
            loss_prev = self.train_loss
        self.train_loss = self.calculate_loss(self.y_train, self.x_train, self.weights) + self.lambda_ * np.sum(np.square(self.weights))
        predictions = self.sigmoid(self.x_train.dot(self.weights))
        train_predictions = self.predict(predictions)
        correct = np.sum(train_predictions == self.y_train)
        self.train_accuracy = correct / self.x_train.shape[0]
    
    def test(self, x_test, y_test):
        self.x_test, self.y_test = x_test, y_test
        self.test_loss = self.calculate_loss(self.y_test, self.x_test, self.weights)
        predictions = self.sigmoid(self.x_test.dot(self.weights))
        self.test_predictions = self.predict(predictions)
        correct = np.sum(self.test_predictions == self.y_test)
        self.test_accuracy = correct / self.x_test.shape[0]