import numpy as np
import math
from typing import Protocol, Tuple
import matplotlib.pyplot as plt

class SGD():
    def __init__(self):
        self.weights = None
        self.w_star = None
        self.avg_weight = None
        self.x = None
        self.y = []

        self.n = 20
        self.m = 100

        self.mean = 0
        self.variance = 0.1

        self.iter_list = []
        self.loss_list = []

    def generate_data(self):
        # generate w_star
        self.w_star = np.ones(self.n)

        # Generate x
        self.x = np.random.normal(self.mean, math.sqrt(self.variance), size=(self.m, self.n))

        # Generate y
        for i in range(self.m):
            rand = np.random.rand()
            if rand < 0.9:
                yi = 1 if np.sign(np.dot(self.w_star, self.x[i])) > 0 else 0
            else:
                yi = 0 if np.sign(np.dot(self.w_star, self.x[i])) > 0 else 1
            
            self.y.append(yi)
        # Convert to a numpy array
        self.y = np.array(self.y)
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr = 1):
        weight = np.random.uniform(-0.01, 0.01)
        bias = np.random.uniform(-0.01, 0.01)

        # Prepend a 1 to every feature in the feature vector axis=1 inserts colum wise
        x_b = np.insert(x, 0, 1, axis=1)

        # Create the weight vector and prepend bias term
        weights = np.full((self.n,), weight)   # shape (n,)
        weights_b = np.insert(weights, 0, bias)   # shape (n+1,)

        iter_list = []
        loss_list = []
        for i in range(epochs + 1):
            # Shuffle Our Data
            x_b, y = shuffle_data(x_b, y)

            # Calculate Loss
            cur_loss = self.calculate_loss(x_b, y, weights_b)
            loss_list.append(cur_loss)
            iter_list.append(i)

            # Process gradients one example at a time
            for j in range(self.m):
                # Gradient update w = w - α*∇(w)
                weights_b = weights_b - lr * self.gradient(x_b[j], y[j], weights_b)
            
            # print wegihts after 40 iterations and write them
            if i == 40:
                with open("sgd_results.txt", "a") as file:
                    file.write("The weights at iteration: " + str(i) + "\n")
                    file.write(str(weights_b))
                    file.write("\n")
        
        self.iter_list = iter_list
        self.loss_list = loss_list
        self.weights = weights_b

    def gradient(self, x, y, w):
        # We are summing vectors
        z = np.dot(w.T, x)
        return (self.sigmoid(z) - y) * x 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_loss(self, x, y, w):
        # To avoid taking the log of 0 add a small epsilon
        eps = 1e-10
        loss = 0
        for i in range(self.m):
            # Take item so we get a single scalar rather than a list
            z = np.dot(w.T, x[i])
            loss += y[i] * np.log(self.sigmoid(z) + eps) + (1 - y[i]) * np.log(1 - self.sigmoid(z) + eps)

        return -loss
    
    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.iter_list, self.loss_list, marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.grid(True)
        plt.show()

def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]

if __name__ == '__main__':
    with open("sgd_results.txt", "w") as file:
        file.write("")
    log_reg = SGD()
    log_reg.generate_data()
    log_reg.train(log_reg.x, log_reg.y, 100, lr=0.1)
    print(log_reg.iter_list)
    print(log_reg.loss_list)
    log_reg.plot()

