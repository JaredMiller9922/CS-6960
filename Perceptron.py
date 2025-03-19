import numpy as np
import math
from typing import Protocol, Tuple

class Perceptron():
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


    
    def generate_data(self):
        # generate w_star
        self.w_star = np.ones(self.n)

        # Generate x
        self.x = np.random.normal(self.mean, math.sqrt(self.variance), size=(self.m, self.n))

        # Generate y
        for i in range(self.m):
            rand = np.random.rand()
            if rand < 0.9:
                yi = np.sign(np.dot(self.w_star, self.x[i]))
            else:
                yi = np.sign(np.dot(self.w_star, self.x[i]))
            
            self.y.append(yi)
        # Convert to a numpy array
        self.y = np.array(self.y)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for
        '''

        # YOUR CODE HERE
        learning_rate = 1

        weight = np.random.uniform(-0.01, 0.01)
        bias = np.random.uniform(-0.01, 0.01)

        # Prepend a 1 to every feature in the feature vector axis=1 inserts colum wise
        x_b = np.insert(x, 0, 1, axis=1)

        # Create the weight vector and prepend bias term
        weights = np.full((self.n, 1), weight)
        weights_b = np.insert(weights, 0, bias, axis=0)
        self.avg_weight = np.zeros(self.n + 1)
        self.avg_weight = self.avg_weight.reshape(-1,1)

        # Start perceptron
        for i in range(epochs + 1):
            # shuffle the data row-wise making sure we maintain data and label consistency
            x_b, y = shuffle_data(x_b, y)

            # For every feature vector
            for j in range(len(x_b)):
                # item is nessacary here because np.dot returns an array
                if y[j]*np.dot(weights_b.T,x_b[j]).item() <= 0:
                    # reshape is nessacary so that x_b[i] is a column vector
                    weights_b = weights_b + learning_rate*y[j]*x_b[j].reshape(-1,1)
            self.avg_weight += weights_b

            if i % 10 == 0:
                with open("perceptron_restults.txt", "a") as file:
                    file.write("The weights at iteration: " + str(i) + "\n")
                    file.write(str(weights_b))
                    file.write("\n")
                    file.write("\n")

                print("The weights at iteration: " + str(i))
                print(weights_b)
                print()

                
        self.avg_weight = self.avg_weight / epochs
        self.weights = weights_b

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
    with open("perceptron_restults.txt", "w") as file:
        file.write("")
    perceptron = Perceptron()
    perceptron.generate_data()

    perceptron.train(perceptron.x, perceptron.y, 100)

    with open("perceptron_restults.txt", "a") as file:
        file.write("Final Weights\n")
        file.write(str(perceptron.weights))
        file.write("\n")
        file.write("\n")

        file.write("Average Weights\n")
        file.write(str(perceptron.avg_weight))

    print("Final Weights")
    print(perceptron.weights)
    print()
    print("Average Weights")
    print(perceptron.avg_weight)
