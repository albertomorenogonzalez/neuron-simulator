import numpy as np
import inspect
import re

# Definition of the perceptron as a class
class Neuron:
    """
    This class represents a perceptron (neuron) in a neural network.

    Attributes:
    - weights (list): List of weights for the neuron.
    - bias (float): Bias term for the neuron.
    - func (str): Activation function to be used ("sigmoid", "relu", or "tanh").
    """

    def __init__(self, weights, bias, func):
        """
        Initializes a neuron with weights, bias, and activation function.

        Parameters:
        - weights (list): List of weights for the neuron.
        - bias (float): Bias term for the neuron.
        - func (str): Activation function to be used ("sigmoid", "relu", or "tanh").
        """
        self.weights = weights
        self.bias = bias
        self.func = func

    def __str__(self) -> str:
        """
        Returns a string representation of the neuron.

        Returns:
        - str: String representation of the neuron.
        """
        return f"Perceptron with weight: {self.weights}, bias = {self.bias} and using the function {self.func}"


    @staticmethod
    def __sigmoid(x):
        """
        Computes the sigmoid activation function.

        Parameters:
        - x (float): Input value.

        Returns:
        - float: Result of the sigmoid function.
        """
        return 1 / (1 + np.e ** -x)


    @staticmethod
    def __relu(x):
        """
        Computes the rectified linear unit (ReLU) activation function.

        Parameters:
        - x (float): Input value.

        Returns:
        - float: Result of the ReLU function.
        """
        if x < 0:
            return 0
        else:
            return x


    @staticmethod
    def __tanh(x):
        """
        Computes the hyperbolic tangent (tanh) activation function.

        Parameters:
        - x (float): Input value.

        Returns:
        - float: Result of the tanh function.
        """
        return np.tanh(x)


    @staticmethod
    def __activation_function(y, activation_function):
        """
        Applies the specified activation function to the input 'y'.

        Parameters:
        - y (float): Input value to be transformed.
        - activation_function (str): Name of the activation function.

        Returns:
        - float: Transformed value after applying the activation function.

        Note:
        This method dynamically applies the chosen activation function within the Neuron
        class to the input 'y'. It uses reflection to find the corresponding function
        based on the provided 'activation_function' string.
        """
        for member_name, member_value in inspect.getmembers(Neuron):
            if re.search(activation_function, member_name):
                return member_value(y)


    def predict(self, input_data):
        """
        Makes a prediction based on input data.

        Parameters:
        - input_data (list): List of input values.

        Returns:
        - float: Prediction result.
        """
        y = sum(np.multiply(input_data, self.weights)) + self.bias

        return Neuron.__activation_function(y, self.func)


    def changeBias(self, bias):
        """
        Changes the bias term of the neuron.

        Parameters:
        - bias (float): New bias value.
        """
        self.bias = bias
        
        