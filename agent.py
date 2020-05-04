import numpy as np


class ttt_agent:
    layer_sizes = []
    nn_parameters = {}
    num_episodes = 0

    def __init__(self, hidden_layer_neurons):
        self.layer_sizes = np.array([9, hidden_layer_neurons, 9])

        # Initialization with He-et-al random initialization
        for i in range(1, 3):
            self.nn_parameters[i] = {}
            self.nn_parameters[i]["w"] = np.random.randn((self.layer_sizes[i], self.layer_sizes[i - 1]))\
                                         * np.sqrt(2 / self.layer_sizes[i - 1])
            self.nn_parameters[i]["b"] = np.zeros((self.layer_sizes[i], 1))
