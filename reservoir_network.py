import numpy as np
from scipy import linalg

class ReservoirNetWork:

    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate=0.1, activator=np.tanh):
        self.inputs = inputs
        # self.reservoir_nodes = np.zeros(num_reservoir_nodes)
        self.internal_states = np.array([np.zeros(num_reservoir_nodes)])

        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])
        
        # number of each layer nodes
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate
        self.activator = activator

    def _get_next_reservoir_nodes(self, input, current_state):
        next_state = (1 - self.leak_rate) * current_state
        next_state += self.leak_rate * (np.array([input]) @ self.weights_input
            + current_state @ self.weights_reservoir)
        return self.activator(next_state)

    def _update_weights_output(self, lambda0):
        # Ridge Regression
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
        inv_x = np.linalg.inv(self.internal_states.T @ self.internal_states + E_lambda0)
        # update weights of output layer
        self.weights_output = (inv_x @ self.internal_states.T) @ self.inputs

    def train(self, lambda0=0.1):
        for input in self.inputs:
            current_state = np.array(self.internal_states[-1])
            self.internal_states = np.append(self.internal_states,
                [self._get_next_reservoir_nodes(input, current_state)], axis=0)
        self.internal_states = self.internal_states[1:]
        self._update_weights_output(lambda0)

    def get_train_result(self):
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for input in self.inputs:
            reservoir_nodes = self._get_next_reservoir_nodes(input, reservoir_nodes)
            # print(f"reservoir_nodes: {reservoir_nodes}")
            outputs.append(self.get_output(reservoir_nodes))
        # print(f"internal_states: {self.internal_states}")
        return outputs

    def predict(self, length_of_sequence, lambda0=0.01):
        predicted_outputs = np.array([self.inputs[-1]])

        return predicted_outputs

    # def predict(self, length_of_sequence, lambda0=0.1):
    #     predicted_outputs = np.array([self.inputs[-1]])

    #     for _ in range(length_of_sequence - 1):
    #         last_output = predicted_outputs[-1]
    #         self._update_reservoir_nodes(last_output)
    #         # self._update_weights_output(last_output, lambda0)
    #         predicted_output = self.get_current_output()
    #         predicted_outputs = np.append(predicted_outputs, predicted_output)

    #     return predicted_outputs

    # get output of current state
    def get_output(self, reservoir_nodes):
        return self.activator(reservoir_nodes @ self.weights_output)

    def get_next_output(self):
        return self.activator(self.internal_states[-1] @ self.weights_output)
    
    def minimum_square_error(self, data_a, data_b):
        return np.sqrt(np.sum((data_a - data_b) ** 2)) / len(data_a)


    ##### private method ########
    # create weights either -0.1 or +0.1
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        return (np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1
    
    # create weights of reservoir layer
    def _generate_reservoir_weights(self, num_nodes):
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius
        



            




    
    

