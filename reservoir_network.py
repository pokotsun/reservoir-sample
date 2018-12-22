import numpy as np
from scipy import linalg
class ReservoirNetWork:

    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate=0.1, activator=np.tanh):
        self.inputs = inputs
        self.reservoir_nodes = np.random.uniform(0, 1, num_reservoir_nodes)
        self.outputs = np.zeros(num_output_nodes)

        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])
        
        # 
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate
        self.activator = activator

    # create weights which variated by -0.1 or +0.1
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        return (np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1
    
    # Reservoir層の重みを作成
    def _generate_reservoir_weights(self, num_nodes, scale_weights_reservoir=0.9):
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return (weights / spectral_radius) * scale_weights_reservoir

    def print_weights(self):
        print(f"weights_input:\n{self.weights_input}\n")
        print(f"weights_reservoir:\n{self.weights_reservoir}\n")
        print(f"weights_output:\n{self.weights_output}\n")

    def train(self, lambda0=0.1):
        for input in self.inputs:
            current_x = (1 - self.leak_rate) * self.reservoir_nodes
            current_x += self.leak_rate * (np.array([input]) @ self.weights_input
             + self.reservoir_nodes @ self.weights_reservoir)
            current_x = self.activator(current_x)
            
            # Ridge Regression
            E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
            # print(f"x @ x.T: {current_x.T @ current_x}")
            inv_x = np.linalg.inv(current_x.T @ current_x + E_lambda0)
            # print(f"inv_x: {inv_x}")
            self.weights_output = (inv_x @ current_x.T).reshape([self.num_reservoir_nodes, self.num_output_nodes]) @ np.array([input])
            # print(f"weights_output: {self.weights_output}")
            self.outputs = self.activator(current_x @ self.weights_output)
            print(f"output, input: {self.outputs}, {input}")
            print(f"RMSE: {np.sqrt((self.outputs - np.array(input))**2)}")



            




    
    

