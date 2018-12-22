import numpy as np

class ReservoirNetWork:

    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate=0.5, activator=np.tanh):
        self.inputs = inputs
        self.nodes_reservoir = np.random.uniform(0, 1, num_reservoir_nodes)
        self.outputs = np.zeros(num_output_nodes)
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])
        self.num_nodes_reservoir = num_reservoir_nodes
        self.leak_rate = leak_rate
        self.activator = activator

    # 重みを作成
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        return np.random.randint(0, 2, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1
    
    # Reservoir層の重みを作成
    def _generate_reservoir_weights(self, num_nodes):
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        return weights

    def print_weights(self):
        print(f"weights_input:\n{self.weights_input}\n")
        print(f"weights_reservoir:\n{self.weights_reservoir}\n")
        print(f"weights_output:\n{self.weights_output}\n")

    def train(self, lambda0=0.1):
        for input in self.inputs:
            current_x = (1 - self.leak_rate) * self.nodes_reservoir
            current_x += self.leak_rate * (np.array([input]) @ self.weights_input
             + self.nodes_reservoir @ self.weights_reservoir)
            
            print(f"current_x: {current_x}")

            # Ridge Regression
            E_lambda0 = np.identity(self.num_nodes_reservoir) * lambda0 # lambda0
            inv_x = np.linalg.inv(current_x.T @ current_x + E_lambda0)
            print(f"inv_x: {inv_x}")
            self.weights_output = inv_x @ self.weights_output
            print(f"weights_output: {self.weights_output}")



            




    
    

