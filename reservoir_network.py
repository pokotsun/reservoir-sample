import numpy as np

class ReservoirNetWork:

    def __init__(self, inputs, num_input_nodes, num_reservoir_nodes, num_output_nodes):
        self.inputs = inputs
        self.nodes_reservoir = np.random.uniform(0, 1, num_reservoir_nodes)
        self.outputs = np.zeros(num_output_nodes)
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])

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

    def fit(self):
        for input in self.inputs:
            x = input @ self.weights_input @ self.nodes_reservoir @ self.weights_reservoir @ self.weights_output
            # Ridge Regression
            lambda0 = np.identity(self.nodes_reservoir) * 0.1 # lambda0
            inv_x = np.linalg.inv(x.T @ x + lambda0)
            self.weights_output = (inv_x @ x.T @ input).T




    
    

