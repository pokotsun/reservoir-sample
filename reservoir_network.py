import numpy as np

class ReservoirNetWork:

    def __init__(self, inputs, num_input_nodes, num_reservoir_layer_nodes, num_output_nodes):
        self.inputs = inputs
        self.weights_input = self._generate_variational_weights(num_input_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_layer_nodes)
        self.weights_output = np.zeros(num_output_nodes)

    # 重みを作成
    def _generate_variational_weights(self, num_nodes):
        return np.random.randint(0, 2, num_nodes) * 2 - 1
    
    # Reservoir層の重みを作成
    def _generate_reservoir_weights(self, num_nodes):
        weights = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        return weights
    
    

