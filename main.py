import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = 50
dt = 0.5

def plot_func(x, y):
    plt.plot(x, y)
    plt.show()


def main():
    print(f"Hello, My Reservoir Network!!")
    inputs = InputGenerator(T, dt).generate_sin()
    rn = ReservoirNetWork(inputs=inputs, num_input_nodes=1, num_reservoir_nodes=10, num_output_nodes=1)
    # rn.print_weights()
    rn.train()
    # rn.print_weights()
    # plot_func(np.arange(int(T/dt)), inputs)
    

if __name__=="__main__":
    main()








