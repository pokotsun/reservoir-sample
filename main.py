import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

PREDICT_T = np.pi * 8
T = np.pi * 4
dt = np.pi * 0.05

def plot_func(x, y):
    plt.plot(x, y)
    plt.show()


def main():
    print(f"Hello, My Reservoir Network!!")
    input_data = InputGenerator(0, T, dt).generate_sin()
    rn = ReservoirNetWork(inputs=input_data, num_input_nodes=1, num_reservoir_nodes=10, num_output_nodes=1)
    # rn.print_weights()
    trained_data = rn.train()
    print(f"MINIMUM_SQUARE_ERROR: {rn.minimum_square_error(trained_data, input_data)}")
    
    predict_step = int((PREDICT_T - T) / dt)
    desired_data = InputGenerator(T, PREDICT_T, dt).generate_sin()
    predicted_data = rn.predict(predict_step)
    print(f"predicted_results: {predicted_data}")
    print(f"PREDICT_MINIMUM: {rn.minimum_square_error(predicted_data, desired_data)}")
    
    plt.plot(np.arange(int(T/dt)), InputGenerator(0, T, dt).generate_sin())
    plt.plot(np.arange(int(T/dt)), trained_data)
   
    # plt.plot(np.arange(int(PREDICT_T/dt)), InputGenerator(0, PREDICT_T, dt).generate_sin())
    # plt.plot(np.arange(int(PREDICT_T/dt)), np.append(trained_data, predicted_data))
    plt.show()

if __name__=="__main__":
    main()








