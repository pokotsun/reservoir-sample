import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

PREDICT_T = np.pi * 16
T = np.pi * 12
dt = np.pi * 0.05
AMPLITUDE = 1.0
LEAK_RATE=0.15

def main():
    input_data = InputGenerator(0, T, dt).generate_sin(amplitude=AMPLITUDE)
    model = ReservoirNetWork(inputs=input_data, num_input_nodes=1, num_reservoir_nodes=100, num_output_nodes=1, leak_rate=LEAK_RATE)

    model.print_weights()
    trained_data = model.train()
    print(f"MINIMUM_SQUARE_ERROR: {model.minimum_square_error(trained_data, input_data)}")
    
    predict_step = int((PREDICT_T - T) / dt)
    desired_data = InputGenerator(T, PREDICT_T, dt).generate_sin(amplitude=AMPLITUDE)
    predicted_data = model.predict(predict_step)
    
    print(f"predicted_results: {predicted_data}")
    print(f"PREDICT_MINIMUM: {model.minimum_square_error(predicted_data, desired_data)}")
    
    # plt.plot(np.arange(0, T, dt), input_data)
    # plt.plot(np.arange(0, T, dt), trained_data)
   
    plt.plot(np.arange(0, PREDICT_T, dt), InputGenerator(0, PREDICT_T, dt).generate_sin(amplitude=AMPLITUDE))
    plt.plot(np.arange(0, PREDICT_T, dt), np.append(trained_data, predicted_data))
    plt.axvline(x=T)
    plt.show()

if __name__=="__main__":
    main()








