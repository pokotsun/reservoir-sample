import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = np.pi * 12
RATIO_TRAIN = 0.8
dt = np.pi * 0.05
AMPLITUDE = 0.9
LEAK_RATE=0.1
NUM_RESERVOIR_NODES = 200

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_sin(amplitude=AMPLITUDE)
    NUM_TRAIN = int(len(data) * RATIO_TRAIN)
    train_data = data[:NUM_TRAIN][:]

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=1, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=1, 
        leak_rate=LEAK_RATE)

    trained_data = model.train()
    print(f"MINIMUM_SQUARE_ERROR: {model.minimum_square_error(trained_data, train_data)}")
    
    # predict_step = int((PREDICT_T - T) / dt)
    # predicted_data = model.predict(predict_step)
    predict_data = data[NUM_TRAIN:]
    predict_step = len(predict_data)
    predicted_data = model.predict(predict_step)
    print(f"PREDICT_MINIMUM: {model.minimum_square_error(predicted_data, predict_data)}")
    
    plt.plot(np.arange(0, T, dt), data, label="inputs")
    plt.plot(np.arange(0, T, dt), np.append(trained_data, predicted_data), label="outputs")
    plt.axvline(x=T, label="end of train", color="green")
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()








