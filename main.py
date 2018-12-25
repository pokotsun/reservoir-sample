import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = np.pi * 16
RATIO_TRAIN = 0.8
# dt = np.pi * 1
dt = np.pi * 0.06
AMPLITUDE = 0.9
LEAK_RATE=0.1
NUM_RESERVOIR_NODES = 50

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_sin(amplitude=AMPLITUDE)
    NUM_TRAIN = int(len(data) * RATIO_TRAIN)
    print(f"NUM_TRAIN: {NUM_TRAIN}")
    train_data = data[:NUM_TRAIN][:]

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=1, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=1, 
        leak_rate=LEAK_RATE)

    model.train()
    trained_data = model.get_train_result()
    print(f"desired_data:\n {len(train_data)}")
    print(f"trained_data:\n {len(trained_data)}")
    print(f"desired_max: {max(train_data)}")
    print(f"trained_max: {max(trained_data)}")
    print(f"desired_data: {np.array(trained_data) * max(train_data) / max(trained_data)}")
    
    plt.plot(np.arange(0, int(T * RATIO_TRAIN), dt), train_data, label="inputs")
    plt.plot(np.arange(0, int(T * RATIO_TRAIN), dt), np.array(trained_data) * max(train_data) / max(trained_data), label="outputs")
    plt.axvline(x=T, label="end of train", color="green")
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()








