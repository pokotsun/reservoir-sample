import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
T = 20
RATIO_TRAIN = 0.6
dt = 0.01
AMPLITUDE = 0.9
LEAK_RATE=0.06
NUM_INPUT_NODES = 3
NUM_RESERVOIR_NODES = 300
NUM_OUTPUT_NODES = 3

# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_lorenz()
    num_train = int(len(data.T) * RATIO_TRAIN)
    train_data = data.T[:num_train]
    print(f"train_data_shape: {train_data.shape}")

    model = ReservoirNetWork(inputs=train_data,
            num_input_nodes=NUM_INPUT_NODES,
            num_reservoir_nodes=NUM_RESERVOIR_NODES,
            num_output_nodes=NUM_OUTPUT_NODES,
            leak_rate=LEAK_RATE)
    model.train()
    train_result = model.get_train_result() # 訓練の結果を取得

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2], marker='o')

    ax.set_title("Echo State Network Sin Prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if __name__=="__main__":
    main()

