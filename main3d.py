import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    

T = 100
RATIO_TRAIN = 0.6
dt = 0.05
AMPLITUDE = 0.9
LEAK_RATE=0.06
NUM_RESERVOIR_NODES = 300

# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_rossler()

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

