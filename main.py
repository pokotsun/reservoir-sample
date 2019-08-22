import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = 50 
RATIO_TRAIN = 0.6
dt = 0.1 
AMPLITUDE = 0.9
LEAK_RATE=0.06
NUM_RESERVOIR_NODES = 300

# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_sin(amplitude=AMPLITUDE)
    num_train = int(len(data) * RATIO_TRAIN)
    train_data = data[:num_train]

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=1, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=1, 
        leak_rate=LEAK_RATE)

    model.train() # 訓練
    train_result = model.get_train_result() # 訓練の結果を取得
    
    num_predict = int(len(data) * (1 - RATIO_TRAIN))
    predict_result = model.predict(num_predict)
    
    ## plot
    plt.plot(np.arange(0, T * RATIO_TRAIN, dt), train_data, label="inputs")
    plt.plot(np.arange(0, T * RATIO_TRAIN, dt), train_result, label="trained")
    plt.plot(np.arange(T * RATIO_TRAIN, T, dt), predict_result, label="predicted")
    plt.axvline(x=int(T * RATIO_TRAIN), label="end of train", color="green") # border of train and prediction
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()
