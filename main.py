import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork
import matplotlib.pyplot as plt

T = np.pi * 16
RATIO_TRAIN = 0.6
dt = np.pi * 0.06
AMPLITUDE = 0.9
LEAK_RATE=0.1
NUM_RESERVOIR_NODES = 150

def main():
    i_gen = InputGenerator(0, T, dt)
    data = i_gen.generate_sin(amplitude=AMPLITUDE)
    NUM_TRAIN = int(len(data) * RATIO_TRAIN)
    train_data = data[:NUM_TRAIN]

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=1, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=1, 
        leak_rate=LEAK_RATE)

    model.train() # 訓練
    trained_data = model.get_train_result() # 訓練の結果を取得
    
    num_predict_step = 108 # 煩悩の数だけ予測してみる
    predict_result = model.predict(num_predict_step)
    
    ## plot
    plt.plot(np.arange(0, int(T * RATIO_TRAIN), dt), train_data, label="inputs")
    plt.plot(np.arange(0, int(T * RATIO_TRAIN), dt), trained_data, label="trained")
    plt.plot(np.arange(int(T * RATIO_TRAIN), T, dt), predict_result, label="predicted")
    plt.axvline(x=int(T * RATIO_TRAIN), label="end of train", color="green") # 予測と訓練のライン 
    plt.legend()
    plt.title("Echo State Network Sin Prediction")
    plt.xlabel("time[ms]")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()








