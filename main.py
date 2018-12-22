import numpy as np
from input_generator import InputGenerator
from reservoir_network import ReservoirNetWork


NUM_TIME_STEPS = 10
TIME_STEP = 0.5
def main():
    print(f"Hello, My Reservoir Network!!")
    inputs = InputGenerator(NUM_TIME_STEPS, TIME_STEP).generate_sin()
    print(f"inputs: {inputs}")
    rn = ReservoirNetWork(np.arange(10), 5, 5, 1)
    print(rn.print_weights())

if __name__=="__main__":
    main()








