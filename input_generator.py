import numpy as np

class InputGenerator:

    def __init__(self, start_time, end_time, dt):
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.num_time_steps = int((end_time - start_time)/dt)
    
    def generate_sin(self, amplitude=1.0):
        return np.sin(np.arange(self.start_time, self.end_time, self.dt)) * amplitude
    

