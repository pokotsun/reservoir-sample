import numpy as np

class InputGenerator:

    def __init__(self, total_time, dt):
        self.total_time = total_time
        self.dt = dt
        self.num_time_steps = int(total_time/dt)
    
    def generate_sin(self, amplitude=1.0):
        return np.sin(np.arange(0, self.total_time, self.dt)) * amplitude
    

