import numpy as np

class InputGenerator:

    def __init__(self, num_time_steps, dt):
        self.num_time_steps = num_time_steps
        self.dt = dt
        self.total_time = int(num_time_steps * dt)
    
    def generate_sin(self, amplitude=1.0):
        return np.sin(np.arange(0, self.total_time, self.dt)) * amplitude
    

