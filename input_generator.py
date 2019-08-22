import numpy as np

class InputGenerator:

    def __init__(self, start_time, end_time, dt):
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.num_time_steps = int((end_time - start_time)/dt)
    
    def generate_sin(self, amplitude=1.0):
        return np.sin(np.arange(self.start_time, self.end_time, self.dt)) * amplitude

    def generate_rossler(self, a=0.2, b=0.2, c=5.7):
        data = [(0,0,0)]
        for t in np.arange(self.dt, self.end_time, self.dt):
            (px, py, pz) = data[-1]
            x = px + self.dt * (-py - pz)
            y = py + self.dt * (px + a * py)
            z = pz + self.dt * (b + pz * (px - c))
            data.append((x, y, z))
        return data

    

