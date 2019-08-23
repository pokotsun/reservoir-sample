import numpy as np
import signalz

class InputGenerator:

    def __init__(self, start_time, end_time, dt):
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.num_time_steps = int((end_time - start_time)/dt)
    
    def generate_sin(self, amplitude=1.0):
        return np.sin(np.arange(self.start_time, self.end_time, self.dt)) * amplitude

    def generate_mackey_glass(self, a=0.2, b=1, c=0.9, d=17, e=10, initial=0.1):
        return signalz.mackey_glass(self.num_time_steps, a=a, b=b, c=c, d=d, e=e, initial=initial) - 0.8

    def generate_rossler(self, a=0.2, b=0.2, c=5.7):
        xdata = [0]
        ydata = [0]
        zdata = [0]
        for t in np.arange(self.dt, self.end_time, self.dt):
            (px, py, pz) = (xdata[-1], ydata[-1], zdata[-1])
            x = px + self.dt * (-py - pz)
            y = py + self.dt * (px + a * py)
            z = pz + self.dt * (b + pz * (px - c))
            xdata.append(x)
            ydata.append(y)
            zdata.append(z)
        return np.array([xdata, ydata, zdata])

    def generate_lorenz(self, p=10, r=28, b=8/3):
        xdata = [0.1]
        ydata = [0.1]
        zdata = [0.1]
        for t in np.arange(self.dt, self.end_time, self.dt):
            (px, py, pz) = (xdata[-1], ydata[-1], zdata[-1])
            x = px + self.dt * p * (py - px) 
            y = py + self.dt * (px * (r - pz) - py)
            z = pz + self.dt * (px * py - b * pz)
            xdata.append(x)
            ydata.append(y)
            zdata.append(z)
        return np.array([xdata, ydata, zdata])


