import numpy as np
import matplotlib.pyplot as plt

T = 10
dt = 0.01
iterations = int(T//dt)

x = np.zeros((iterations, 1))

for i in range(iterations-1):
    dx = np.sin(i*2*np.pi/100)
    x[i+1] = x[i] + dt*dx

plt.figure()
plt.plot(x)

plt.show()