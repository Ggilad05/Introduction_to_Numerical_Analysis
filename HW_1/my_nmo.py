import matplotlib.pyplot as plt
import math
import numpy as np

# nmo formula
def nmo(x, v, t_0):
    return t_0 * t_0 + ((x * x)/(v * v))


def my_nmo():
    x = np.arange(0, 10010, 10)
    v = 2200
    t_0 = 0.22

    t_square = nmo(x, v, t_0)
    t = [math.sqrt(i) for(i) in t_square]

    # Plotting
    plt.plot(x, t)
    plt.xlabel("X [m]")
    plt.ylabel("time [sec]")
    plt.title("My NMO - X VS TIME")
    plt.show()


if __name__ == '__main__':
    my_nmo()
