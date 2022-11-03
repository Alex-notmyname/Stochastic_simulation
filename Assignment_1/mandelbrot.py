import numpy as np
from numpy import random

def is_in(c, num_iterations):
    z = 0.0
    for _ in range(num_iterations):
        z = z ** 2 + c

    return abs(z) <= 2   # True or False?


def Monte_carlo_sampling(num, R=2):
    r = R * np.sqrt(random.random_sample(size=num))  # random radius
    theta = random.random_sample(size=num) * 2 * np.pi  # random angel

    # convert to Cartesian coordinates and map to a complex plane
    c_array = r * np.cos(theta) + 1j * r * np.sin(theta)

    return c_array
