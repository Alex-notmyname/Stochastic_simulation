import numpy as np
from numpy import random
from scipy.stats import qmc

def is_in(c, num_iterations):
    z = 0.0
    for _ in range(num_iterations):
        z = z ** 2 + c

    return abs(z) <= 2   # True or False?


def Monte_carlo_sampling(num, R=2, circle='Y'): # Monte Carlo sampling in a circle
    if circle == 'Y':
        r = R * np.sqrt(random.random_sample(size=num))  # random radius
        theta = random.random_sample(size=num) * 2 * np.pi  # random angel

        # convert to Cartesian coordinates and map to a complex plane
        c_array = r * np.cos(theta) + 1j * r * np.sin(theta)

    else:
        x = 4 * random.random_sample(size=num) - 2  # scale from [0, 1] to [-2, 2]
        y = 4 * random.random_sample(size=num) - 2  # scale from [0, 1] to [-2, 2]

        c_array = x + 1j * y

    return c_array

def Latin_hypercube_sampling(num):
    l_bounds= [-2, -2]
    u_bounds = [2, 2]

    # Plain Latin hypercube sampling
    sampler = qmc.LatinHypercube(d=2, strength=1)
    c_array = sampler.random(n=num)
    c_array = qmc.scale(c_array, l_bounds, u_bounds)
    c_array = c_array[:, 0] + 1j * c_array[:, 1]

    return c_array

def Orthogonal_Latin_hypercube_sampling(num):
    l_bounds= [-2, -2]
    u_bounds = [2, 2]

    # orthogonal-array-based Latin hypercube sampling
    # Note! The number can only be a square of prime number!
    sampler = qmc.LatinHypercube(d=2, strength=2)
    c_array = sampler.random(n=num)
    c_array = qmc.scale(c_array, l_bounds, u_bounds)
    c_array = c_array[:, 0] + 1j * c_array[:, 1]

    return c_array