import numpy as np
from numpy import random
from scipy.stats import qmc
import warnings

# treat warnings as error
# warnings.filterwarnings("error")

def is_in(c, num_iterations):
    z = 0.0
    for _ in range(num_iterations):
        # try:
        z = z ** 2 + c
        # except RuntimeWarning:  # for handling very large z values
        #     break

    return abs(z) <= 2   # True or False?


def Monte_carlo_sampling(num, R=2, circle='N'): # Monte Carlo sampling
    if circle == 'Y':
        r = R * np.sqrt(random.random_sample(size=num**2))  # random radius
        theta = random.random_sample(size=num**2) * 2 * np.pi  # random angel

        # convert to Cartesian coordinates and map to a complex plane
        c_array = r * np.cos(theta) + 1j * r * np.sin(theta)

    else:
        x = 4 * random.random_sample(size=num**2) - 2  # scale from [0, 1] to [-2, 2]
        y = 4 * random.random_sample(size=num**2) - 2  # scale from [0, 1] to [-2, 2]

        c_array = x + 1j * y

    return c_array

def Latin_hypercube_sampling(num):
    l_bounds= [-2, -2]
    u_bounds = [2, 2]

    # Plain Latin hypercube sampling
    sampler = qmc.LatinHypercube(d=2, strength=1)
    c_array = sampler.random(n=num**2)
    c_array = qmc.scale(c_array, l_bounds, u_bounds)
    c_array = c_array[:, 0] + 1j * c_array[:, 1]

    return c_array

def Orthogonal_Latin_hypercube_sampling(num):
    n_samples = num**2
    scales = 4.0 / n_samples

    M = np.arange(start=0, stop=n_samples, step=1, dtype=int).reshape((num, num))
    x_list, y_list = M, M
    
    # permute the x and y list
    rng = np.random.default_rng()
    x_list = rng.permuted(x_list, axis=1)
    y_list = rng.permuted(y_list, axis=1)

    # generate sample points
    x = -2 + scales * (x_list.flatten() + random.random_sample(size=num**2))
    y = -2 + scales * (y_list.transpose().flatten() + random.random_sample(size=num**2))

    c_array = x + 1j * y

    return c_array