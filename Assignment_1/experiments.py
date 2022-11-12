from mandelbrot import Monte_carlo_sampling, Latin_hypercube_sampling, \
    Orthogonal_Latin_hypercube_sampling, is_in
import numpy as np
from multiprocessing import Pool
import pandas as pd

# s_length = 100
# s_start = 100
# s_step = 100
# s1 = s_length * s_step  # The biggest s value
# s_list = np.arange(0, s_length) * s_step + s_start  # the list containing s values

'''
The Orthogonal sampling method from scipy can only work with sampling size = p**2,
where p is a prime number. May apply a different method in future.
'''
# get prime numbers
def primes_method(n):
    out = list()
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p] and sieve[p]%2==1):
            out.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return out

s_list = np.array(primes_method(100)) ** 2
s_list = np.delete(s_list, 0)
s1 = s_list[-1]
s_length = len(s_list)

ite = 60
# tune the i (number of iterations) value to see the impact
ite_list = np.arange(1, ite+1, dtype=int)


def mandelbrot_sampling():
    '''i experiment'''
    mand_area_i = [np.zeros(ite), np.zeros(ite), np.zeros(ite)]
    for i in range(ite):
        # initialize the sampling points (random sampling)
        c_MC = Monte_carlo_sampling(s1, circle='N')
        c_LHS = Latin_hypercube_sampling(s1)
        c_OS = Orthogonal_Latin_hypercube_sampling(s1)
        C = [c_MC, c_LHS, c_OS]
        # compute the mandelbrot set area
        for j, c in enumerate(C):
            mand_area = (is_in(c, ite_list[i]).sum()/s1) * np.pi * 4
            mand_area_i[j][i] = mand_area

    '''s experiment'''
    mand_area_s = [np.zeros(s_length), np.zeros(s_length), np.zeros(s_length)]
    for i in range(s_length):
        # initialize the sampling points (random sampling)
        c_MC = Monte_carlo_sampling(s_list[i], circle='N')
        c_LHS = Latin_hypercube_sampling(s_list[i])
        c_OS = Orthogonal_Latin_hypercube_sampling(s_list[i])
        C = [c_MC, c_LHS, c_OS]
        # compute the mandelbrot set area
        for j, c in enumerate(C):
            mand_area = (is_in(c, ite).sum()/s_list[i]) * np.pi * 4
            mand_area_s[j][i] = mand_area

    return mand_area_i, mand_area_s


if __name__ == '__main__':
    pool = Pool(10)

    # repeat experiments for 10 times (stochasticity)
    repetition = 10
    MAND_AREA = pool.starmap(mandelbrot_sampling, [() for _ in range(repetition)])
    # Note the shape of MAND_AREA: a 3d list, the first dimension: [rep_1, rep_2, ..., rep_n],
    # the second dimension: (mand_area_i, mand_area_s),
    # the thrid dimension: (area_MC, area_LHS, area_OS).
    # That is:
    # MAND_AREA_MC = MAND_AREA[:][:][0]
    # MAND_AREA_LHS = MAND_AREA[:][:][1]
    # MAND_AREA_OS = MAND_AREA[:][:][2]

    s_methods = ['MC', 'LHS', 'OS']

    df_ite = pd.DataFrame(columns=['iteration', 'area', 'repetition'])
    df_sampling = pd.DataFrame(columns=['num_s', 'area', 'repetition'])

    for i in range(repetition):
        for j in range(len(s_methods)):
            df_i = pd.DataFrame({'iteration': ite_list, \
                                'area': MAND_AREA[i][0][j], \
                                'repetition': np.repeat(i+1, ite), \
                                's_method': [s_methods[j]] * ite})
            df_ite = pd.concat([df_ite, df_i], ignore_index=True)

            df_s = pd.DataFrame({'num_s': s_list, \
                                'area': MAND_AREA[i][1][j], \
                                'repetition': np.repeat(i+1, s_length), \
                                's_method': [s_methods[j]] * s_length})
            df_sampling = pd.concat([df_sampling, df_s], ignore_index=True)

    df_ite.to_csv('test_i.csv', index=False)
    df_sampling.to_csv('test_s.csv', index=False)