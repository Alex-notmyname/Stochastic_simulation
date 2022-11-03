from mandelbrot import Monte_carlo_sampling, is_in
import numpy as np
from multiprocessing import Pool
import pandas as pd

ite = 60
s1 = int(1e4)
s_length = 40

# tune the s value to see the impact
s_list = np.linspace(100, s1, s_length, dtype=int)

# tune the i value to see the impact
ite_list = np.arange(1, ite+1, dtype=int)


def mandelbrot_sampling():
    '''i experiment'''
    mand_area_i = np.zeros(ite)
    for i in range(ite):
        # initialize the sampling points (random sampling)
        c = Monte_carlo_sampling(s1)
        # compute the mandelbrot set area
        mand_area = (is_in(c, ite_list[i]).sum()/s1) * np.pi * 4
        mand_area_i[i] = mand_area

    '''s experiment'''
    mand_area_s = np.zeros(s_length)
    for i in range(s_length):
        # initialize the sampling points (random sampling)
        c = Monte_carlo_sampling(s_list[i])
        # compute the mandelbrot set area
        mand_area = (is_in(c, ite).sum()/s_list[i]) * np.pi * 4
        mand_area_s[i] = mand_area

    return mand_area_i, mand_area_s


if __name__ == '__main__':
    pool = Pool(10)

    # repeat experiments for 10 times (stochasticity)
    repetition = 10
    MAND_AREA = pool.starmap(mandelbrot_sampling, [()for _ in range(repetition)])

    df_ite = pd.DataFrame(columns=['iteration', 'area', 'repetition'])
    df_sampling = pd.DataFrame(columns=['num_s', 'area', 'repetition'])

    for i in range(repetition):
        df_i = pd.DataFrame({'iteration': ite_list, \
                             'area': MAND_AREA[i][0], \
                             'repetition': np.repeat(i+1, ite)})
        
        df_ite = pd.concat([df_ite, df_i], ignore_index=True)

        df_s = pd.DataFrame({'num_s': s_list, \
                             'area': MAND_AREA[i][1], \
                             'repetition': np.repeat(i+1, s_length)})

        df_sampling = pd.concat([df_sampling, df_s], ignore_index=True)

    df_ite.to_csv('test_i.csv', index=False)
    df_sampling.to_csv('test_s.csv', index=False)