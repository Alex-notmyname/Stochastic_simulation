from mandelbrot import Monte_carlo_sampling, Latin_hypercube_sampling, \
    Orthogonal_Latin_hypercube_sampling, is_in
import numpy as np
from multiprocessing import Pool
import pandas as pd
import time

# Note the actual sampling space is s**2!
s1 = 90
s_list = np.arange(10, s1+1, step=1, dtype=int)  # the list containing s values
s_length = len(s_list)

ite = 150
# tune the i (number of iterations) value to see the impact
ite_list = np.arange(6, ite+1, dtype=int)
i_length = len(ite_list)

ite_s = 150
sampling_i = 20

def mandelbrot_sampling():
    '''i experiment'''
    mand_area_i = [np.zeros(i_length, dtype=np.float64), np.zeros(i_length, dtype=np.float64), \
        np.zeros(i_length, dtype=np.float64), np.zeros(i_length, dtype=np.float64)]
    # mand_area_i = [np.zeros(ite, dtype=np.float64)] * 3
    for i in range(i_length):
        # initialize the sampling points (random sampling)
        c_MC = Monte_carlo_sampling(sampling_i, circle='N')
        c_LHS = Latin_hypercube_sampling(sampling_i)
        c_OS = Orthogonal_Latin_hypercube_sampling(sampling_i)
        c_MC_c = Monte_carlo_sampling(sampling_i, circle='Y')
        C = [c_MC, c_LHS, c_OS]
        # compute the mandelbrot set area
        for j, c in enumerate(C):
            mand_area = (is_in(c, ite_list[i]).sum()/sampling_i**2) * 16
            mand_area_i[j][i] = mand_area
        # new method
        mand_area = (is_in(c_MC_c, ite_list[i]).sum()/sampling_i**2) * np.pi * 4
        mand_area_i[3][i] = mand_area

    '''s experiment'''
    mand_area_s = [np.zeros(s_length, dtype=np.float64), np.zeros(s_length, dtype=np.float64), \
        np.zeros(s_length, dtype=np.float64), np.zeros(s_length, dtype=np.float64)]
    # mand_area_s = [np.zeros(s_length, dtype=np.float64)] * 3
    for i in range(s_length):
        # initialize the sampling points (random sampling)
        c_MC = Monte_carlo_sampling(s_list[i], circle='N')
        c_LHS = Latin_hypercube_sampling(s_list[i])
        c_OS = Orthogonal_Latin_hypercube_sampling(s_list[i])
        c_MC_c = Monte_carlo_sampling(s_list[i], circle='Y')
        C = [c_MC, c_LHS, c_OS]
        # compute the mandelbrot set area
        for j, c in enumerate(C):
            mand_area = (is_in(c, ite_s).sum()/s_list[i]**2) * 16
            mand_area_s[j][i] = mand_area
        # new method
        mand_area = (is_in(c_MC_c, ite_s).sum()/s_list[i]**2) * np.pi * 4
        mand_area_s[3][i] = mand_area

    return mand_area_i, mand_area_s


if __name__ == '__main__':
    start_time = time.time()

    pool = Pool()

    # repeat experiments for 10 times (stochasticity)
    repetition = 100
    MAND_AREA = pool.starmap(mandelbrot_sampling, [() for _ in range(repetition)])
    pool.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    # Note the shape of MAND_AREA: a 3d list, the first dimension: [rep_1, rep_2, ..., rep_n],
    # the second dimension: (mand_area_i, mand_area_s),
    # the thrid dimension: (area_MC, area_LHS, area_OS).
    # That is:
    # MAND_AREA_MC = MAND_AREA[:][:][0]
    # MAND_AREA_LHS = MAND_AREA[:][:][1]
    # MAND_AREA_OS = MAND_AREA[:][:][2]

    s_methods = ['MC', 'LHS', 'OS', 'MC_c']

    df_ite = pd.DataFrame(columns=['iteration', 'area', 'repetition'])
    df_sampling = pd.DataFrame(columns=['num_s', 'area', 'repetition'])

    for i in range(repetition):
        for j in range(len(s_methods)):
            df_i = pd.DataFrame({'iteration': ite_list, \
                                'area': MAND_AREA[i][0][j], \
                                'repetition': np.repeat(i+1, i_length), \
                                's_method': [s_methods[j]] * i_length})
            df_ite = pd.concat([df_ite, df_i], ignore_index=True)

            df_s = pd.DataFrame({'num_s': s_list**2, \
                                'area': MAND_AREA[i][1][j], \
                                'repetition': np.repeat(i+1, s_length), \
                                's_method': [s_methods[j]] * s_length})
            df_sampling = pd.concat([df_sampling, df_s], ignore_index=True)

    df_ite.to_csv('results_i.csv', index=False)
    df_sampling.to_csv('results_s.csv', index=False)