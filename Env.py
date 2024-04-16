import numpy as np

SET_M = [ 5, 10, 30 ]
SET_N_INIT =  [ 20 ]
SET_N_ADD = [ 20, 30, 50, 100 ]

U_RELAX_FACTOR = [ 0.5, 2 ]

SET_E_AVER = [ 30, 50, 100 ]

U_N_JOB = [ 1, 30 ]
U_OPER_TIME = [ 0, 100 ]
U_EARLY_COST = [ 1, 1.5 ]
U_TARDI_COST = [ 1, 2 ]


class env:
    def __init__(self):
        self.m = np.random.choice(SET_M)
        self.n_init = np.random.choice(SET_N_INIT)
        self.n_add = np.random.choice(SET_N_ADD)

        # f_i

        self.e_ave = np.random.choice(SET_E_AVER);

