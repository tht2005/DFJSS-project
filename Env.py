import numpy as np

from statistics import mean

# number of machines and jobs
# all values must be greater than 0
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
    def initEpisode(self):
        # number of machines and jobs
        self.m = np.random.choice(SET_M)
        self.n_init = np.random.choice(SET_N_INIT)
        self.n_add = np.random.choice(SET_N_ADD)
        self.n_total = self.n_init + self.n_add

        # Arrives time of jobs
        E_ave = np.random.choice(SET_E_AVER);
        random_event_distance = np.random.exponential(scale=E_ave, size=self.n_add)
        self.A = [ 1 ] * self.n_init
        for d in random_event_distance:
            self.A.append(self.A[-1] + d)

        # random operation number
        self.h = np.random.randint(low=U_N_JOB[0], high=U_N_JOB[1] + 1, size=self.n_total)

        # random operation processing time
        self.t = [ 0 ] * self.n_total
        for i in range(self.n_total):
            self.t[i] = [ 0 ] * self.h[i]
            for j in range(self.h[i]):
                self.t[i][j] = np.random.randint(low=U_OPER_TIME[0], high=U_OPER_TIME[1]+1, size=self.m)

        # early-cost and tardi-cost
        self.early_cost = np.random.uniform(low=U_EARLY_COST[0], high=U_EARLY_COST[1], size=self.n_total)
        self.tardy_cost = np.random.uniform(low=U_TARDI_COST[0], high=U_TARDI_COST[1], size=self.n_total)
        
        # due time
        self.D = [ 0 ] * self.n_total
        for i in range(self.n_total):
            # relaxation factor
            f = np.random.uniform(U_RELAX_FACTOR[0], U_RELAX_FACTOR[1])
            # estimate time for operation j of job i to be done
            est = [ mean(self.t[i][j]) for j in range(self.h[i]) ]
            # total estimate time
            total_est = sum(est)
            self.D[i] = self.A[i] + f * total_est

        # machines properties
        self.CT = [ 1 ] * self.m
        self.total_worktime = [ 0 ] * self.m

        # jobs properties
        self.OP = [ 0 ] * self.n_total
        self.ET = [ [1] ] * self.n_total

        # state properties
        self.U_m = 0
        self.ET_e = 0
        self.ET_a = 0
        self.P_a = 0

    def calcU_m(self):
        self.U_m = mean([ (self.total_worktime[i] / self.CT[i]) for i in range(self.m) ])

    def calcET_e(self):
        T_cur = mean(self.CT)
        NJtard = 0
        NJearly = 0
        for i in range(self.n_total):
            if self.OP[i] < self.h[i]:
                T_left = 0
                for j in range(self.OP[i] + 1, self.h[i]):
                    tij = mean(self.t[i][j])
                    T_left += tij
                    if T_cur + T_left > self.D[i]:
                        NJtard += 1
                        break;

                if T_cur + T_left < self.D[i]:
                    NJearly += 1

        self.ET_e = (NJtard + NJearly) / self.n_total

    def calcET_a(self):
        NJa_tard = 0
        NJa_early = 0
        for i in range(self.n_total):
            if self.OP[i] < self.h[i]:
                T_left = 0
                if self.ET[i][self.OP[i]] > self.D[i]:
                    NJa_tard += 1
                    continue
                else:
                    for j in range(self.OP[i] + 1, self.h[i]):
                        tij = mean(self.t[i][j])
                        T_left += tij
                        if self.ET[i][self.OP[i]] + T_left > self.D[i]:
                            NJa_tard += 1
                            break;
                    if self.ET[i][self.OP[i]] + T_left < self.D[i]:
                        NJa_early += 1
        self.ET_a = (NJa_tard + NJa_early) / self.n_total

    def calcP_a(self):
        P2 = [1] * self.n_total
        P = [0] * self.n_total
        for i in range(self.n_total):
            if self.OP[i] < self.h[i]:
                T_left = 0
                for j in range(self.OP[i] + 1, self.h[i]):
                    tij = mean(self.t[i][j])
                    T_left += tij
                
                if self.ET[i][self.OP[i]] > self.D[i]:
                    P[i] = self.tardy_cost[i] * (self.ET[i][self.OP[i]] + T_left - self.D[i])
                    P2[i] = self.tardy_cost[i] * (self.ET[i][self.OP[i]] + T_left - self.D[i]) + 10
                if self.ET[i][self.OP[i]] + T_left < self.D[i]:
                    P[i] = self.early_cost[i] * (self.D[i] - self.ET[i][self.OP[i]] - T_left)
                    P2[i] = self.early_cost[i] * (self.D[i] - self.ET[i][self.OP[i]] - T_left) + 10
        self.P_a = sum(P) / sum(P2)


    def makeAction(self, machine, job):
        assert self.OP[job] < self.h[job], "MakeAction on a completed job"

        o = self.OP[job]
        
        # change machine properties
        self.total_worktime[machine] += self.t[job][o][machine]
        self.CT[machine] = max([ self.CT[machine], self.ET[job][o], self.A[job] ]) + self.t[job][o][machine]

        # change job properties
        self.OP[job] += 1
        self.ET[job].append(self.CT[machine])

        self.calcU_m()
        self.calcET_e()
        self.calcET_a()
        self.calcP_a()

        print(self.U_m, self.ET_e, self.ET_a, self.P_a)

