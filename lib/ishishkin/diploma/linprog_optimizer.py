from lib.ishishkin.diploma.models import LinearFunctionOneDimensionalNoise
from scipy.optimize import linprog
import numpy as np

class OneDimensionalNoiseOptimizer:

    def __init__(self, q_bounds = (None, None), a0_bounds=(None,None), a1_bounds=(0.9, 1.1)):

        self.q_bounds = q_bounds
        self.a0_bounds = a0_bounds
        self.a1_bounds = a1_bounds


    def fit(self, x_true, y_noise, intervals_y, verbose=0):

        l = [1, 0, 0]
        A = []
        b = []

        for delta_i, x_i, y_i in zip(intervals_y, x_true, y_noise):
            A.append([-delta_i, -x_i, -1])
            A.append([-delta_i, x_i, 1])
            b.append(-y_i)
            b.append(y_i)

        res = linprog(l, A_ub=A, b_ub=b, bounds=[self.q_bounds, self.a1_bounds, self.a0_bounds])

        if verbose > 0:
            print(res)
        else:
            print(res['message'])

        self.res = res
        self.a_0 = self.res['x'][2]
        self.a_1 = self.res['x'][1]

        return self

    def get_q(self):
        return self.res['x'][0]

    def get_res(self):
        return self.res

    def predict(self, x):

        a_1_rec = self.res['x'][1]
        a_0_rec = self.res['x'][2]

        reconstructed = LinearFunctionOneDimensionalNoise(a_0=a_0_rec, a_1=a_1_rec)

        y_rec = reconstructed.get_y(x)

        return y_rec

class TwoDimensionalNoiseOptimizer:

    def __init__(self, a1_bounds=(-2, 2), q_bounds=(None, None), a0_bounds=(None, None)):

        self.q_bounds = q_bounds
        self.a0_bounds = a0_bounds
        self.a1_bounds = a1_bounds


    def fit(self, x_noise, intervals_x, y_noise, intervals_y, a1_steps=10, verbose=0, force_mute = False):

        min_a1 = self.a1_bounds[0]
        max_a1 = self.a1_bounds[1]
        step_a1 = (max_a1 - min_a1)/a1_steps

        q_min = 10**5
        res_min = None
        a_1_min = np.nan


        for a_1 in np.arange(min_a1, max_a1+step_a1, step_a1):

            l = np.append([1, 0], np.zeros(intervals_x.shape))
            A = []
            b = []
            bounds_list = [(None, None), (None, None)]

            for i, (x_i, sigma_i, y_i, tau_i) in enumerate(zip(x_noise, intervals_x, y_noise, intervals_y)):
                m_list = np.zeros(intervals_x.shape)
                m_list[i] = 1

                A.append(np.append([-tau_i, -1], a_1 * m_list))
                b.append(-y_i + a_1 * x_i)

                A.append(np.append([-tau_i, 1], -a_1 * m_list))
                b.append(y_i - a_1 * x_i)

                A.append(np.append([-sigma_i, 0], m_list))
                b.append(0)

                A.append(np.append([-sigma_i, 0], -1 * m_list))
                b.append(0)

                bounds_list.append((-sigma_i, sigma_i))

            res = linprog(l, A_ub=A, b_ub=b, bounds=bounds_list)
            q_res = res['x'][0]

            if q_res < q_min:

                if verbose > 2:
                    print(res)
                elif not force_mute:
                    print(q_res)
                    print(f"a1 = {a_1: .4f}")
                    print(res['message'])
                    print("_"*100)

                res_min = res
                a_1_min = a_1
                q_min = q_res

        self.a_1 = a_1_min
        self.a_0 = res_min['x'][1]

        self.res = res_min
        self.q = res_min['x'][0]

        self.x  = x_noise - res_min['x'][2:]
        self.y = self.a_1*self.x + self.a_0

        if verbose > 1:
            print(f"a_1 = {self.a_1}; a_0 = {self.a_0}; q = {q_min}")


        return self

    def get_q(self):
        return self.res['x'][0]

    def get_res(self):
        return self.res

    def predict(self, x):

        reconstructed = LinearFunctionOneDimensionalNoise(a_0=self.a_0, a_1=self.a_1)

        y_rec = reconstructed.get_y(x)

        return y_rec



