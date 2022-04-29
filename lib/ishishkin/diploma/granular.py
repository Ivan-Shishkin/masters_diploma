import numpy as np
from lib.ishishkin.diploma.normal_distribution_functions import chi_func, q_func, phi_func_reversed


class RobbinsMonroeGranular:

    def __init__(self, g_num, g_list, eps=0.01, early_stop=10**4, a=1.001, max_tries=1000):

        self.early_stop = early_stop
        self.eps = eps
        self.init_g_list = np.array(g_list)
        self.g_num = g_num
        self.a = a
        self.max_tries = max_tries

    def normal_granular(self, mu, sample_scale):

        # loop stop conditions
        early_stop = self.early_stop
        k = 1
        try_num = 1
        eps_max = np.inf
        eps = self.eps

        # type 1 error probability
        a = self.a

        # Robbins Monroe coefficients
        alpha = 1

        # data storage
        xi_list = np.empty(0)
        chi_list = []
        g_list = self.init_g_list.copy()
        g_values_list = []

        for num in range(0, self.g_num):
            chi_list.append(np.empty(0))
            g_values_list.append(np.array(g_list[num]))

        while eps_max >= eps:

            # random variable realisation
            xi = np.random.normal(loc=0.0, scale=sample_scale)
            xi_list = np.append(xi_list, xi)

            # granulas adjustment
            for num in range(0, self.g_num):
                g_i = g_list[num]
                g_value_list_i =g_values_list[num]

                chi_i = chi_func(g_i, xi, mu)
                chi_list[num] = np.append(chi_list[num], chi_i)

                g_i_corr = g_i + alpha / k * (chi_i - q_func(a, num + 1))
                g_list[num] = g_i_corr
                g_values_list[num] = np.append(g_values_list[num], g_i_corr)

            # errors calculation
            eps_list = []
            for num in range(0, self.g_num - 1):
                eps_i = abs(2 * chi_list[num].mean() - chi_list[num + 1].mean() - 1 - (a - 1) / (a + 1))
                eps_list.append(eps_i)

            eps_max = max(eps_list)

            # update iteration counter
            k += 1

            # early stopping report
            if k > early_stop:

                try_num += 1
                print(f"try to early stop; num iterations = {k}")
                print(f" g > 0 -> {[g_i > 0 for g_i in g_list]}")
                granulas_mes_list = []


                if (min(g_list) > 0) or (try_num > self.max_tries):
                    print('_' * 50, 'EARLY STOP', '_' * 50)

                    for num in range(0, self.g_num):
                        g_i = g_list[num]

                        if g_i < 0:
                            invalid_value = g_i
                            g_i = 10**(-4)/(num+1)
                            g_list[num] = g_i
                            print(f"Invalid value {invalid_value} of granula number {num+1} were set to {g_i}")

                        print(f'g{num + 1} = {g_i}')
                        print(f'G{num + 1} = [-{phi_func_reversed(g_i, mu)}, {phi_func_reversed(g_i, mu)}]')

                        granulas_mes_list.append(chi_list[num].mean())
                        print(f"measure of g{num + 1} = {chi_list[num].mean()}")
                        print()

                    break

        # achievement of the specified accuracy report
        if k < early_stop:
            granulas_mes_list = []
            for num in range(0, self.g_num):
                g_i = g_list[num]
                print(f'g{num + 1} = {g_i}')
                print(f'G{num + 1} = [-{phi_func_reversed(g_i, mu)}, {phi_func_reversed(g_i, mu)}]')

                granulas_mes_list.append(chi_list[num].mean())
                print(f"measure of g{num + 1} = {chi_list[num].mean()}")
                print()

        self.g_list = np.array(g_list)
        self.granulas_mes = np.array(granulas_mes_list)

        self.xi_list = np.array(xi_list)
        self.chi_list = np.array(chi_list).T
        self.g_values = np.array(g_values_list).T

        return self




