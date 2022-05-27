import pandas as pd
import numpy as np
from lib.ishishkin.diploma.normal_distribution_functions import norm_dence, phi_func, phi_func_reversed
from lib.ishishkin.diploma.granular import RobbinsMonroeGranular


# количество гранул
granules_num = 8


# условия остановки алгоритма
early_stop = 10**5
k = 1
eps_max = np.inf
eps = 0.01

# предположения о виде распределения
mu = 0

# вероятность ошибки первого рода
a = 1.001

# коэффициенты ряда Роббинса Монро
alpha = 1

# начальные предположения
init_g_list = []
chi_list = []
xi_list = np.empty(0)

for num in range(0, granules_num):
    g_i = phi_func(3+num/10)

    init_g_list.append(phi_func(g_i, mu))
    chi_list.append(np.empty(0))

granulator = RobbinsMonroeGranular(granules_num, init_g_list, eps, early_stop, a)
granulator.normal_granular(mu, 1)

print(f"init g = {granulator.init_g_list}")
print(f"adjusted g = {granulator.g_list}")
print(f"mes g = {granulator.granulas_mes}")

df_granulas = pd.DataFrame(data={'init_g': granulator.init_g_list,
                                 'adj_g': granulator.g_list,
                                 'G': [phi_func_reversed(g_i, mu) for g_i in granulator.g_list],
                                 'g_mes': granulator.granulas_mes})

df_chi = pd.DataFrame(data=granulator.chi_list)
df_g = pd.DataFrame(data=granulator.g_values)
df_xi = pd.DataFrame(data=granulator.xi_list, columns=['xi'])

writer = pd.ExcelWriter('../data/oversized_granules.xlsx', engine='xlsxwriter')
df_granulas.to_excel(writer, sheet_name='granules', index=False)
df_xi.to_excel(writer, sheet_name='xi_history', index=False)
df_chi.to_excel(writer, sheet_name='characteristics_functions', index=False)
df_g.to_excel(writer, sheet_name='g_history', index=False)

writer.save()
#
# print()
# print("chi")
# print(df_chi.head())
#
# print()
# print('g')
# print(df_g.head())


