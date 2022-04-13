from lib.ishishkin.diploma.models import LinearFunctionTwoDimensionalNoise
from lib.ishishkin.diploma.linprog_optimizer import TwoDimensionalNoiseOptimizer

from lib.ishishkin.diploma.normal_distribution_functions import norm_dence, phi_func, phi_func_reversed
from lib.ishishkin.diploma.granular import RobbinsMonroeGranular

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

# linear function object

a_0 = 1
a_1 = 1

model_2 = LinearFunctionTwoDimensionalNoise(a_0=a_0, a_1=a_1)

# generating samples

min_x = 0
max_x = 110
step_x = 10


loc_x = 0
scale_x = 1

loc_y = 0
scale_y = 1

x_true = model_2.make_true_x(min_x, max_x, step_x)
y_true = model_2.make_true_y(min_x, max_x, step_x)
(x_noise, intervals_x_true), (y_noise, intervals_y_true) = model_2.uni_normal_noise(min_x, max_x, step_x, loc_x, scale_x, loc_y, scale_y)

# reconstruct noise model
granules_dict = {}

for label in ["x_granules", "y_granules"]:

    # количество гранул
    granules_num = 5

    # условия остановки алгоритма
    early_stop = 10 ** 4
    k = 1
    eps_max = np.inf
    eps = 0.01

    # предположения о виде распределения
    mu = 0

    # вероятность ошибки первого рода
    a = 1 + 0.1 ** 4

    # коэффициенты ряда Роббинса Монро
    alpha = 1

    # начальные предположения
    init_g_list = []
    chi_list = []
    xi_list = np.empty(0)

    for num in range(0, granules_num):
        init_g_list.append(phi_func(num * 0.1, mu))
        chi_list.append(np.empty(0))

    granulator = RobbinsMonroeGranular(granules_num, init_g_list, eps, early_stop, a)
    granulator.normal_granular(mu)

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

    granules_dict.update({label:[df_granulas, df_chi, df_g, df_xi]})


# reconstruct function

min_a1 = -2
max_a1 = 2
steps = 100

min_a0 = None
max_a0 = None

optimizer = TwoDimensionalNoiseOptimizer(a1_bounds=(min_a1, max_a1), a0_bounds=(min_a0, max_a0))

fuzzy_rec_dec = {"x_err": [],
                 "y_err": [],
                 "a_0": [],
                 "a_1": [],
                 "q": []}

for g_x in granules_dict['x_granules'][0]['adj_g'].values:

    x_err = phi_func_reversed(g_x, mu)
    intervals_x = np.ones(x_noise.shape[0])*x_err

    for g_y in granules_dict['y_granules'][0]['adj_g'].values:

        y_err = phi_func_reversed(g_y, mu)
        intervals_y = np.ones(y_noise.shape[0]) * y_err

        optimizer.fit(x_noise, intervals_x, y_noise, intervals_y, a1_steps=steps, verbose=1, force_mute=True)

        # x_rec = optimizer.x
        # y_rec = optimizer.y
        # res = optimizer.get_res()
        # print(res)

        fuzzy_rec_dec['x_err'].append(x_err)
        fuzzy_rec_dec['y_err'].append(y_err)
        fuzzy_rec_dec['a_0'].append(optimizer.a_0)
        fuzzy_rec_dec['a_1'].append(optimizer.a_1)
        fuzzy_rec_dec['q'].append(optimizer.get_q())

# OLS reconstruction


def func(x, a_1, a_0):
    return a_1*x + a_0


popt, pcov = curve_fit(func, x_noise, y_noise)
y_ols = func(x_noise, popt[0], popt[1])

for key in fuzzy_rec_dec.keys():
    print(f"{key} shape = {len(fuzzy_rec_dec[key])}")


# save data

df_points = pd.DataFrame(data={'x_true': x_true,
                               'y_true': y_true,

                               'x_noise': x_noise,
                               'y_noise': y_noise,

                               'x_err_true': intervals_x_true,
                               'y_err_true': intervals_y_true})

df_params = pd.DataFrame(data={'a_0': [a_0, popt[1]],
                               'a_1': [a_1, popt[0]]},
                         index=['true', 'ols'])

df_noise_models = pd.DataFrame(data=fuzzy_rec_dec)

df_optimizer = pd.DataFrame(data={'min': [min_a1, min_a0],
                                  'max': [max_a1, max_a0],
                                  'steps': [steps, np.nan]},
                            index=['a1', 'a0'])


writer = pd.ExcelWriter('../data/function_reconstruction_with_fuzzy_noise_model.xlsx', engine='xlsxwriter')
df_points.to_excel(writer, sheet_name='data', index=False)
df_params.to_excel(writer, sheet_name='params')
df_noise_models.to_excel(writer, sheet_name='noise_rec')
df_optimizer.to_excel(writer, sheet_name='a1_optimizer')

writer.save()
