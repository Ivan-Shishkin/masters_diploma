from lib.ishishkin.diploma.models import LinearFunctionTwoDimensionalNoise
from lib.ishishkin.diploma.linprog_optimizer import TwoDimensionalNoiseOptimizer

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit

# linear function object

a_0 = 1
a_1 = 1

model_2 = LinearFunctionTwoDimensionalNoise(a_0=a_0, a_1=a_1)

# generating samples

min_x = 0
max_x = 1100
step_x = 100

x_true = model_2.make_true_x(min_x, max_x, step_x)
y_true = model_2.make_true_y(min_x, max_x, step_x)
(x_noise, intervals_x), (y_noise, intervals_y) = model_2.normal_noise(min_x, max_x, step_x)

# reconstruct function

min_a1 = -2
max_a1 = 2
steps = 100

min_a0 = None
max_a0 = None

optimizer = TwoDimensionalNoiseOptimizer(a1_bounds=(min_a1, max_a1), a0_bounds=(min_a0, max_a0))
optimizer.fit(x_noise, intervals_x, y_noise, intervals_y, a1_steps=steps, verbose=0)

x_rec = optimizer.x
y_rec = optimizer.y
res = optimizer.get_res()
print(res)

# OLS reconstruction


def func(x, a_1, a_0):
    return a_1*x + a_0


popt, pcov = curve_fit(func, x_noise, y_noise)
y_ols = func(x_noise, popt[0], popt[1])

popt_with_errs, pcov_with_errs = curve_fit(func, x_noise, y_noise, sigma=1./(intervals_y*intervals_y))
y_ols_with_errs = func(x_noise, popt_with_errs[0], popt_with_errs[1])


# save data

df_points = pd.DataFrame(data={'x_true': x_true,
                               'y_true': y_true,

                               'x_noise': x_noise,
                               'y_noise': y_noise,

                               'x_err': intervals_x,
                               'y_err': intervals_y,

                               'x_rec': x_rec,
                               'y_rec': y_rec,

                               'y_ols': y_ols,
                               'y_ols_with_errs': y_ols_with_errs})

df_params = pd.DataFrame(data={'a_0': [a_0, optimizer.a_0, popt[1], popt_with_errs[1]],
                               'a_1': [a_1, optimizer.a_1, popt[0], popt_with_errs[0]],
                               'q': [np.nan, res['x'][0], np.nan, np.nan]},
                         index=['true', 'reconstructed', 'ols', 'ols_with_errs'])

df_optimizer = pd.DataFrame(data={'min': [min_a1, min_a0],
                                  'max': [max_a1, max_a0],
                                  'steps': [steps, np.nan]},
                            index=['a1', 'a0'])


writer = pd.ExcelWriter('../data/two_dimensional_unequal_true_errs.xlsx', engine='xlsxwriter')
df_points.to_excel(writer, sheet_name='data', index=False)
df_params.to_excel(writer, sheet_name='params')
df_optimizer.to_excel(writer, sheet_name='a1_optimizer')

writer.save()