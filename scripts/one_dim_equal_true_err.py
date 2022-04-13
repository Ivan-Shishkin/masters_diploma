from lib.ishishkin.diploma.models import LinearFunctionOneDimensionalNoise
from lib.ishishkin.diploma.linprog_optimizer import OneDimensionalNoiseOptimizer

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit

# linear function object

a_0 = 1
a_1 = 1

model_1 = LinearFunctionOneDimensionalNoise(a_0=a_0, a_1=a_1)

# generating samples

min_x = 0
max_x = 31
step_x = 1

x_true = model_1.make_true_x(min_x, max_x, step_x)
y_true = model_1.make_true_y(min_x, max_x, step_x)
y_noise, intervals_y = model_1.uni_normal_noise(min_x, max_x, step_x, 0, 1)

# reconstruct function

optimizer = OneDimensionalNoiseOptimizer()
optimizer.fit(x_true, y_noise, intervals_y, verbose=0)

y_rec = optimizer.predict(x_true)
res = optimizer.get_res()
print(type(res))

# OLS reconstruction

def func(x, a_1, a_0):
    return a_1*x + a_0


popt_with_errs, pcov_with_errs = curve_fit(func, x_true, y_noise, sigma=1./(intervals_y*intervals_y))
y_ols_with_errs = func(x_true, popt_with_errs[0], popt_with_errs[1])

popt, pcov = curve_fit(func, x_true, y_noise)
y_ols = func(x_true, popt[0], popt[1])

# save data

df_points = pd.DataFrame(data={'x_true': x_true,
                               'y_true': y_true,
                               'y_noise': y_noise,
                               'y_err': intervals_y,
                               'y_rec': y_rec,
                               'y_ols': y_ols,
                               'y_ols_with_errs': y_ols_with_errs})

df_params = pd.DataFrame(data={'a_0': [a_0, res['x'][2], popt[1], popt_with_errs[1]],
                               'a_1': [a_1, res['x'][1], popt[0], popt_with_errs[0]],
                               'q': [np.nan, res['x'][0], np.nan, np.nan]},
                         index=['true', 'reconstructed', 'ols', 'ols_with_errs'])


writer = pd.ExcelWriter('../data/one_dimensional_equal_true_errs.xlsx', engine='xlsxwriter')
df_points.to_excel(writer, sheet_name='data', index=False)
df_params.to_excel(writer, sheet_name='params')

writer.save()