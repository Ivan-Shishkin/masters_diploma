from lib.ishishkin.diploma.models import LinearFunctionTwoDimensionalNoise

from lib.ishishkin.diploma.plotter import plot_report_granules

from lib.ishishkin.diploma.normal_distribution_functions import norm_dence, phi_func, phi_func_reversed
from lib.ishishkin.diploma.granular import RobbinsMonroeGranular

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# granules reconstruction break conditions
early_stop = 10**5
k = 1
eps_max = np.inf
eps = 0.01

# x granules reconstruction

    # x granules number
granules_num_x = 8

    # normal distribution loc
mu_x = 0

    # probability of invalid granulation
a_x = 1 + 0.1**5


    # initial propositions
init_g_list_x = []
chi_list_x = []

for num in range(0, granules_num_x):
  init_g_list_x.append(phi_func(num*2, mu_x))
  chi_list_x.append(np.empty(0))

granulator_x = RobbinsMonroeGranular(granules_num_x, init_g_list_x, eps, early_stop, a_x)
granulator_x.normal_granular(mu_x, sample_scale=1)
x_xi_loc = granulator_x.xi_list.mean()
x_xi_scale = granulator_x.xi_list.std(ddof=1)
# save results in pandas dataframes

df_x_granules = pd.DataFrame(data={"init_g": granulator_x.init_g_list,
                                   "init_G+": phi_func_reversed(granulator_x.init_g_list, mu_x),
                                   "adjusted_g": granulator_x.g_list,
                                   "adjusted_G+": phi_func_reversed(granulator_x.g_list, mu_x),
                                   "mes_g": granulator_x.granulas_mes})

df_x_noise_values = pd.DataFrame(data={"x_noise": granulator_x.xi_list})
df_x_noise_moments = pd.DataFrame(data={"loc": x_xi_loc,
                                        "scale": x_xi_scale},
                                  index=[0])
# y granules reconstruction

    # y granules number
granules_num_y = 8

    # normal distribution loc
mu_y = 0

    # probability of invalid granulation
a_y = 1 + 0.1**5


    # initial propositions
init_g_list_y = []
chi_list_y = []

for num in range(0, granules_num_y):
  init_g_list_y.append(phi_func(num*0.1, mu_y))
  chi_list_y.append(np.empty(0))

granulator_y = RobbinsMonroeGranular(granules_num_y, init_g_list_y, eps, early_stop, a_y)
granulator_y.normal_granular(mu_y, sample_scale=1)
y_xi_loc = granulator_y.xi_list.mean()
y_xi_scale = granulator_y.xi_list.std(ddof=1)
# save results in pandas dataframes

df_y_granules = pd.DataFrame(data={"init_g": granulator_y.init_g_list,
                                   "init_G+": phi_func_reversed(granulator_y.init_g_list, mu_y),
                                   "adjusted_g": granulator_y.g_list,
                                   "adjusted_G+": phi_func_reversed(granulator_y.g_list, mu_y),
                                   "mes_g": granulator_y.granulas_mes})

df_y_noise_values = pd.DataFrame(data={"y_noise": granulator_y.xi_list})
df_y_noise_moments = pd.DataFrame(data={"loc": y_xi_loc,
                                        "scale": y_xi_scale},
                                  index=[0])

# linear function object

a_0 = 10
a_1 = 1

df_true_func = pd.DataFrame(data={'a_0_true': a_0, 'a_1_true': a_1}, index=[0])

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

y_noise[0] = y_true[0] + 2
y_noise[-1] = y_true[-1] - 2

x_noise[0] = x_true[0] - 2
x_noise[-1] = x_true[-1] + 2

# OLS reconstruction


def func(x, a_1, a_0):
    return a_1*x + a_0


popt, pcov = curve_fit(func, x_noise, y_noise)
y_ols = func(x_noise, popt[0], popt[1])


sigma_y = np.ones(y_noise.shape[0])*(3*scale_y)
popt_with_errs, pcov_with_errs = curve_fit(func, x_noise, y_noise, sigma=1./(sigma_y*sigma_y))
y_ols_with_errs = func(x_noise, popt_with_errs[0], popt_with_errs[1])


# save_data
df_points = pd.DataFrame(data={'x_true': x_true,
                               'y_true': y_true,

                               'x_noise': x_noise,
                               'y_noise': y_noise,

                               'x_err_true': intervals_x_true,
                               'y_err_true': intervals_y_true,

                               'x_err_point': x_true-x_noise,
                               'y_err_point': y_true-y_noise,})
#
df_ols_params = pd.DataFrame(data={'a_0': [popt[1], popt_with_errs[1]],
                                   'a_1': [popt[0], popt_with_errs[0]]},
                             index=['ols', 'ols_with_errs'])

df_ols_points = pd.DataFrame(data={'x_ols': x_noise,
                                   'y_ols': y_ols,
                                   'y_ols_with_errs': y_ols_with_errs})

writer = pd.ExcelWriter('../data/granules_report.xlsx', engine='xlsxwriter')

df_x_granules.to_excel(writer, sheet_name="x_granules", index=False)

df_x_noise_values.to_excel(writer, sheet_name="x_noise", index=False)
df_x_noise_moments.to_excel(writer, sheet_name="x_noise", index=False, startrow=1, startcol=2)

df_y_granules.to_excel(writer, sheet_name="y_granules", index=False)

df_y_noise_values.to_excel(writer, sheet_name="y_noise", index=False)
df_y_noise_moments.to_excel(writer, sheet_name="y_noise", index=False, startrow=1, startcol=2)

df_points.to_excel(writer, sheet_name='data', index=False)
df_true_func.to_excel(writer, sheet_name='data', index=False, startrow=0, startcol=10)

df_ols_params.to_excel(writer, sheet_name='ols_params')

df_ols_points.to_excel(writer, sheet_name='ols_data')




# graphs


    # x noise distr

fig, ax = plt.subplots(figsize=(12,5))
sns.distplot(granulator_x.xi_list, bins=100, ax=ax)
ax.grid()
plt.savefig('../data/img/x_noise_distplot.png')

x_noise_ws = writer.sheets['x_noise']
x_noise_ws.insert_image('C5', '../data/img/x_noise_distplot.png')

    # y noise distr

fig, ax = plt.subplots(figsize=(12,5))
sns.distplot(granulator_y.xi_list, bins=100, ax=ax)
ax.grid()
plt.savefig('../data/img/y_noise_distplot.png')

y_noise_ws = writer.sheets['y_noise']
y_noise_ws.insert_image('C5', '../data/img/y_noise_distplot.png')

    # x granules

plot_report_granules(df_x_granules, '../data/img/x_granules.png')

x_granules_ws = writer.sheets['x_granules']
x_granules_ws.insert_image('F2', '../data/img/x_granules.png')

    # y granules

plot_report_granules(df_y_granules, '../data/img/y_granules.png')

y_granules_ws = writer.sheets['y_granules']
y_granules_ws.insert_image('F2', '../data/img/y_granules.png')

    # true points

fig, ax = plt.subplots(figsize=(15,10))

plt.errorbar(df_points["x_noise"],
             df_points["y_noise"],
             xerr=df_points["x_err_true"],
             yerr=df_points["y_err_true"],
             fmt='.k',
             label="measured",
             mfc='red',
             markersize=10, capsize=2)
plt.plot(df_points["x_true"],
         df_points["y_true"],
         label=f"$y_t = {a_1} \cdot x + {a_0}$")
plt.grid()
ax.legend(fontsize=15, loc="upper left")
plt.savefig('../data/img/true_vs_measured_points.png')

true_points_ws = writer.sheets['data']
true_points_ws.insert_image('H4', '../data/img/true_vs_measured_points.png')

    # ols points

fig, ax = plt.subplots(figsize=(15,10))

plt.errorbar(df_points["x_noise"],
             df_points["y_noise"],
             xerr=df_points["x_err_true"],
             yerr=df_points["y_err_true"],
             fmt='.k',
             label="measured",
             mfc='red',
             markersize=10, capsize=2)
plt.plot(df_points["x_true"],
         df_points["y_true"],
         label=f"$y_t = {a_1} \cdot x + {a_0}$")
plt.plot(x_noise,
         y_ols,
         label=f"$y_o = {popt[0]: .2f} \cdot x + {popt[1]: .2f}$")

plt.grid()
ax.legend(fontsize=15, loc="upper left")
plt.savefig('../data/img/true_vs_ols_points.png')

ols_points_ws = writer.sheets['ols_data']
ols_points_ws.insert_image('H4', '../data/img/true_vs_ols_points.png')

writer.save()
