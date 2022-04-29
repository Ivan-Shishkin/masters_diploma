import pandas as pd
import numpy as np
from lib.ishishkin.diploma.linprog_optimizer import TwoDimensionalNoiseOptimizer
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# read data about granules
df_points = pd.read_excel("../data/granules_report.xlsx", sheet_name="data", usecols=range(6))

x_noise = df_points["x_noise"].values
y_noise = df_points["y_noise"].values

x_true = df_points["x_true"].values
y_true = df_points["y_true"].values

for i, x_i in enumerate(x_true):
    if i < 5:
        x_noise[i] = x_true[i] - 1.
        y_noise[i] = y_true[i] + 1.

    if i >5:
        x_noise[i] = x_true[i] + 1.
        y_noise[i] = y_true[i] - 1.

df_x_granules = pd.read_excel("../data/granules_report.xlsx", sheet_name="x_granules", usecols=range(6))
adj_x_err = df_x_granules["adjusted_G+"].iloc[-1]
print(f"x_err = {adj_x_err}")
intervals_x = np.ones(x_noise.shape[0])*adj_x_err

df_y_granules = pd.read_excel("../data/granules_report.xlsx", sheet_name="y_granules", usecols=range(6))
adj_y_err = df_y_granules["adjusted_G+"].iloc[-1]
print(f"y_err = {adj_y_err}")
intervals_y = np.ones(y_noise.shape[0])*adj_x_err


##########################################################

# q_min = 10**5
# res_min = None
# a_1_min = np.nan
#
# for a_1 in np.arange(0.5, 1.6, 0.1):
#
#         l = np.append([1, 0], np.zeros(intervals_x.shape))
#         A = []
#         b = []
#
#         bounds_list = [(None, None), (None, None)]
#
#         for i, (x_i, sigma_i, y_i, tau_i) in enumerate(zip(x_noise, intervals_x, y_noise, intervals_y)):
#                 m_list = np.zeros(intervals_x.shape)
#                 m_list[i] = 1
#
#                 A.append(np.append([-tau_i, -1], a_1 * m_list))
#                 b.append(-y_i + a_1 * x_i)
#
#                 A.append(np.append([-tau_i, 1], -a_1 * m_list))
#                 b.append(y_i - a_1 * x_i)
#
#                 A.append(np.append([-sigma_i, 0], m_list))
#                 b.append(0)
#
#                 A.append(np.append([-sigma_i, 0], -1 * m_list))
#                 b.append(0)
#
#                 bounds_list.append((-sigma_i, sigma_i))
#
#         res = linprog(l, A_ub=A, b_ub=b, bounds=bounds_list)
#         print(f"a_1 = {a_1}")
#         print(res)
#         print('_'*500)

# reconstruct function

min_a1 = 0.8
max_a1 = 1.3
steps = 5

min_a0 = None
max_a0 = None

optimizer = TwoDimensionalNoiseOptimizer(a1_bounds=(min_a1, max_a1), a0_bounds=(min_a0, max_a0))
optimizer.fit(x_noise, intervals_x, y_noise, intervals_y, a1_steps=steps, verbose=0)

x_rec = optimizer.x
y_rec = optimizer.y
a_0 = optimizer.a_0
a_1 = optimizer.a_1
res = optimizer.get_res()
q = optimizer.get_q()
print(res)

#########################################################

# ols

def func(x, a_1, a_0):
    return a_1*x + a_0

popt, pcov = curve_fit(func, x_noise, y_noise)
y_ols = func(x_noise, popt[0], popt[1])

a_1_ols = popt[0]
a_0_ols = popt[1]

df_rec = pd.DataFrame(data={"x_rec": x_rec,
                            "y_rec": y_rec,

                            "x_noise": x_noise,
                            "y_noise": y_noise,

                            "x_err": intervals_x,
                            "y_err": intervals_y,

                            "x_true": x_true,
                            "y_true": y_true,

                            "x_ols": x_noise,
                            "y_ols": func(x_noise, a_1_ols, a_0_ols)
                            })

df_params = pd.DataFrame(data={"a_0": [a_0, a_0_ols],
                               "a_1": [a_1, a_1_ols],
                               "q": [q, np.nan],
                               "mse": [mean_squared_error(y_true, y_rec), mean_squared_error(y_true, y_ols)]},
                         index=['rec', 'ols'])

writer = pd.ExcelWriter('../data/reconstruction_report_two_dim_eq.xlsx', engine='xlsxwriter')
df_rec.to_excel(writer, sheet_name="reconstructed", index=False)
df_params.to_excel(writer, sheet_name="params")



fig, ax = plt.subplots(figsize=(15,10))

plt.errorbar(df_rec["x_rec"],
             df_rec["y_rec"],
             xerr=df_rec["x_err"],
             yerr=df_rec["y_err"],
             fmt='.k',
             label="reconstructed points",
             mfc='red',
             markersize=10, capsize=2)

plt.plot(df_rec["x_rec"],
         df_rec["y_rec"],
         label=f"$y_r = {a_1} \cdot x + {a_0}$")

plt.plot(df_rec["x_noise"],
         df_rec["y_noise"],
         'o',
         label=f"measured_points")

plt.plot(df_rec['x_true'],
         df_rec['y_true'],
         '--',
         label=f"$y_t = 1 \cdot x + 10$")

plt.plot(x_noise,
         func(x_noise, a_1_ols, a_0_ols),
         label=f"$y_o = {a_1_ols} \cdot x + {a_0_ols}$")



plt.grid()
ax.legend(fontsize=15, loc="upper left")
plt.savefig('../data/img/rec_func.png')

ols_points_ws = writer.sheets['reconstructed']
ols_points_ws.insert_image('K4', '../data/img/rec_func.png')


writer.save()

