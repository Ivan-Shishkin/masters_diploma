import pandas as pd
import numpy as np
from lib.ishishkin.diploma.linprog_optimizer import OneDimensionalNoiseOptimizer
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# read data about granules
df_points = pd.read_excel("../data/granules_report.xlsx", sheet_name="data", usecols=range(6))

# x_noise = df_points["x_noise"].values
y_noise = df_points["y_noise"].values

x_true = df_points["x_true"].values
y_true = df_points["y_true"].values

for i, x_i in enumerate(x_true):
    if i < 3:
        y_noise[i] = y_true[i] + 1.

    if i > 7:
        y_noise[i] = y_true[i] - 1.


df_y_granules = pd.read_excel("../data/granules_report.xlsx", sheet_name="y_granules", usecols=range(6))
adj_y_err = df_y_granules["adjusted_G+"].iloc[-1]
print(f"y_err = {adj_y_err}")
intervals_y = np.ones(y_noise.shape[0])*1.2


##########################################################

# reconstruct function

min_a1 = 0.8
max_a1 = 1.3
steps = 5

min_a0 = None
max_a0 = None

optimizer = OneDimensionalNoiseOptimizer()
optimizer.fit(x_true, y_noise, intervals_y, verbose=0)

y_rec = optimizer.predict(x_true)
a_0 = optimizer.a_0
a_1 = optimizer.a_1
res = optimizer.get_res()
q = optimizer.get_q()
print(res)

#########################################################

# ols

def func(x, a_1, a_0):
    return a_1*x + a_0

popt, pcov = curve_fit(func, x_true, y_noise)
y_ols = func(x_true, popt[0], popt[1])

a_1_ols = popt[0]
a_0_ols = popt[1]

df_rec = pd.DataFrame(data={"x_rec": x_true,
                            "y_rec": y_rec,

                            "y_noise": y_noise,

                            "y_err": intervals_y,

                            "x_true": x_true,
                            "y_true": y_true,

                            "x_ols": x_true,
                            "y_ols": func(x_true, a_1_ols, a_0_ols)
                            })

df_params = pd.DataFrame(data={"a_0": [a_0, a_0_ols],
                               "a_1": [a_1, a_1_ols],
                               "q": [q, np.nan],
                               "mse": [mean_squared_error(y_true, y_rec), mean_squared_error(y_true, y_ols)]},
                         index=['rec', 'ols'])

writer = pd.ExcelWriter('../data/reconstruction_report_one_dim_eq_imposible.xlsx', engine='xlsxwriter')
df_rec.to_excel(writer, sheet_name="reconstructed", index=False)
df_params.to_excel(writer, sheet_name="params")



fig, ax = plt.subplots(figsize=(15,10))

plt.errorbar(df_rec["x_true"],
             df_rec["y_rec"],
             yerr=df_rec["y_err"],
             fmt='.k',
             label="reconstructed points",
             mfc='red',
             markersize=10, capsize=2)

plt.plot(df_rec["x_true"],
         df_rec["y_rec"],
         label=f"$y_r = {a_1} \cdot x + {a_0}$")

plt.plot(df_rec["x_true"],
         df_rec["y_noise"],
         'o',
         label=f"measured_points")

plt.plot(df_rec['x_true'],
         df_rec['y_true'],
         '--',
         label=f"$y_t = 1 \cdot x + 10$")

plt.plot(x_true,
         func(x_true, a_1_ols, a_0_ols),
         label=f"$y_o = {a_1_ols} \cdot x + {a_0_ols}$")



plt.grid()
ax.legend(fontsize=15, loc="upper left")
plt.savefig('../data/img/rec_func.png')

ols_points_ws = writer.sheets['reconstructed']
ols_points_ws.insert_image('K4', '../data/img/rec_func.png')


writer.save()

