import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
import numpy as np
from lib.ishishkin.diploma.normal_distribution_functions import norm_dence


def plot_q(df_data, df_params):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_true'],
                             name='$y_t= {a_1: .3f} \cdot x + {a_0: .3f}$'.format(a_1=df_params.loc['true']['a_1'],
                                                                                  a_0=df_params.loc['true']['a_0']),
                             mode='lines'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_noise'],
                             error_y=dict(type='data', array=df_data['y_err'], visible=True),
                             name='measured points',
                             mode='markers'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_true'],
                             name='true points',
                             mode='markers'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_rec'],
                             name='$y_t= {a_1: .3f} \cdot x + {a_0: .3f}$'.format(
                                 a_1=df_params.loc['reconstructed']['a_1'],
                                 a_0=df_params.loc['reconstructed']['a_0']),
                             mode='lines'))

    fig.update_layout(
        title=f"Модель равноточных измерений в линейной схеме. Адекватная модель<br>\t q = {df_params.loc['reconstructed']['q']: .4f}")
    fig.show()


def plot_mse(df_data, df_params):
    ols_mse = mean_squared_error(df_data['y_true'], df_data['y_ols'])
    rec_mse = mean_squared_error(df_data['y_true'], df_data['y_rec'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_true'],
                             name='$y_t= {a_1: .3f} \cdot x + {a_0: .3f}$'.format(a_1=df_params.loc['true']['a_1'],
                                                                                  a_0=df_params.loc['true']['a_0']),
                             mode='lines'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_noise'],
                             error_y=dict(type='data', array=df_data['y_err'], visible=True),
                             name='measured points',
                             mode='markers'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_rec'],
                             name='$y_r= {a_1: .3f} \cdot x + {a_0: .3f}$'.format(
                                 a_1=df_params.loc['reconstructed']['a_1'],
                                 a_0=df_params.loc['reconstructed']['a_0']),
                             mode='lines'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_ols'],
                             name='$y_o= {a_1: .3f} \cdot x + {a_0: .3f}$'.format(a_1=df_params.loc['ols']['a_1'],
                                                                                  a_0=df_params.loc['ols']['a_0']),
                             mode='lines'))

    fig.add_trace(go.Scatter(x=df_data['x_true'],
                             y=df_data['y_true'],
                             name='true points',
                             mode='markers'))

    fig.update_layout(
        title=f"Модель равноточных измерений в линейной схеме<br>\tMSE(OLS, true) = {ols_mse: .2f}<br>\tMSE(rec, true) = {rec_mse: .2f}")
    fig.show()


def plot_report_granules(df_granules, path_to_graph):

    x_max = df_granules["adjusted_G+"].values[-1] * 1.5

    x_graph = np.arange(-x_max, x_max + 0.01, 0.01)
    y_graph = norm_dence(x_graph, mu=0, sigma=1)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5))

    sns.lineplot(x=x_graph, y=y_graph, label='true distr', ax=ax[0])
    sns.lineplot(x=x_graph, y=y_graph, label='true distr', ax=ax[1])

    G_prev_adj = 0.0
    G_prev_init = 0.0

    for i, row in df_granules.iterrows():

        G_i_adj = row["adjusted_G+"]
        G_i_init = row["init_G+"]

        ax[0].add_patch(patches.Rectangle((-G_i_init, 0), (G_i_init - G_prev_init), 0.4, alpha=0.08 * i))
        ax[0].add_patch(patches.Rectangle((G_i_init, 0), (G_prev_init - G_i_init), 0.4, alpha=0.08 * i))
        ax[0].axvline(x=-G_i_init, ymin=0, ymax=1)
        ax[0].axvline(x=G_i_init, ymin=0, ymax=1)

        ax[1].add_patch(patches.Rectangle((-G_i_adj, 0), (G_i_adj - G_prev_adj), 0.4, alpha=0.08 * i))
        ax[1].add_patch(patches.Rectangle((G_i_adj, 0), (G_prev_adj - G_i_adj), 0.4, alpha=0.08 * i))
        ax[1].axvline(x=-G_i_adj, ymin=0, ymax=1)
        ax[1].axvline(x=G_i_adj, ymin=0, ymax=1)

        G_prev_adj = G_i_adj
        G_prev_init = G_i_init

    ax[0].grid()
    ax[0].set_title("init granules")

    ax[1].grid()
    ax[1].set_title("adjusted granules")

    plt.savefig(path_to_graph)