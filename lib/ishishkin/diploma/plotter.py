import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


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
