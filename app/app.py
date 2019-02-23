import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


input_col_labels = ['General', 'Resistivity', 'Density', 'Actual', 'Predicted', 'Error']

input_col_curves = [
    ['GR', 'BS', 'CALI', 'SP'],
    ['RESS', 'RESM', 'RESD'],
    ['DENS', 'NEUT', 'PEF'],
    ['DTC_act'],
    ['DTC_pred'],
    ['Error']
]


error_curve_name = 'Error curve'


def load_csv():
    data = {}
    for sbp in input_col_curves:
        for label in sbp:
            start_depth = np.random.randint(0, 200)
            end_depth = np.random.randint(start_depth, 200)
            data[label] = (
                np.random.random(end_depth-start_depth)*0.5+np.random.randint(1,3),
                np.arange(start_depth,end_depth),
            )
    return data

# def load_dtc():
#     data = {}
#     for sbp in output_col_curves:
#         for label in sbp:
#             start_depth = np.random.randint(0, 200)
#             end_depth = np.random.randint(start_depth, 200)
#             data[label] = (
#                 np.random.random(end_depth - start_depth) * 0.5 + np.random.randint(1, 3),
#                 np.arange(start_depth, end_depth),
#             )
#     return data

def get_curve(xdata, ydata, dataname):
    return go.Scatter(
        x=xdata,
        y=ydata,
        name=dataname,
        mode="lines",
        hoverinfo='x'
    )

data = load_csv()
# output_data = load_dtc()

# fig_out = tools.make_subplots(rows=1, cols=len(output_col_labels),
#                                 subplot_titles=tuple(output_col_labels),
#                                 shared_yaxes=True
#                               )
fig = tools.make_subplots(rows=1, cols=len(input_col_labels),
                        subplot_titles=tuple(input_col_labels),
                        shared_yaxes=True
                          )
col_idx = 1
for sbp in input_col_curves:
    for label in sbp:
        fig.add_trace(get_curve(data[label][0], data[label][1], label), 1, col_idx)
    col_idx += 1

# col_idx = 1
# for sbp in output_col_curves:
#     for label in sbp:
#         fig_out.add_trace(get_curve(output_data[label][0], output_data[label][1], label), 1, col_idx)
#     col_idx += 1



fig['layout'].update(
    xaxis=dict(
        hoverformat = '.2f',
        ),
    yaxis=dict(
        hoverformat = '.2f',
        showspikes=True,
        spikedash='solid',
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        autorange='reversed',
        zeroline=False
        ),
    hovermode= 'closest',
    legend=dict(orientation="h", y=1.05)

)

# fig_out['layout'].update(
#     xaxis=dict(
#         hoverformat = '.2f',
#         ),
#     yaxis=dict(
#         hoverformat = '.2f',
#         showspikes=True,
#         spikedash='solid',
#         spikemode='across',
#         spikesnap='cursor',
#         spikethickness=1,
#         zeroline=False,
#         autorange='reversed'
#         ),
#     hovermode= 'closest',
# )

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            figure=fig,
            style={'height': 1900},
        ),
        # dcc.Graph(
        #     figure=fig_out,
        #     style={'height': 1900, 'width': '49%', 'display': 'inline-block'},
        # )
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)