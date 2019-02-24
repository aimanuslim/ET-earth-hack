import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


input_col_labels = ['General', 'Resistivity', 'Density', 'Measured DTC', 'Predicted DTC', 'Error']

input_col_curves = [
    ['GR', 'BS', 'CALI', 'SP'],
    ['RESS', 'RESM', 'RESD'],
    ['DENS', 'NEUT', 'PEF'],
    # ['DTC_act'],
    # ['DTC_pred'],
    # ['Error']
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

def get_curve(xdata, ydata, dataname, tn):
    return go.Scatter(
        x=xdata,
        y=ydata,
        name=dataname,
        mode="lines",
        hoverinfo='x',
        xaxis='x' + str(tn)
    )



def dictdata(filepath):
    curvedict = {}
    df = pd.read_csv(filepath).replace('-9999',np.NaN).iloc[1:]
    for curve in list(df):
        if curve != 'DEPT':
            curvedict[curve] = (df[curve].tolist(),df['DEPT'].tolist())
    return curvedict

data = dictdata('../data/Cheal-B8_Clean.csv')
# data = load_csv()
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

trace_num = 1
for sbp in input_col_curves:
    for label in sbp:
        fig.add_trace(get_curve(data[label][0], data[label][1], label, trace_num), 1, col_idx)
        trace_num += 1
    col_idx += 1

# col_idx = 1
# for sbp in output_col_curves:
#     for label in sbp:
#         fig_out.add_trace(get_curve(output_data[label][0], output_data[label][1], label), 1, col_idx)
#     col_idx += 1

fig['layout']['xaxis2'].update(
    type='log',
    # title='x2dgdsf',

)

fig['layout']['xaxis4'].update(
    overlaying='x1',
    showgrid=False,
    title='second curve'
)

fig['layout'].update(
    xaxis=dict(
        title='x1',
        hoverformat = '.2f',
        zeroline=False
        ),
    # xaxis2=dict(
    #         title= 'xaxis2 title',
    #         overlaying='x',
    #         side='top'
    # ),
    yaxis=dict(
        hoverformat = '.2f',
        showspikes=True,
        spikedash='solid',
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        autorange='reversed',
        zeroline=False,
        tickmode='linear',
        ),
    hovermode= 'closest',
    # legend=dict(orientation="h", y=1.05)
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
    html.H1('7logprod'),
    html.Div([
        html.Div([
            dcc.Dropdown(
            id='well-dropdown',
            options=[
                {'label': 'Cheal-A10', 'value': '../data/prep/Cheal-A10_Clean.csv'},
                {'label': 'Cheal-A11', 'value': '../data/prep/Cheal-A11_Clean.csv'},
                {'label': 'Cheal-A12', 'value': '../data/prep/Cheal-A12_Clean.csv'},
                {'label': 'Cheal-B8', 'value': '../data/prep/Cheal-B8_Clean.csv'},
                {'label': 'Cheal-C3', 'value': '../data/prep/Cheal-C3_Clean.csv'},
                {'label': 'Cheal-C4', 'value': '../data/prep/Cheal-C4_Clean.csv'},
                {'label': 'Cheal-G1', 'value': '../data/prep/Cheal-G1_Clean.csv'},
                {'label': 'Cheal-G2', 'value': '../data/prep/Cheal-G2_Clean.csv'},
                {'label': 'Cheal-G3', 'value': '../data/prep/Cheal-G3_Clean.csv'}
            ],
            value='../data/prep/Cheal-B8_Clean.csv',
        ),
        ],
        style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            dcc.Dropdown(
                id='algo-dropdown',
                options=[
                    {'label': 'Gradient Boosting Regressor', 'value': '../data/{}/GBRscores.txt'},
                    {'label': 'Random Forest Regressor', 'value': ''},
                    {'label': 'Algo3', 'value': 'asd'},
                ],
                value='',
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}
        ),
    ]),
    html.Div([
       html.H3(id='xval-score',
               style={'width': '49%', 'display': 'inline-block', 'text-align': 'center'},
               children="Cross validation score: {:2f}".format(np.random.randint(0,565)),
               ),
        html.H3(id='pred-score',
                style={'width': '49%', 'display': 'inline-block', 'text-align': 'center'},
                children="Well score: {:2f}".format(np.random.randint(0,565)),
                ),
    ]),
    html.Div([
        dcc.Graph(
            figure=fig,
            style={'height': 99999},
        ),
        # dcc.Graph(
        #     figure=fig_out,
        #     style={'height': 1900, 'width': '49%', 'display': 'inline-block'},
        # )
    ])
])

@app.callback(dash.dependencies.Output(component_id='xval-score', component_property='children'),
            [dash.dependencies.Input(component_id='well-dropdown',  component_property='value')])
def get_xval_score(wellname):
    return "Cross validation score: {:2f}".format(np.random.randint(0,565))

@app.callback(dash.dependencies.Output(component_id='pred-score',  component_property='children'),
            [dash.dependencies.Input(component_id='well-dropdown',  component_property='value')])
def get_well_score(wellname):
    return "Well validation score: {:2f}".format(np.random.randint(0,565))

if __name__ == '__main__':
    app.run_server(debug=True)
