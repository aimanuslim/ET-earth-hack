import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
import pandas as pd
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


input_col_labels = ['General', 'Resistivity', 'Density', 'DTC Comparison']

input_col_curves = [
    ['GR', 'BS', 'CALI'],
    ['RESS', 'RESM', 'RESD'],
    ['DENS', 'NEUT', 'PEF'],
    ['Pred_DTC', 'DTC']
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
        hoverinfo='x+name',
        xaxis='x' + str(tn)
    )



def dictdata(filepath):
    curvedict = {}
    df = pd.read_csv(filepath).replace('-9999',np.NaN).iloc[1:]
    for curve in list(df):
        if curve != 'DEPT':
            curvedict[curve] = (df[curve].tolist(),df['DEPT'].tolist())
    return curvedict

def readres(filepath):
    df = pd.read_csv(filepath).replace('-9999',np.NaN).iloc[1:]
    pred_data  = (df['Pred_DTC'].tolist(),df['DEPT'].tolist())
    return pred_data


# def clean():
#     curvedict = {}
#     df1 = pd.read_csv('../clean/Cheal-A10_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df2 = pd.read_csv('../clean/Cheal-A11_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df3 = pd.read_csv('../clean/Cheal-A12_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df4 = pd.read_csv('../clean/Cheal-B8_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df5 = pd.read_csv('../clean/Cheal-C3_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df6 = pd.read_csv('../clean/Cheal-C4_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df7 = pd.read_csv('../clean/Cheal-G1_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df8 = pd.read_csv('../clean/Cheal-G2_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df9 = pd.read_csv('../clean/Cheal-G3_Clean.csv').replace('-9999',np.NaN).iloc[1:]
#     df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9])
#     for curve in list(df):
#         if curve != 'DEPT':
#             curvedict[curve] = (df[curve].tolist(),df['DEPT'].tolist())
#     return curvedict

# def newdict():
#     newcurvedict = {}
#     oripath = "../data/clean"
#     for root, dirs, files in os.walk(oripath):
#         for file in files:
#             if file.endswith(".csv"):
#                 laspath = os.path.join(root, file)
#                 curvedict = {}
#                 df = pd.read_csv(laspath).replace('-9999',np.NaN).iloc[1:]
#                 name = df['wellName'][1]
#                 for curve in list(df):
#                     if curve != 'DEPT':
#                         curvedict[curve] = (df[curve].tolist(),df['DEPT'].tolist())
#                 newcurvedict[name] = curvedict[curve]
#     return newcurvedict

# data = newdict()
# data = clean()
# data = dictdata('../data/prep/Cheal-B8_Clean.csv')
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
# col_idx = 1
#
# trace_num = 1
# for sbp in input_col_curves:
#     for label in sbp:
#         fig.add_trace(get_curve(data[label][0], data[label][1], label, trace_num), 1, col_idx)
#         trace_num += 1
#     col_idx += 1

# col_idx = 1
# for sbp in output_col_curves:
#     for label in sbp:
#         fig_out.add_trace(get_curve(output_data[label][0], output_data[label][1], label), 1, col_idx)
#     col_idx += 1

# fig['layout']['xaxis2'].update(
#     type='log',
#     # title='x2dgdsf',
#
# )

# fig['layout']['xaxis4'].update(
#     overlaying='x1',
#     showgrid=False,
#     title='second curve'
# )

# fig['layout'].update(
#     xaxis=dict(
#         title='x1',
#         hoverformat = '.2f',
#         zeroline=False
#         ),
#     # xaxis2=dict(
#     #         title= 'xaxis2 title',
#     #         overlaying='x',
#     #         side='top'
#     # ),
#     yaxis=dict(
#         hoverformat = '.2f',
#         showspikes=True,
#         spikedash='solid',
#         spikemode='across',
#         spikesnap='cursor',
#         spikethickness=1,
#         autorange='reversed',
#         zeroline=False,
#         tickmode='linear',
#         ),
#     hovermode= 'closest',
#     # legend=dict(orientation="h", y=1.05)
# )

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
                    {'label': 'Cheal-A10', 'value': 'A10'},
                    {'label': 'Cheal-A11', 'value': 'A11'},
                    {'label': 'Cheal-A12', 'value': 'A12'},
                    {'label': 'Cheal-B8', 'value': 'B8'},
                    {'label': 'Cheal-C3', 'value': 'C3'},
                    {'label': 'Cheal-C4', 'value': 'C4'},
                    {'label': 'Cheal-G1', 'value': 'G1'},
                    {'label': 'Cheal-G2', 'value': 'G2'},
                    {'label': 'Cheal-G3', 'value': 'G3'}
                ],
            value='B8'
            ),

        ],
        style={'width':'49%', 'display':'inline-block'}
        ),
        html.Div([
            dcc.Dropdown(
                id='algo-dropdown',
                options=[
                    {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
                    {'label': 'Linear regression', 'value': 'Linear regression'},
                    {'label': 'Random forest', 'value': 'Random forrest'},
                ],
            value='Gradient Boosting'
            ),

        ],
        style={'width':'49%', 'display':'inline-block'}
        ),
    ]),
    html.Div([
        html.H3(id='pred-score',
                style={'text-align': 'center'},
                children="Well score: {:2f}".format(np.random.randint(0,565)),
                ),
    ]),
    html.Div([
        dcc.Graph(
            figure=fig,
            style={'height': 50000},
            id='well-plot',
        ),
        # dcc.Graph(
        #     figure=fig_out,
        #     style={'height': 1900, 'width': '49%', 'display': 'inline-block'},
        # )
    ])
])



@app.callback(dash.dependencies.Output('well-plot', 'figure'),
            [
            dash.dependencies.Input(component_id='well-dropdown',  component_property='value'),
            dash.dependencies.Input(component_id='algo-dropdown',  component_property='value')
            ])
def plot(wellname, algoname):
    path = '../clean/Cheal-'+wellname+'_Clean.csv'
    data = dictdata(path)
    respath = '../Results/Cheal-'+wellname+'/'+algoname+'.csv'
    pred_data = readres(respath)
    col_idx = 1

    trace_num = 1
    for sbp in input_col_curves:
        for label in sbp:
            if label == 'Pred_DTC':
                fig.add_trace(get_curve(pred_data[0], pred_data[1], label, trace_num), 1, col_idx)
            else:
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


    fig['layout'].update(
        xaxis=dict(
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



@app.callback(dash.dependencies.Output(component_id='pred-score',  component_property='children'),
            [dash.dependencies.Input(component_id='well-dropdown',  component_property='value'),
            dash.dependencies.Input(component_id='algo-dropdown',  component_property='value')
            ]
            )
def get_well_score(wellname,algoname):
    respath = '../Results/Cheal-'+wellname+'/'+algoname+'_score.csv'
    df = pd.read_csv(respath)
    error = df['ERROR'][0]
    print(error)
    return "R² score: {:2f}".format(error)

if __name__ == '__main__':
    app.run_server(debug=True)
