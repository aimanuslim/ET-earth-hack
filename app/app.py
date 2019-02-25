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

wellnames = [
    'A10',
    'A11',
    'A12',
    'B8',
    'C3',
    'C4',
    'G1',
    'G2',
    'G3',
]

algorithms = [
    'Gradient boosting',
    'Linear regression',
    'Random forrest',
]


def load_all_data():
    well_dict = {}
    for well in wellnames:
        algo_dict = {}
        for algo in algorithms:
            respath = '../Results/Cheal-' + well + '/' + algo + '.csv'
            algo_dict[algo] = readres(respath)
        in_out_dict = {}
        path = '../clean/Cheal-' + well + '_Clean.csv'
        in_out_dict['in'] = dictdata(path)
        in_out_dict['out'] = algo_dict
        well_dict[well] = in_out_dict
    return well_dict


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



fig = tools.make_subplots(rows=1, cols=len(input_col_labels),
                        subplot_titles=tuple(input_col_labels),
                        shared_yaxes=True
                          )

well_dict = load_all_data()


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


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
                    {'label': 'Gradient Boosting', 'value': 'Gradient boosting'},
                    {'label': 'Linear regression', 'value': 'Linear regression'},
                    {'label': 'Random forest', 'value': 'Random forrest'},
                ],
            value='Gradient boosting'
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
    ])
])



@app.callback(dash.dependencies.Output('well-plot', 'figure'),
            [
            dash.dependencies.Input(component_id='well-dropdown',  component_property='value'),
            dash.dependencies.Input(component_id='algo-dropdown',  component_property='value')
            ])
def plot(wellname, algoname):
    # path = '../clean/Cheal-'+wellname+'_Clean.csv'
    # data = dictdata(path)
    # respath = '../Results/Cheal-'+wellname+'/'+algoname+'.csv'
    # pred_data = readres(respath)

    col_idx = 1
    data = well_dict[wellname]['in']
    pred_data = well_dict[wellname]['out'][algoname]

    fig.data = []
    trace_num = 1
    for sbp in input_col_curves:
        for label in sbp:
            if label == 'Pred_DTC':
                fig.add_trace(get_curve(pred_data[0], pred_data[1], label, trace_num), 1, col_idx)
            else:
                fig.add_trace(get_curve(data[label][0], data[label][1], label, trace_num), 1, col_idx)
            trace_num += 1
        col_idx += 1

    ylist = [d['y'] for d in fig.data]
    ylist = np.array(ylist).flatten()
    newl = []
    for y in ylist:
        for d in y:
            newl.append(float(d))

    maxy = np.max(newl)
    miny = np.min(newl)

    fig['layout']['xaxis2'].update(
        type='log',
    )




    fig['layout'].update(
        xaxis=dict(
            hoverformat = '.2f',
            zeroline=False
            ),
        yaxis=dict(
            hoverformat = '.2f',
            showspikes=True,
            spikedash='solid',
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            # autorange='reversed',
            zeroline=False,
            tickmode='linear',
            range=[maxy, miny]
            ),
        yaxis2=dict(
            range=[maxy, miny]
        ),
        yaxis3=dict(
            range=[maxy, miny]
        ),
        yaxis4=dict(
            range=[maxy, miny]
        ),
        yaxis5=dict(
            range=[maxy, miny]
        ),
        yaxis6=dict(
            range=[maxy, miny]
        ),
        yaxis7=dict(
            range=[maxy, miny]
        ),
        yaxis8=dict(
            range=[maxy, miny]
        ),
        yaxis9=dict(
            range=[maxy, miny]
        ),

        hovermode= 'closest',
    )
    return fig



@app.callback(dash.dependencies.Output(component_id='pred-score',  component_property='children'),
            [dash.dependencies.Input(component_id='well-dropdown',  component_property='value'),
            dash.dependencies.Input(component_id='algo-dropdown',  component_property='value')
            ]
            )
def get_well_score(wellname,algoname):
    respath = '../Results/Cheal-'+wellname+'/'+algoname+'_score.csv'
    df = pd.read_csv(respath)
    error = df['ERROR'][0]
    return "R² score: {:2f}".format(error)

app.config.supress_callback_exceptions = True

if __name__ == '__main__':
    app.run_server(debug=False, threaded=True, host='0.0.0.0')

