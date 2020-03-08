import pandas as pd
import numpy as np
import dash
import flask
import math
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_table
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

app.layout=html.Div(children=[
            html.H1(id='header',children=['Linear Regression Demonstration'],style={
                              'textAlign': 'center',
                              'color': 'blue',
                              'fontSize':50}),
            html.Div(id='instructions',
                     children=['Enter coordinates in table below and watch as Linear Regression model find the line of best fit',
                                                html.Br(),'You can select different metrics to see how the model performs'],
                    style={'textAlign': 'center',
                              'color': 'skyblue',
                              'fontSize':20}              
                    ),
            dcc.Graph(id='adding-rows-graph'),
            dcc.RadioItems(id='residuals',options=[{'label':'Residuals Off','value':'no'},
                                                   {'label':'Residuals On','value':'yes'}],value='no'),
            dcc.RadioItems(id='mean',options=[{'label':'Mean Off','value':'no'},
                                              {'label':'Mean On','value':'yes'}],value='no'),
            html.Br(),
            html.Div(id='metrics'),
            html.Br(),
            html.Div(id='input',children=[
                
            
                dash_table.DataTable(id='table', columns=[{'name':'x','id':'x','type':'numeric'},
                                                      {'name':'y','id':'y','type':'numeric'}],
                                             data=[{'x':5,'y':6}],
                                             editable=True,
                                             row_deletable=True),
                html.Button('Add Row', id='editing-rows-button', n_clicks=0)
                    ])
            
])   

@app.callback(
    Output('table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('table', 'data'),
     State('table', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows 
@app.callback(
    [Output('adding-rows-graph', 'figure'),
     Output('metrics','children')],
    [Input('table', 'data'),
     Input('table', 'columns'),
     Input('residuals','value'),
     Input('mean','value')])
def display_output(rows, columns,resid,mean):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    df['rd']=df.x.apply(lambda x: int(round(float(x),0)))
    x=[i for i in range(df.rd.max()+2)]
    mod=LinearRegression().fit(df[['x']],df[['y']])
    df['pred']=mod.predict(df[['x']])
    r2=mod.score(df[['x']],df[['y']])
    mae=mean_absolute_error(df[['y']],df[['pred']])
    rmse=mean_squared_error(df[['y']],df[['pred']])**0.5
    if mean=='yes':
        mn=df.y.mean()
    else:
        mn=[]
    children=['R^2 Score: ' + str(r2),html.Br(),'Mean Absolute Error: ' +str(mae),html.Br(),'Root Mean Squared Error: '+str(rmse)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=[i for i in df['x']],
        y=[j for j in df['y']],
        mode='markers',
        showlegend=False
    )),
    fig.add_trace(go.Scatter(
        name='Regression Line',
        x=[i for i in x],
        y=[mod.intercept_[0]+mod.coef_[0][0]*x[i] for i in x],
       
    )),
    fig.add_trace(go.Scatter(
        name='Mean Line',
        x=[i for i in x],
        y=[mn for i in range(df.rd.max()+2)],
       
    )),
    if resid=='yes':
        for index in range(len(df.x)):
            if df.y[index]>df.pred[index]:
                fig.add_shape(
                    dict(type='line',x0=df.x[index],y0=df.pred[index],x1=df.x[index],y1=df.y[index]))
            else:
                fig.add_shape(
                    dict(type='line',x0=df.x[index],y0=df.y[index],x1=df.x[index],y1=df.pred[index]))   
      
    return fig,children
                                   
                                   
            

application=app.server
if __name__ == '__main__':
    
    application.run(debug=False)
