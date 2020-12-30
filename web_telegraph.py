import os
import dash
import json
import time
import math
import struct
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import astropy.units as u
import plotly.express as px
from astropy.time import Time
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from collections import OrderedDict
from dash.exceptions import PreventUpdate
from pymodbus.constants import Endian
from plotly.subplots import make_subplots
from matplotlib.dates import  DateFormatter
from astropy.coordinates import FK4, ICRS, FK5
from dash.dependencies import Input, Output, State
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.payload import BinaryPayloadDecoder
from astropy import coordinates as coord, units as u
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from astropy.coordinates import SkyCoord, Angle, Latitude, Longitude, EarthLocation

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#plot forbidden zones
#Forbiden regions dictionary

f_zone = {'west': {'left_x':np.array([9.3,  8.9,  8.6,  8.2,  7.8,  7.3,  7., 
                                        6.6,  6.2,  5.6,  5. ,4.6,  4.1,  3.7, 
                                        3.3,  2.6,  1.5, 0,0]),
                    'left_y': np.array([-35, -30, -25, -20, -15, -10, -5, 0, 5,
                                         10, 15, 20, 25, 30, 35, 40, 45, 
                                         50, -35]),

                    'right_x': np.array([22.4, 21.6,21., 20.5, 20., 19.6 ,19.4, 
                                        19.,  18.5, 18.,  17.5, 17.4, 16.7 ,16.,
                                        15.4, 14.7, 13.4 , 12,  24, 24]),

                    'right_y':np.array([50,45,40,35,30,25,20,15,10,5,0,-5,-10,
                                        -15,-20,-25,-30,-35,-35, 50]),

                    'up_x':np.array([15.5,15.15,15.07,15.,15.,15.07,15.15,15.19,
                                        15.15,14.9,14.5,14.4,14.2,16.1,17.5,
                                        18.1,18.8,19.,19.4,19.9,20.3,20.6,20.95,
                                        21.,20.9,21.5,21.6]),
                    'up_y':np.array([ 90,85,80,75,70,65,60,55,50,45,40,35,30,
                                    25,30,35,40,45,50,55,60,65,70,75,80,85,90]),

                    'down_x': np. array([0,0,24,24]),

                    'down_y': np.array ([-35, -20, -20 , -35])
                        },
            'east': {    'left_x': np.array([10.8, 10,9.4, 8.8, 8.7, 8.6 , 8.2,
                                              8., 7.4 ,7. ,6.6 ,6.3 ,6.1 , 5.5, 
                                              5.2, 4.6 , 4. , 0,0]),

                            'left_y': np.array([-35, -30, -25, -20, -15, -10, 
                                                 -5, 0, 5, 10, 15, 20, 25, 30, 
                                                 35, 40, 45, 50, -35]),

                            'right_x': np.array([22.4, 21.6,21.,  20.5, 20.,  
                                                19.6 ,19.4, 19.,  18.5, 18., 
                                                17.5, 17.4, 16.7 ,16.,
                                                15.4, 14.7, 13.4 , 12,  24, 24]),

                            'right_y': np.array([50,45,40,35,30,25,20,15,10,5,
                                                0,-5,-10,-15,-20,-25,-30,-35,
                                                -35, 50]),

                            'up_x': np.array([6.,6.4,7.,7.7,8.2, 8.5 , 8.7 , 
                                              8.3, 8.4,9.2, 9.2 , 9.2 , 9. , 
                                              8.9 ,8.6 , 9.6,  9.7]),

                            'up_y': np.array([90,85,80,75,70,65,60,55,55,60,65,
                                              70,75,80,85,90,90]),

                            'down_x': np. array([0,0,24,24]),

                            'down_y': np.array ([-35, -20, -20 , -35]),
                            }
                }

FILEPATH = '/home/marina/Downloads/ra_dec_array.csv'
SUHORALATIT = 49.56917288
SUHORALONGIT= 20.06728579

app.layout = html.Div([
    dcc.Graph(id='my-graph'),
    dcc.Interval(id='main-interval', interval=2000),

    html.Div([
        html.Div(
            dcc.Textarea(id='text-field',style={'width': 350, 'height': 200},)),
            html.Button('Submit', id='submit-button', n_clicks=0),
            html.Div(id='data-div'),
            ]),

    html.Div( id="tele-info", 
        children=[
            html.H6("Telescope Information Box:"),
            html.Br(),
            html.Div(id ='info-box'),
            html.Div(id='telescope-graph')
            ])
])
#put a second output
@app.callback(
    Output('info-box', 'children'),
    Output('telescope-graph', 'children'),
    Input('tele-info', 'children'))
def update_output_div(sample_tele, orientation='west'):
    telescope1 = fake_telescope()
    return (f'The telescope coordinates are: \
            \n Hourangle: {telescope1[0]}\
            \n Declination:{telescope1[1]}\
            \n Orientation: {orientation}', telescope1)


@app.callback(
    Output('data-div', 'children'),
    Input('submit-button', 'n_clicks'),
    State('text-field', 'value'))
def update_output(n_clicks, value):
    return value

@app.callback(
    Output('my-graph', 'figure'),
    Input('main-interval', 'n_intervals'),
    Input('data-div', 'children'),
    Input('telescope-graph', 'children'))
def update_figure(n_intervals, data, telescope, orientation='west'):
    
    if orientation == 'east':
        f_zone_data = f_zone['east']
    else:
        f_zone_data = f_zone['west']

    print(telescope)
    # telescope = mount_telescope(telescope_coordinates)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x = f_zone_data['down_x'], 
                             y = f_zone_data['down_y'], 
                             fill='toself', fillcolor='yellow',
                             mode='none'),row=1,col=1)

    fig.add_trace(go.Scatter(x = f_zone_data['left_x'], 
                             y = f_zone_data['left_y'], 
                             fill='toself',fillcolor='green',
                             mode='none'),row=1,col=1)

    fig.add_trace(go.Scatter(x = f_zone_data['right_x'], 
                             y = f_zone_data['right_y'], 
                             fill='toself',fillcolor='green',
                             mode='none'))

    fig.add_trace(go.Scatter(x = f_zone_data['up_x'], 
                             y = f_zone_data['up_y'], 
                             fill='toself',fillcolor='purple',
                             mode='none')) 
    if telescope is not None:
        fig.add_trace(go.Scatter(x=[telescope[0]], y=[telescope[1]], 
                                mode="markers+text", text=['telescope']),
                                row=1, col=1)

    if data:
        input_array = read_textfield(data)
        array_for_plot = coordinates_calculations(input_array)
        # array_for_plot =  coordinates_calculations()
        fig.add_trace(go.Scatter(x=array_for_plot[:,1], 
                                 y=array_for_plot[:,2], 
                                 mode="markers+text", 
                                 text=array_for_plot[:,0]),row=1,col=1)

    fig.update_xaxes(showgrid=True, range=[0, 24])
    fig.update_yaxes(showgrid=True, range=[-30, 90])
    fig.update_traces(showlegend=False)
    plot_time = datetime.datetime.utcnow()
    plot_time = plot_time.strftime('%d-%m-%Y %H:%M:%S')
    telescope_side = orientation.upper()
    fig.update_layout(transition_duration=500, autosize=False,width=1000,
                        height=600, 
                        title={'text': f'{plot_time}, {telescope_side}',
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                                })

    return fig

#FAKE TELESCOPE
def fake_telescope():
    fake_tel = [random.uniform(0., 24.), random.uniform(-10., 90)]
    print('the fake telescope function', fake_tel)
    return fake_tel


#REAL TELESCOPE 
# def mount_telescope(reg):
#     telescope = []
#     for r in reg:
#         time.sleep(1)
#         result = register_decoder(client, r)
#         result = result*(180/math.pi)
#         if r == 8212:
#             result = result/15
#             if result < 12:
#                 result = result +12
#             else:
#                 result = result -12
#         telescope.append(result)
#         telescope.append(reg)
#     time.sleep(0.5)
#     return telescope

def read_textfield(text):
    object_data = []
    splitted_text = text.split('\n')
    # print(splitted_text)
    for i, row in enumerate (splitted_text):
        if not row:
            continue
        # print("row", row)
        splitted_row = row.split(',')
        # print("splitted_row", splitted_row)
        str_row = [str(f) for f in splitted_row[0:]]
        # print("str_row", str_row)
        object_data.append(str_row)
    # print("object_data", object_data)

    object_array = np.array(object_data)
    # print('object_array',object_array)
    return object_array

#LOADING OF THE FILE WITH THE COORDINATES OF THE OBJECTS
def exract_csv_file(filepath):
    
    objects_file = filepath
    # extracting the file and converting it into np array
    object_data = []
    with open(objects_file, 'r') as f:
        file_to_split = f.read()
        for i , row in enumerate(file_to_split.split('\n')):
            print('row', row)
            if i ==0 or not row:
                continue
            splitted_row = row.split(',')
            str_row = [str(f) for f in splitted_row[0:]]
            object_data.append(str_row)

        object_array = np.array(object_data)
        print('i am in the object array', object_array)

    return object_array

#CALCULATION OF THE LOCAL SIDERAL TIME AND THE OBSERVATION TIME
def LST_calculation(latit, longit):

    suhora = EarthLocation(lat=latit*u.deg, lon=longit*u.deg)

    observing_time = Time(datetime.datetime.utcnow(), 
                          scale='utc', 
                          location=suhora
                          )

    LST = observing_time.sidereal_time('mean')

    return observing_time, LST

#CONVERTING RA TO LOCAL HOURANGLE
def hourangle_conversion(object_array, LST, observing_time):
    np_object_array = np.array(object_array)
    """Conversion of ra in hourangle"""
    object_array_LHA = np_object_array.copy()
    
    coo  = SkyCoord(object_array_LHA[:,1], 
                    object_array_LHA[:,2], 
                    frame='icrs', 
                    unit=(u.hourangle, 
                    u.deg)
                    )  # passing in string format
    object_array_LHA[:,1] = coo.ra
    object_array_LHA[:,2] = coo.dec
    FK5_Jnow = FK5(equinox=observing_time)
    LHA = LST - coo.transform_to(FK5_Jnow).ra.to(u.hourangle)
    object_array_LHA[:,1] = LHA
    
    for i, l in enumerate(object_array_LHA[:,1]):
        l = float(l) * u.hourangle
        if l < 0 * u.hourangle:
            l += 24 * u.hourangle
            object_array_LHA[i,1] = l
    
    for i, l in enumerate(object_array_LHA[:,1]):
        l = float(l) * u.hourangle
        if l < 12*u.hourangle:
            object_array_LHA[i,1] = l + 12*u.hourangle
        elif l >= 12*u.hourangle:
            object_array_LHA[i,1] = l - 12*u.hourangle
    
    
    return object_array_LHA


def coordinates_transformation(object_array_LHA):
    array_for_plot = object_array_LHA.copy()
    array_for_plot[:,1] = object_array_LHA[:,1].astype(float)
    array_for_plot[:,2] = object_array_LHA[:,2].astype(float)
    return array_for_plot


def coordinates_calculations(input_array_data):#input: object's data
    
    observing_time, LST = LST_calculation(SUHORALATIT, SUHORALONGIT)
    object_array_LHA = hourangle_conversion(input_array_data, LST, observing_time)
    array_for_plot = coordinates_transformation(object_array_LHA)
    return array_for_plot

if __name__ == '__main__':
    # csv_object_array = exract_csv_file(FILEPATH)
    
    app.run_server(debug=True)