import os
import sys
import dash
import json
import time
import math
import struct
import random
import logging
import datetime
import argparse
import pymodbus
import numpy as np
import pandas as pd
import astropy.units as u
import plotly.express as px
from astropy.time import Time
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from collections import OrderedDict
from pymodbus.constants import Endian
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from matplotlib.dates import  DateFormatter
from astropy.coordinates import FK4, ICRS, FK5
from dash.dependencies import Input, Output, State
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.payload import BinaryPayloadDecoder
from astropy import coordinates as coord, units as u
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from astropy.coordinates import SkyCoord, Angle, Latitude, Longitude, EarthLocation

script_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(filename=os.path.join(script_path, 'app.log'),
                    filemode='w',
                    level=logging.DEBUG
                    )
# client = ModbusClient('127.0.0.1', port=8888)
client = ModbusClient('192.168.2.16', port=502)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#UNIT FOR MODBUS CLIENT
UNIT = 0x1
#Degrees transformation
degrees_transformation = 180/math.pi
#Registers
REGISTERS_DICT = {'ra':8212,
                  'dec':8210
                 }

#Forbiden regions dictionary
f_zone = {'WEST': {'left_x':np.array([9.3,  8.9,  8.6,  8.2,  7.8,  7.3,  7., 
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
            'EAST': {    'left_x': np.array([10.8, 10,9.4, 8.8, 8.7, 8.6 , 8.2,
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

table_header = [
    html.Thead(html.Tr([html.Th("Telescope Information Box:")]))
]
row1 = html.Tr([html.Td("Hourangle:"), html.Td(id="hourangle", children="")])
row2 = html.Tr([html.Td("Declination:"), html.Td(id="declination", children="")])
row3 = html.Tr([html.Td("Orientation:"), html.Td(id="orientation", children="")])

table_body = [html.Tbody([row1, row2, row3])]

table = dbc.Table(table_header + table_body, bordered=False)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(children=[
                            dcc.Graph(id='my-graph'),
                            dcc.Interval(id='main-interval', interval=5000)
                                 ]
                        )
            ], align='center',
                ),
        dbc.Row(
            [
                dbc.Col(children=[
                
                    html.Div([
                        html.Div(
                            dcc.Textarea(id='text-field',
                                         style={'width': 350, 'height': 200},)
                        ),
                        html.Button('Submit', id='submit-button', n_clicks=0),
                        html.Div(id='data-div', style={'display': 'none'}),
                        ])
                ]),
                dbc.Col(table, width={'size': 5, 'offset': 0}),

                dbc.Col(),
                html.Div(id='telescope_position', 
                                 style={'display': 'none'})
            ], align='center',
        ),
])

@app.callback(
    Output('telescope_position', 'data-main'),
    Input('main-interval', 'n_intervals'))
def update_telescope_state(interval):

    logging.info('Updating telescope state')
    
    # is_open = client.is_socket_open()
    telescope_position = {}
    if True:
        try:
            telescope_position = mount_telescope(client)
        except:
            pass
        # telescope_position = mount_telescope(client)
         

    return telescope_position

@app.callback(
    Output('hourangle', 'children'),
    Output('declination', 'children'),
    Output('orientation','children'),
    Input('telescope_position', 'data-main'),
    Input('main-interval', 'n_intervals'))
def update_info_box(telescope_position, intervals):

    logging.info(f'Updating info box with telescope coordinates {telescope_position}')
    if telescope_position:
        hourangle = telescope_position['hours']
        mins = hourangle-math.modf(hourangle)[1]
        mins = mins*60
        secs = mins - math.modf(mins)[1]
        secs = secs*60
        ra = (int(math.modf(hourangle)[1]), 
            int(math.modf(mins)[1]),
            int(math.modf(secs)[1]))
        right_ascension = f'{ra[0]:02d}:{ra[1]:02d}:{ra[2]:02d}'

        degr = telescope_position['degrees']
        mins = degr-math.modf(degr)[1]
        mins = mins*60
        secs = mins - math.modf(mins)[1]
        secs = secs*60
        if degr<0:
            mins = -mins
            secs = -secs
        dec = (int(math.modf(degr)[1]), int(math.modf(mins)[1]), int(math.modf(secs)[1]))
        declination = f'{dec[0]:02d}:{dec[1]:02d}:{dec[2]:02d}'

        
    else:
        right_ascension = '.. : .. : ..'
        declination = '.. :.. : ..'
   

    return right_ascension, declination, telescope_position.get('orientation', 'EAST')

@app.callback(
    Output('data-div', 'children'),
    Input('submit-button', 'n_clicks'),
    State('text-field', 'value'))
def update_output(n_clicks, value):

    logging.info('Button update')

    return value

@app.callback(
    Output('my-graph', 'figure'),
    Input('main-interval', 'n_intervals'),
    Input('data-div', 'children'),
    Input('telescope_position', 'data-main'))
def update_figure(n_intervals, data, telescope_position):
    logging.info('Figure update')
    #ORIENTATION
    f_zone_data = f_zone[telescope_position.get('orientation', 'EAST')]

    #PLOTS
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
    if telescope_position:
        
        fig.add_trace(go.Scatter(x=[telescope_position['hours']], y=[telescope_position['degrees']], 
                                mode="markers+text", text=['Telescope']),
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
    telescope_side = telescope_position.get('orientation', 'EAST').upper()
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
    # fake_tel = [random.uniform(0., 24.), random.uniform(-10., 90)]
    ra = (random.uniform(0., 24.)*math.pi)/180
    dec = (random.uniform(-10., 180.)*math.pi)/180
    
    return {'ra':ra,
            'dec':dec
            }


# REAL TELESCOPE DECODER
def register_decoder(client):
    rr_ra = client.read_input_registers(REGISTERS_DICT['ra'], 
                                        2, unit=UNIT)
    logging.info(f'rr_ra {rr_ra}')
    decoder_ra = BinaryPayloadDecoder.fromRegisters(rr_ra.registers, 
                                                byteorder=Endian.Big, 
                                                wordorder=Endian.Big
                                                )
    logging.info(decoder_ra)
    decoded_ra = decoder_ra.decode_32bit_float()
    time.sleep(1)
    rr_dec = client.read_input_registers(REGISTERS_DICT['dec'], 
                                        2, unit=UNIT)
    logging.info(rr_dec)
    decoder_dec = BinaryPayloadDecoder.fromRegisters(rr_dec.registers, 
                                                byteorder=Endian.Big, 
                                                wordorder=Endian.Big
                                                )
    logging.info(decoder_dec)
    decoded_dec = decoder_dec.decode_32bit_float()

    decoded_result = {'ra':decoded_ra,
                      'dec':decoded_dec
                     }


    return decoded_result

#REAL TELESCOPE MOUNT
def mount_telescope(client):
    
    decoded_result = register_decoder(client)
    "FAKE REGISTER DECODER"
    # decoded_result = fake_telescope()
    decoded_result['ra'] = decoded_result['ra']*(180/math.pi)
    # hours = decoded_result['ra'] 

    hours = decoded_result['ra']/15
    
    if hours < 12:
        hours = hours +12
    else:
        hours = hours -12
    time.sleep(1)
    decoded_result['dec'] = decoded_result['dec']*(180/math.pi)
    orientation='EAST'
    degrees = decoded_result['dec']
    if  degrees > 90:
        degrees = 180-degrees
        orientation = 'WEST'
    elif degrees < -90:
        degrees = -180-degrees
        orientation = 'WEST'

    telescope_position = {'hours': hours,
                          'degrees': degrees,
                          'orientation':orientation}
    logging.info(f'TELESCOPE MOUNTED {telescope_position}')  
    time.sleep(0.5)
    return telescope_position

#READ DATA FROM TEXTFIELD BOX
def read_textfield(text):
    object_data = []
    splitted_text = text.split('\n')
    for i, row in enumerate (splitted_text):
        if not row:
            continue
        splitted_row = row.split(',')
        str_row = [str(f) for f in splitted_row[0:]]
        object_data.append(str_row)

    object_array = np.array(object_data)
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

#TRANSFORMATION TO LHA
def coordinates_transformation(object_array_LHA):
    array_for_plot = object_array_LHA.copy()
    array_for_plot[:,1] = object_array_LHA[:,1].astype(float)
    array_for_plot[:,2] = object_array_LHA[:,2].astype(float)
    return array_for_plot

#CALCULATION OF THE COORDINATES
def coordinates_calculations(input_array_data):#input: object's data
    
    observing_time, LST = LST_calculation(SUHORALATIT, SUHORALONGIT)
    object_array_LHA = hourangle_conversion(input_array_data, LST, observing_time)
    array_for_plot = coordinates_transformation(object_array_LHA)
    return array_for_plot

if __name__ == '__main__':
    # csv_object_array = exract_csv_file(FILEPATH)
    app.run_server(host=os.getenv("HOST", "0.0.0.0"), 
                   port=os.getenv("PORT", "8050"),
                   debug=True)