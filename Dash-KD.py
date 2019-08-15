import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pyodbc
from dash.dependencies import Input, Output, State
import BDAkdtree as kd



#mapbox_access_token = 'pk.eyJ1Ijoia3dvbm0iLCJhIjoiY2p4MHk0NTlhMDF4bjN6bnp6bm8xcmswOSJ9.OANG2d0eU8VCjsShWpccgQ'
# USE MARKDOWN FOR HTML

# connect to DB server
# server = 'CKCWBDA2'
# database = 'BDA_RWI'
# username = 'BDA_READ'
# password = 'readonly'
# cnxn = pyodbc.connect(
#     'DRIVER={ODBC Driver 17 for SQL Server}; SERVER=' + server + '; DATABASE=' + database + '; UID=' + username + '; PWD=' + password + ';MARS_Connection=yes')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

#
# INITIAL_QUERY = pd.read_sql_query(
#             '''SELECT
#                 [PROJECT_NAME],
#                 [WELL_COMMON_NAME],
#                 [API_SUFFIX],
#                 [MD],
#                 [TVDSS] * -1 as [TVDSS],
#                 [MAP_NORTHING],
#                 [MAP_EASTING],
#                 [LATITUDE],
#                 [LONGITUDE],
#                 [mrkname] as [MRKNAME],
#                 [top_perf] as [TOP_PERF],
#                 [bot_perf] as [BOT_PERF],
#                 [frac_flag] as [FRAC_FLAG],
#                 [perf_status] [PERF_STATUS],
#                 [SURVEY_DATE],
#                 [perf_start_date] as [PERF_START_DATE],
#                 CONCAT(WELL_COMMON_NAME,'_',API_SUFFIX) as NEW_WELL_NAME
#                 FROM [BDA_RWI].[dbo].[surveys_markers_perfs_v] where [WELL_COMMON_NAME] in ('B748', 'B623', 'D716', 'B634', 'B737', 'B748', 'B706', 'D719', 'B739', 'B568', 'A303', 'A307', 'D748')
#                  order by [MD] asc;''', cnxn
# )

INITIAL_QUERY = pd.read_csv(r'C:\Users\kwonm\Documents\TEST\TEXAS SOURCE FILES\points.csv')
INITIAL_QUERY['API_SUFFIX'] = '0' + INITIAL_QUERY['API_SUFFIX'].astype(str)
INITIAL_QUERY_1 = INITIAL_QUERY[INITIAL_QUERY['WELL_COMMON_NAME'].isin(['B748', 'B623', 'D716', 'B634', 'B737', 'B748', 'B706', 'D719', 'B739', 'B568', 'A303', 'A307', 'D748'])]
tree = kd.SurveyKdtree(INITIAL_QUERY)

def clean_smp(query):
    """
    Clean, filter query
    Create differentiated well names based on API suffix

    PARAMETERS:
        INPUT: SQL QUERY
        OUTPUT: DATAFRAME
    """

    query['MRKNAME'] = query['MRKNAME'].str.strip()
    query['MRKNAME'] = query['MRKNAME'].replace('F0', 'FO')
    # drop rows with null API suffixes
    query = query.dropna(subset=['API_SUFFIX'])

    return query

def make_trace(well_name, df, color):
    """
    Plot all given well bore points
    PARAMETERS:
        INPUT: WELL NAME
        OUTPUT: PLOTLY DATA TRACE FOR WELL
    """

    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]  # filtered df

    x = well_coord['MAP_EASTING']
    y = well_coord['MAP_NORTHING']
    z = well_coord['TVDSS']

    trace = go.Scatter3d(
        x = x, y = y, z = z,
        mode = 'lines',
        line = dict(width = 3,
                  color = color),
        name = well_name,
        legendgroup = well_name,
    )

    return trace

def make_marker_trace(well_name, df):
    """
    Plot color-coded markers of a given well name.
    Relies on marker_colordict. Color dictionary for all markers.
    PARAMETERS:
        INPUT: WELL NAME
        OUTPUT: PLOTLY DATA TRACE FOR WELL MARKERS
    """

    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]  # filtered df
    well_coord = well_coord[pd.notnull(well_coord['MRKNAME'])]  # filter out null values for marker name
    grouped = well_coord.groupby('MRKNAME').first()
    marker_index = grouped.index
    mrk_clr = marker_index.map(marker_colordict)

    x = np.concatenate([grouped['MAP_EASTING']], axis = 0)
    y = np.concatenate([grouped['MAP_NORTHING']], axis = 0)
    z = np.concatenate([grouped['TVDSS']], axis = 0)

    trace = go.Scatter3d(
        x = x, y = y, z = z,
        mode = 'markers+text',
        marker = dict(color = mrk_clr,
                    size = 5),
        name = well_name + "<br>" + "Markers",
        text = marker_index,
        textposition = "middle right",
        textfont = dict(
            size = 10),
        legendgroup = well_name,
        # hoverinfo = 'text',   <--------- gives only the text
        showlegend = True
    )

    return trace

def make_perf_trace(well_name, df):
    """
    Plot perfs of a given well name. Provides perf position and status.
    PARAMETERS:
        INPUT: WELL NAME
        OUTPUT: PLOTLY DATA TRACE FOR WELL PERFS
    """

    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]  # filtered df
    well_coord = well_coord[pd.notnull(well_coord['PERF_STATUS'])]
    well_coord.loc[well_coord['MRKNAME'].isnull(), 'MRKNAME'] = 'N/A'

    perf_status_color = {'INACTIVE': '#f57b42', 'ACTIVE': 'green'}
    perf_clr = well_coord['PERF_STATUS'].map(perf_status_color)

    trace = go.Scatter3d(
        x = well_coord['MAP_EASTING'], y = well_coord['MAP_NORTHING'], z = well_coord['TVDSS'],
        mode = 'markers',
        marker = dict(color = perf_clr,
                    size = 5,
                    symbol = 'diamond',
                    opacity = .1),
        name = well_name + ' Perfs',
        text = 'Top Perf: ' + well_coord['TOP_PERF'].astype(str) + "<br>" + "Bottom Perf: " + well_coord[
            'BOT_PERF'].astype(str) + "<br>" + 'Perf Status: ' + well_coord['PERF_STATUS'].astype(str) + "<br>" +
             "Marker: " + well_coord['MRKNAME'],
        legendgroup = 'Perfs',
        # hoverinfo = 'text',   <--------- gives only the text
        showlegend = True
    )

    return trace

def make_frac_trace(well_name, df):
    """
    Plot fracs of a given well name. Indicates T/F for frac location.
    PARAMETERS:
        INPUT: WELL NAME
        OUTPUT: PLOTLY DATA TRACE FOR WELL FRACS
    """
    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]  # filtered df
    well_coord = well_coord.loc[(well_coord.FRAC_FLAG == 'F') | (well_coord.FRAC_FLAG == 'X')]

    trace = go.Scatter3d(
        x = well_coord['MAP_EASTING'], y = well_coord['MAP_NORTHING'], z = well_coord['TVDSS'],
        mode = 'markers',
        marker = dict(color = 'black',
                    size = 5,
                    symbol = 'diamond',
                    opacity = .08),
        name = well_name + ' Fracs',
        text = "Fracs: True" + "<br>" + "Marker: " + well_coord['MRKNAME'],
        legendgroup = 'Fracs',
        # hoverinfo = 'text',   <--------- gives only the text
        showlegend = True
    )

    return trace

def add_fault(fault):

    #fault path variable
    data = pd.read_csv(r'C:\Users\kwonm\Documents\TEST\fault files\%s' % fault, sep=' ', header=None, skiprows=20)
    x = data[data.columns[0]].values
    y = data[data.columns[1]].values
    z = data[data.columns[2]].values * -1

    trace = [go.Mesh3d(
        x = x, y = y, z = z,
        color = 'black',
        opacity = 0.50,
        name = fault
        )]

    return trace

# def generate_well_map(df):
#     grouped = df.groupby('WELL_COMMON_NAME')
#     well_long = np.concatenate(grouped['LONGITUDE'].unique(), axis=0)
#     well_lat = np.concatenate(grouped['LATITUDE'].unique(), axis=0)
#     project_name = np.concatenate(grouped['PROJECT_NAME'].unique(), axis=0)
#     well_index = grouped['PROJECT_NAME'].unique().index
#
#     data = [go.Scattermapbox(
#         lat=well_lat,
#         lon=well_long,
#         mode="markers",
#         marker=dict(size=7,
#                     color='red'
#                     ),
#         text=project_name + '<br>' + well_index
#         # can add name,
#         # selected points = index
#         # custom data)
#     )]
#
#     layout = go.Layout(
#         autosize=True,
#         hovermode='closest',
#         mapbox=go.layout.Mapbox(
#             accesstoken=mapbox_access_token,
#             bearing=0,
#             pitch=0,
#             zoom=15,
#             center=go.layout.mapbox.Center(lat=33.76004, lon=-118.18054)),
#         height=700
#     )
#
#     return {'data': data, 'layout': layout}

# def parse_contents():

def plot_3d_wellbore(wellbore_df, fault, color='#f542f2'):
    """
    Plot  the 3d graph
    PARAMETERS:
        INPUT: Surveys_markers_perfs dataframe, fault, color (defaulted to pink unless entered otherwise)
        OUTPUT: List of data traces to be plugged into FIGURE
    """
    data_traces = []

    if wellbore_df.empty:
        for i in fault:
            data_traces.extend(add_fault(i))
    else:
        df = clean_smp(wellbore_df)
        if fault == None: fault = []

        for i in df['NEW_WELL_NAME'].unique():
            trace = make_trace(i, df, color)
            data_traces.append(trace)

            marker_trace = make_marker_trace(i, df)
            data_traces.append(marker_trace)

        for i in df['NEW_WELL_NAME'].unique():
            trace = make_perf_trace(i, df)
            data_traces.append(trace)

        for i in df['NEW_WELL_NAME'].unique():
            trace = make_frac_trace(i, df)
            data_traces.append(trace)

        if fault != None:
            for i in fault:
                data_traces.extend(add_fault(i))

    return data_traces

#color dictionary for markers only need to be enumerated once
marker_all = ['A', 'AA', 'AB', 'AC', 'AD', 'AE', 'AI', 'AM', 'AO', 'AR', 'AU', 'AX', 'BA', 'F', 'FO',
              'G', 'G4', 'G5', 'G6', 'H', 'H1', 'HX', 'HX1', 'HXA', 'HXB', 'HXC', 'HXO', 'J', 'K', 'M',
              'M1', 'S', 'T', 'W', 'X', 'Y', 'Y4', 'Z']

marker_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b', '#2ca02c', '#a667bd', '#e377b5', '#7f7f7f', '#6522bd',
                 '#17c5cf']

# color index for markers
marker_colordict = {i: marker_colors[j % len(marker_colors)] for j, i in enumerate(marker_all)}
#itertools / zip

lb_faults = {'LBU': 'LBU FLT',
            'LBU A1-N': 'LBU A1-N FLT',
            'LBU A1-S': 'LBU A1-S FLT',
            'Junipero': 'JUNIPERO FLT',
            'Junipero A1': 'JUNIPERO A1 FLT',
            'Belmont': 'BELMONT FLT',
            'Belmont A1': 'BELMONT A1 FLT',
            'Belmont B1': 'BELMONT B1 FLT',
            'Daisy': 'DAISY_1',
            'Daisy B1': 'DAISY_B-1',
            'Daisy B2': 'DAISY AVE-B2 FLT'}

layout = go.Layout(
                                                height = 700,
                                                margin = dict(l = 0, r = 20, t = 20, b = 0),
                                                scene = dict(
                                                    xaxis = dict(
                                                        title = 'X (EASTING)',
                                                        backgroundcolor = "rgb(200, 200, 230)",
                                                        gridcolor = 'rgb(255, 255, 255)',
                                                        showbackground = True,
                                                        zerolinecolor = 'rgb(255, 255, 255)'
                                                    ),
                                                    yaxis = dict(title = 'Y (NORTHING)',
                                                               backgroundcolor = "rgb(230, 200,230)",
                                                               gridcolor = "rgb(255, 255, 255)",
                                                               showbackground = True,
                                                               zerolinecolor = "rgb(255, 255, 255)"
                                                               ),
                                                    zaxis = dict(title = 'SUBSURFACE Z',
                                                               backgroundcolor = "rgb(230, 230,200)",
                                                               gridcolor = "rgb(255, 255, 255)",
                                                               showbackground = True,
                                                               zerolinecolor = "rgb(255, 255, 255)"
                                                               )
                                                        )
                                        )
"""
START BODY CONTENT FOR DASH APP
"""
count = 0
body = dbc.Container([
    dbc.Row([
            dbc.Col([
                html.H1('KD Tree Neighbor Search'),
                html.Br(),
                dbc.FormGroup(
                    [
                        html.Div('Enter a well to search neighbors'),
                        dbc.Input(
                            id='query_well',
                            maxLength=4,
                            placeholder='Enter well name'
                        ),
                        html.Div('Enter re-drill code if applicable'),
                        dbc.Input(
                            id='query_rd',
                            maxLength=2,
                            placeholder='Type in well re-drill code'
                        ),
                        html.Div('Enter radius distance to be searched'),
                        dbc.Input(
                            id='query_distance',
                            type='number',
                            placeholder='Enter radius'  ####required field constraint??
                        ),
                        html.Br(),
                        dbc.Button('Submit', id='button', size='md')
                        ]
                ),
                html.Br(),
                html.H2('Geoprog Search'),
                dcc.Upload(
                    id = 'geoprog',
                    children = html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                    style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            }
                ),
                html.Div('Select Fault by Fault Name'),
                dcc.Dropdown(id = 'faults',
                             placeholder= 'Select Fault(s)',
                             multi= True,
                            options = [{'label': i, 'value': lb_faults[i]} for i in lb_faults]),

                ],
                width = 3, align= 'center'
            ),
            dbc.Col([
                    html.H3('Subsurface Map'),
                    dcc.Graph(id='subsurface viz',
                              figure={'data': [],#plot_3d_wellbore(INITIAL_QUERY_1, fault= []),
                                     'layout': layout},
                              ),
                    #html.H3('Well Map'),
                    # dcc.Graph(id = 'well_map',
                    #           figure= generate_well_map(INITIAL_QUERY)
                    #           )
                ],
                width = 9
            )
        ])
], fluid = True)

app.layout = html.Div(body)

#update graph and well_map upon user selection changes
@app.callback(
    Output('subsurface viz', 'figure'),
     #Output('well_map', 'figure')],
    [Input('faults', 'value'),
     Input('button', 'n_clicks')],
    [State('query_well', 'value'),
     State('query_rd', 'value'),
     State('query_distance', 'value'),
     State('subsurface viz', 'figure')]
)

def update_3d_graph(fault, button_click, well_query, rd_query, distance_query, figure):

    print( 'this is initialized:', well_query)
    print('this is initialized:', rd_query)
    print('this is initialized:', distance_query)

    if button_click:

        query_well = well_query + '_' + rd_query
        print(query_well)
        query_well_df = INITIAL_QUERY.loc[INITIAL_QUERY['NEW_WELL_NAME'] == query_well]
        print(query_well_df)
        well_neighbor_df = tree.query_well_rad(well_query, rd_query, distance_query)[3]
        well_neighbor_df['NEW_WELL_NAME'] = well_neighbor_df['WELL_COMMON_NAME_nbr'] + '_' + well_neighbor_df['API_SUFFIX_nbr']
        neighbor_well_names = well_neighbor_df['NEW_WELL_NAME'].tolist()

        # placeholders = ','.join('?' for i in range(len(new_well_names)))
        #
        # SMP_QUERY = '''SELECT
        #             [PROJECT_NAME],
        #             [WELL_COMMON_NAME],
        #             [API_SUFFIX],
        #             [MD],
        #             [TVDSS] * -1 as [TVDSS],
        #             [MAP_NORTHING],
        #             [MAP_EASTING],
        #             [LATITUDE],
        #             [LONGITUDE],
        #             [mrkname] as [MRKNAME],
        #             [top_perf] as [TOP_PERF],
        #             [bot_perf] as [BOT_PERF],
        #             [frac_flag] as [FRAC_FLAG],
        #             [perf_status] [PERF_STATUS],
        #             [SURVEY_DATE],
        #             [perf_start_date] as [PERF_START_DATE],
        #             CONCAT(WELL_COMMON_NAME,'_',API_SUFFIX) as NEW_WELL_NAME
        #             FROM [BDA_RWI].[dbo].[surveys_markers_perfs_v] where CONCAT(WELL_COMMON_NAME,'_',API_SUFFIX) in (%s) order by [MD] asc;''' % placeholders
        #
        # df = pd.read_sql(SMP_QUERY, cnxn, params = new_well_names)

        df = INITIAL_QUERY[INITIAL_QUERY['NEW_WELL_NAME'].isin(neighbor_well_names)]
        data = plot_3d_wellbore(df, fault)
        data.extend(plot_3d_wellbore(query_well_df, fault = None, color = '#037bfc'))
        # need to convert fault input to string to pass into fault function. one fault at time at first?? multiple means big list. need
        # to break to big list to smaller strings.
        figure['data'] = data
        figure['layout'] = layout
        return figure

    if not figure['data'] or figure['data'][0]['name'] != fault:
        df = pd.DataFrame()
        figure['data'] = plot_3d_wellbore(df, fault=fault)
        figure['layout'] = layout
        return figure

# run the server
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug = False)
# dash will automatically refresh browser when change in code when debug = True

#data/logic files
#single quotes

