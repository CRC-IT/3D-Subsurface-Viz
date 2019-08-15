import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pyodbc
from dash.dependencies import Input, Output, State


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

# BLKENG_BOREHOLE_QUERY = pd.read_sql_query(
#     '''SELECT
#         [wellkey],
#         [api_suffix],
#         [wellid],
#         [well_name],
#         [well_redrill],
#         [comp_flag],
#         CONCAT(wellkey,'_',api_suffix) as NEW_WELL_NAME
#         FROM [BDA_RWI].[dbo].[blkeng_borehole_v]; ''', cnxn
# )
#
# BLKENG_COMPL_DF = pd.read_sql_query(
#     '''SELECT
#         [well_name],
#         [crb_desc],
#         [cut_bk],
#         [new_pool_desc],
#         [well_redrill],
#         CONCAT(well_name,well_redrill) as REDRILL_WELL_NAME
#         FROM [BDA_RWI].[blkeng].[compl];''', cnxn
# )
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

BLKENG_BOREHOLE_QUERY = pd.read_csv(r'C:\Users\kwonm\Documents\TEST\TEXAS SOURCE FILES\borehole.csv')
BLKENG_COMPL_DF = pd.read_csv(r'C:\Users\kwonm\Documents\TEST\TEXAS SOURCE FILES\comp.csv')
INITIAL_QUERY = pd.read_csv(r'C:\Users\kwonm\Documents\TEST\TEXAS SOURCE FILES\points.csv')
INITIAL_QUERY_1 = INITIAL_QUERY[INITIAL_QUERY['WELL_COMMON_NAME'].isin(['B748', 'B623', 'D716', 'B634', 'B737', 'B748', 'B706', 'D719', 'B739', 'B568', 'A303', 'A307', 'D748'])]


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

def make_trace(well_name, df):
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
                  color = '#f542f2'),
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
    """
    Plot faults (LBU)
    PARAMETERS:
        INPUT: NAME OF FAULT FILE
        OUTPUT: PLOTLY 3D MESH DATA TRACE FOR FAULTS
    """
    if fault:
        #fault path variable
        data = pd.read_csv(r'C:\Users\kwonm\PycharmProjects\Project_Texas_CRC-IT\Faults\%s' % fault, sep=' ', header=None, skiprows=20)
        x = data[data.columns[0]].values
        y = data[data.columns[1]].values
        z = data[data.columns[2]].values * -1

        trace = [go.Mesh3d(
            x = x, y = y, z = z,
            color = 'black',
            opacity = 0.50,
            name = fault
        )]
    else:
        trace = []

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
#         autosize = True,
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

def plot_3d_wellbore(wellbore_df,fault):
    """
    Plot  the 3d graph
    PARAMETERS:
        INPUT: Surveys_markers_perfs dataframe, fault
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
            trace = make_trace(i, df)
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

    # ^^^^^ NEED TO CONVERT TO LIST COMPREHENSION

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

#dictionaries for reservoirs, pools, crbs

reservoir = {'Ranger': 'R',
            'Terminal': 'TE',
            'UPFord': 'UF'}

pool_desc = { 'R' : ['R08A', 'R08B', 'R6S', 'R6N', 'R90S', 'R07S', 'R07N', 'R90N'],
              'TE' : ['TE08A', 'TE90S', 'TE90N'],
              'UF' : ['UF08', 'UF98', 'UF90']}

pool_crb = { 'R07N' : [10, 11, 9],
             'R07S' : [12, 13, 37, 7, 8],
             'R08A' : [14, 15],
             'R08B' : [16],
             'R6N' : [1, 2, 3],
             'R6S' : [3, 36, 4, 5],
             'R90N' : [17, 18, 20, 32, 33],
             'R90S' : [21, 22],
             'TE08A': [24],
             'TE90N': [42],
             'TE90S': [43],
             'UF08': [27],
             'UF90': [44, 45, 46],
             'UF98': [26, 31]
             }

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
body = dbc.Container([
    dbc.Row([
            dbc.Col([
                html.H1('3D Subsurface Visualization'),
                html.Div('Select Reservoir'),
                dcc.Dropdown(id = 'reservoir',
                             options = [{'label': i, 'value': reservoir[i]} for i in reservoir],
                             placeholder = 'Select Reservoir',
                             value = 'R'),
                html.Div('Select Pool '),
                dcc.Dropdown(id = 'pool',
                             placeholder = 'Select Pool'),
                html.Div('Select CRB '),
                dcc.Dropdown(id = 'crb',
                             placeholder = 'Select crb',
                             multi = True),
                html.Div('Select Well'),
                dcc.Dropdown(id = 'wells',
                             placeholder = 'Select Wells',
                             multi = True),
                dcc.Markdown(
                    'Select checkboxes below to see correlated incomplete wells.  \nThis checklist will generate only if applicable.'),
                dcc.Checklist(id='incompletes',
                              values=[],
                              labelStyle={'display': 'block'}),
                html.Div('Select Fault by Fault Name'),
                dcc.Dropdown(id = 'faults',
                             placeholder= 'Select Fault(s)',
                             multi= True,
                            options = [{'label': i, 'value': lb_faults[i]} for i in lb_faults])
                    ], width = 3
                ),
            dbc.Col([
                    html.H3('Subsurface Map'),
                    dcc.Graph(id='subsurface viz',
                              figure={'data': plot_3d_wellbore(INITIAL_QUERY_1, fault= []),
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

@app.callback(
    Output('pool', 'options'),
    [Input('reservoir', 'value')]
)
def return_pool_options(reservoir_value):
    return [{'label': i, 'value': i} for i in pool_desc[reservoir_value]]

#update CRB based on pool selection.
@app.callback(
    Output('crb', 'options'),
    [Input('pool', 'value')]
)
def return_well_options_from_pool(pool):
    df = BLKENG_COMPL_DF[BLKENG_COMPL_DF['new_pool_desc'].str.contains(pool)]
    shared_crb = set(pool_crb[pool]).intersection(df['cut_bk'].unique())
    return [{'label': i, 'value': i} for i in shared_crb]

#update wells based on CRB selection or pool selection
@app.callback(
    Output('wells', 'options'),
    [Input('pool', 'value'),
     Input('crb', 'value')]
)
def return_well_options_from_crb(pool, crb):
    df = BLKENG_COMPL_DF[BLKENG_COMPL_DF['new_pool_desc'].str.contains(pool)]
    if crb:
        well_names = df[df['cut_bk'].isin(crb)]['REDRILL_WELL_NAME'].str.strip()
        return [{'label': i, 'value': i} for i in well_names.unique()]
    else:
        well_names = df['REDRILL_WELL_NAME'].str.strip()
        return [{'label': i, 'value': i} for i in well_names.unique()]

#provide checklist of incomplete wells
@app.callback(
    Output('incompletes', 'options'),
    [Input('wells', 'value')],
    [State('incompletes', 'values')]
)
def return_incomplete_wells(well_selection, incompletes):
    well_keys = list(set([i[:4] for i in well_selection]))
    compl_df = BLKENG_BOREHOLE_QUERY[BLKENG_BOREHOLE_QUERY['wellkey'].isin(well_keys)]
    compl_df = compl_df.loc[compl_df['comp_flag'] == 0]
    compl_df = compl_df[compl_df.api_suffix != '0-1']
    wells = list(compl_df['NEW_WELL_NAME'])
    wells = wells + incompletes
    return [{'label': i, 'value': i} for i in wells]

#update graph and well_map upon user selection changes
@app.callback(
    Output('subsurface viz', 'figure'),
     #Output('well_map', 'figure')],
    [Input('wells', 'value'),
    Input('incompletes', 'values'),
     Input('faults', 'value')],
    [State('subsurface viz', 'figure')]
)

def update_3d_graph(well_names, incomplete_wells, fault, figure):

    if (well_names == None or incomplete_wells == None) and fault != None:
        df = pd.DataFrame()
        figure['data'] = plot_3d_wellbore(df, fault = fault)
        figure['layout'] = layout

        return figure

    if (figure['data'] == [] or (figure["data"][0]["name"] != well_names or incomplete_wells)):

        new_well_names = list(BLKENG_BOREHOLE_QUERY[BLKENG_BOREHOLE_QUERY['wellid'].isin(well_names)]['NEW_WELL_NAME'])
        new_well_names.extend(incomplete_wells)
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

        df = INITIAL_QUERY[INITIAL_QUERY['NEW_WELL_NAME'].isin(new_well_names)]
        figure['data'] = plot_3d_wellbore(df, fault)
        figure['layout'] = layout
        return figure


# run the server
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug = False)

#data/logic files


