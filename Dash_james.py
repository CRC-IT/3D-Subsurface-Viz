r"""
3D Visualization of the subsurface using Plotly and Dash

Code by Mary Kwon, CRC Visualization Intern in Summer 2019
Data engineering of LBU wellbore schema by Justin Clark
Refactor by James Scarborough in September 2019

Package install notes:
Copied csv files to local repo directory, gitignore points.csv
pip install pandas, dash, dash_bootstrap_components, pyodbc
pip install ... GitHub\3D-Subsurface-Viz\pyproj-2.2.2-cp36-cp36m-win_amd64.whl
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import pyodbc
import itertools
import pyproj

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = '3D Viz'
mapbox_access_token = 'pk.eyJ1Ijoia3dvbm0iLCJhIjoiY2p4MHk0NTlhMDF4bjN6bnp6bm8'\
                      'xcmswOSJ9.OANG2d0eU8VCjsShWpccgQ'
USE_FROZEN_DATA = True

if not USE_FROZEN_DATA:
    server, database = 'CKCWBDA2', 'BDA_RWI'
    # uid, pwd = 'BDA_READ', 'readonly'
    cnxn = pyodbc.connect(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};DATABASE={database};MARS_Connection=yes;'
        # f'UID={uid};PWD={pwd};'
        f'Trusted_Connection=yes;')

    WELLBORE = pd.read_sql_query(
        """SELECT
            wellkey,
            api_suffix,
            wellid,
            well_name,
            well_redrill,
            comp_flag,
           CONCAT(wellkey, '_', api_suffix) as NEW_WELL_NAME
           FROM [BDA_RWI].[dbo].[blkeng_borehole_v];""", cnxn)

    WELLCOMP = pd.read_sql_query(
        """SELECT
            well_name,
            crb_desc,
            cut_bk,
            new_pool_desc,
            well_redrill,
           CONCAT(well_name, well_redrill) as REDRILL_WELL_NAME
           FROM [BDA_RWI].[blkeng].[compl];""", cnxn)

    WELLPATH = pd.read_sql_query(
        """SELECT
            PROJECT_NAME,
            WELL_COMMON_NAME,
            API_SUFFIX,
            MD,
            TVDSS * -1 as TVDSS,
            MAP_NORTHING,
            MAP_EASTING,
            LATITUDE,
            LONGITUDE,
            mrkname as MRKNAME,
            top_perf as TOP_PERF,
            bot_perf as BOT_PERF,
            frac_flag as FRAC_FLAG,
            perf_status PERF_STATUS,
            SURVEY_DATE,
            perf_start_date as PERF_START_DATE,
           CONCAT(WELL_COMMON_NAME, '_', API_SUFFIX) as NEW_WELL_NAME
           FROM [BDA_RWI].[dbo].[surveys_markers_perfs_v]
           --WHERE WELL_COMMON_NAME in ('B748', 'B623', 'D716', 'B634', 'B737', 'B748', 'B706', 'D719', 'B739', 'B568', 'A303', 'A307', 'D748')
           ORDER BY MD ASC;""", cnxn)
else:
    WELLBORE = pd.read_csv('data/borehole.csv')  # BLKENG_BOREHOLE_QUERY
    WELLCOMP = pd.read_csv('data/comp.csv')  # BLKENG_COMPL_DF
    WELLPATH = pd.read_csv('data/points.csv', low_memory=False) # INITIAL_QUERY
DEFAULTS = ['B748', 'B623', 'D716', 'B634', 'B737', 'B748', 'B706', 'D719',
            'B739', 'B568', 'A303', 'A307', 'D748']
INITIAL = WELLPATH[WELLPATH['WELL_COMMON_NAME'].isin(DEFAULTS)]


def clean(df):
    """
    Clean & filter query dataframe.
    Create differentiated well names based on API suffix
    """
    df.loc[:, 'MRKNAME'] = df['MRKNAME'].str.strip()
    df.loc[:, 'MRKNAME'] = df['MRKNAME'].replace('F0', 'FO')
    df = df.dropna(subset=['API_SUFFIX'])  # drop null API suffixes
    return df


def make_trace(well_name, df):
    """
    Plot well bore points
    """
    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]

    trace = go.Scatter3d(
        x=well_coord['MAP_EASTING'],
        y=well_coord['MAP_NORTHING'],
        z=well_coord['TVDSS'],
        mode='lines', line={'width': 3, 'color': '#f542f2'},
        name=well_name, legendgroup=well_name)
    return trace


def make_marker_trace(well_name, df):
    """
    Plot color-coded markers of a given well name.
    Relies on colormap, the color dictionary for all markers.
    """
    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]
    well_coord = well_coord[pd.notnull(well_coord['MRKNAME'])]  # null markers
    tops = well_coord.groupby('MRKNAME').first()
    marker_index = tops.index  # This is the marker letter
    marker_colors = marker_index.map(colormap)

    trace = go.Scatter3d(
        x=tops['MAP_EASTING'],
        y=tops['MAP_NORTHING'],
        z=tops['TVDSS'],
        mode='markers+text',
        marker=dict(color=marker_colors, size=5),
        name=well_name + '<br>Markers',
        text=marker_index, textposition='middle right', textfont={'size': 10},
        legendgroup=well_name,
        # hoverinfo='text',   <- gives only the text
        showlegend=True)
    return trace


def make_perf_trace(well_name, df):
    """
    Plot perfs of a given well name. Provides perf position and status.
    """
    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]
    well_coord = well_coord[pd.notnull(well_coord['PERF_STATUS'])]
    well_coord.loc[well_coord['MRKNAME'].isnull(), 'MRKNAME'] = 'N/A'

    perf_status_color = {'INACTIVE': '#f57b42', 'ACTIVE': 'green'}
    perf_clr = well_coord['PERF_STATUS'].map(perf_status_color)

    trace = go.Scatter3d(
        x=well_coord['MAP_EASTING'],
        y=well_coord['MAP_NORTHING'],
        z=well_coord['TVDSS'],
        mode='markers',
        marker={'color': perf_clr, 'size': 5, 'symbol': 'diamond',
                'opacity': 0.1},
        name=well_name + ' Perfs',
        text='Top Perf: ' + well_coord['TOP_PERF'].astype(str) +
             '<br>Bottom Perf: ' + well_coord['BOT_PERF'].astype(str) +
             '<br>Perf Status: ' + well_coord['PERF_STATUS'] +
             '<br>Marker: ' + well_coord['MRKNAME'],
        legendgroup='Perfs',
        # hoverinfo = 'text',   <--------- gives only the text
        showlegend=True)
    return trace


def make_frac_trace(well_name, df):
    """
    Plot fracs of a given well name. Indicates T/F for frac location.
    """
    well_coord = df.loc[df['NEW_WELL_NAME'] == well_name]
    well_coord = well_coord.loc[(well_coord.FRAC_FLAG == 'F') |
                                (well_coord.FRAC_FLAG == 'X')]

    trace = go.Scatter3d(
        x=well_coord['MAP_EASTING'],
        y=well_coord['MAP_NORTHING'],
        z=well_coord['TVDSS'],
        mode='markers',
        marker=dict(color='black', size=5, symbol='diamond', opacity=0.08),
        name=well_name + ' Fracs',
        text='Fracs: True<br>Marker: ' + well_coord['MRKNAME'],
        legendgroup='Fracs',
        # hoverinfo = 'text',   <- gives only the text
        showlegend=True)
    return trace


def add_fault(fault):
    """
    Plot faults (LBU) from name of fault file to 3D mesh data trace
    """
    df = pd.read_csv(f'Faults/{fault}', sep=' ', header=None, skiprows=20)
    trace = [go.Mesh3d(x=df[0], y=df[1], z=df[2] * -1,
                       color='black', opacity=0.50, name=fault)]
    return trace


def plot_3d_wellbore(wellbore_df, fault=None):
    """
    Plot the 3D graph, list of data traces to be plugged into FIGURE
    """
    traces = []
    if wellbore_df is not None:
        df = clean(wellbore_df)
        uniques = df['NEW_WELL_NAME'].unique()
        for i in uniques:
            traces.append(make_trace(i, df))
            traces.append(make_marker_trace(i, df))
        for i in uniques:
            traces.append(make_perf_trace(i, df))
        for i in uniques:
            traces.append(make_frac_trace(i, df))
    if fault:
        for i in fault:
            traces.extend(add_fault(i))
    return traces


def unproject(df, epsg=26799):
    """Realtime unproject of State Plane XYs to Lat/Lon with datum shift"""
    proj = pyproj.Proj(init=f'epsg:{epsg}')
    lonlats = [proj(x, y, inverse=True) for x, y in df.values]
    return pd.DataFrame(lonlats, columns=['LONGITUDE', 'LATITUDE'])


def plot_2d_map(df):
    """Regular map in MapBox"""
    grouped = df.groupby('WELL_COMMON_NAME').first()
    wellpath = unproject(df[['MAP_EASTING', 'MAP_NORTHING']])
    welltops = unproject(grouped[['MAP_EASTING', 'MAP_NORTHING']])
    data = [
        go.Scattermapbox(
            lat=welltops['LATITUDE'],
            lon=welltops['LONGITUDE'],
            mode='markers',
            marker=dict(size=7,
                        color='red'),
            text=grouped.index,
            hoverinfo='text',
            showlegend=False
            ),
        go.Scattermapbox(
            lat=wellpath['LATITUDE'],
            lon=wellpath['LONGITUDE'],
            mode='markers',
            marker=dict(size=1,
                        color='#f542f2',
                        opacity=0.5),
            hoverinfo='none',
            showlegend=False
            )
        ]
    return data


maplayout = go.Layout(
    height=200,
    margin=dict(l=0, r=0, t=0, b=0),
    autosize=True,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        pitch=0,
        zoom=11,
        center=dict(lat=33.75, lon=-118.17)))

# Color dictionary for markers, same marker gets the same color
markers = ['A', 'AA', 'AB', 'AC', 'AD', 'AE', 'AI', 'AM', 'AO', 'AR', 'AU',
           'AX', 'BA', 'F', 'FO',
           'G', 'G4', 'G5', 'G6', 'H', 'H1', 'HX', 'HX1', 'HXA', 'HXB',
           'HXC', 'HXO', 'J', 'K', 'M',
           'M1', 'S', 'T', 'W', 'X', 'Y', 'Y4', 'Z']

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b', '#2ca02c',
          '#a667bd', '#e377b5', '#7f7f7f', '#6522bd', '#17c5cf']

colormap = dict(zip(markers, itertools.cycle(colors)))

# Dictionaries for reservoirs, pools, CRBs, faults
reservoirs = {'Ranger': 'R',
              'Terminal': 'TE',
              'UPFord': 'UF'}

pools = {'R':  ['R08A', 'R08B', 'R6S', 'R6N', 'R90S', 'R07S', 'R07N', 'R90N'],
         'TE': ['TE08A', 'TE90S', 'TE90N'],
         'UF': ['UF08', 'UF98', 'UF90']}

crbs = {'R07N':  [10, 11, 9],
        'R07S':  [12, 13, 37, 7, 8],
        'R08A':  [14, 15],
        'R08B':  [16],
        'R6N':   [1, 2, 3],
        'R6S':   [3, 36, 4, 5],
        'R90N':  [17, 18, 20, 32, 33],
        'R90S':  [21, 22],
        'TE08A': [24],
        'TE90N': [42],
        'TE90S': [43],
        'UF08':  [27],
        'UF90':  [44, 45, 46],
        'UF98':  [26, 31]}

faults = {'LBU': 'LBU FLT',
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
    height=700,
    margin=dict(l=0, r=20, t=20, b=0),
    scene=dict(
        xaxis=dict(
            title='X (Easting)',
            backgroundcolor='rgb(200, 200, 230)', showbackground=True,
            gridcolor='white', zerolinecolor='white', tickformat=',f'),
        yaxis=dict(
            title='Y (Northing)',
            backgroundcolor='rgb(230, 200, 230)', showbackground=True,
            gridcolor='white', zerolinecolor='white', tickformat=',f'),
        zaxis=dict(
            title='Subsurface Z',
            backgroundcolor='rgb(230, 230, 200)', showbackground=True,
            gridcolor='white', zerolinecolor='white', tickformat='f')))


"""
START BODY CONTENT FOR DASH APP
"""
body = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('3D Subsurface Visualization'),
            html.Div('Select Reservoir'),
            dcc.Dropdown(id='reservoir',
                         options=[{'label': k, 'value': v}
                                  for k, v in reservoirs.items()],
                         placeholder='Select Reservoir',
                         value='R'),
            html.Div('Select Pool'),
            dcc.Dropdown(id='pool',
                         placeholder='Select Pool',
                         value=''),
            html.Div('Select CRB'),
            dcc.Dropdown(id='crb',
                         placeholder='Select CRB',
                         value=[],
                         multi=True),
            dcc.Checklist(id='crbc',
                          options=[{'label': 'Show all wells (takes a minute)',
                                    'value': True}],
                          value=[]),
            html.Div('Select Well'),
            dcc.Dropdown(id='wells',
                         #style={'overflow': 'auto'},
                         placeholder='Select Wells',
                         value='STARTUP',
                         multi=True),
            dcc.Markdown(
                'Select checkboxes below to see associated incomplete wells. '
                'This checklist will generate only if applicable.'),
            dcc.Checklist(id='incompletes',
                          value=[],
                          labelStyle={'display': 'inline-block'},
                          style={'overflow': 'auto'}),
            html.Div('Select Fault'),
            dcc.Dropdown(id='faults',
                         placeholder='Select Faults',
                         multi=True,
                         options=[{'label': k, 'value': v}
                                  for k, v in faults.items()]),
            html.Br(),
            html.H4('Location Map'),
            dcc.Graph(id='surface viz',
                      figure=dict(data=plot_2d_map(INITIAL),
                                  layout=maplayout)
                      ),
            ], width=3),
        dbc.Col([
            html.H3('Subsurface Map'),
            dcc.Graph(id='subsurface viz',
                      figure=dict(data=plot_3d_wellbore(INITIAL),
                                  layout=layout)
                      ),
            ], width=9)
        ])
    ],
    fluid=True)

app.layout = html.Div(body)


def _options(sequence):
    """Helper function to generate this dash format."""
    return [{'label': i, 'value': i} for i in sorted(sequence)]


# Update pool based on reservoir selection.
@app.callback(
    Output('pool', 'options'),
    [Input('reservoir', 'value')]
)
def return_pool_options(reservoir_value):
    return _options(pools[reservoir_value])


# Update CRB based on pool selection.
@app.callback(
    Output('crb', 'options'),
    [Input('pool', 'value')]
)
def return_well_options_from_pool(pool):
    if not pool:
        return []
    df = WELLCOMP[WELLCOMP['new_pool_desc'].str.contains(pool)]
    shared_crb = set(crbs[pool]).intersection(df['cut_bk'].unique())
    return _options(shared_crb)


# Update wells based on CRB selection or pool selection
@app.callback(
    Output('wells', 'options'),
    [Input('pool', 'value'),
     Input('crb', 'value')]
)
def return_well_options_from_crb(pool, crb):
    if not pool:
        return []
    df = WELLCOMP[WELLCOMP['new_pool_desc'].str.contains(pool)]
    if crb:
        df = df[df['cut_bk'].isin(crb)]
    well_names = df['REDRILL_WELL_NAME'].str.strip().unique()
    return _options(well_names)


# EXPERIMENT TO AUTO PLOT ALL WELLS IN A CRB -JS ON 10/1
@app.callback(
    Output('wells', 'value'),
    [Input('crb', 'value'),
     Input('crbc', 'value'),
     Input('wells', 'options')]
)
def return_well_value_from_crb(crb, crbc, wells_options):
    if crb and crbc:
        return [d['value'] for d in wells_options]
    else:
        pass
        #return [] #wells_value

    #df = WELLCOMP[WELLCOMP['new_pool_desc'].str.contains(pool)]
    #if crb:
    #   df = df[df['cut_bk'].isin(crb)]
    #well_names = df['REDRILL_WELL_NAME'].str.strip().unique()
    #return _options(well_names)


# Provide checklist of incomplete wells
@app.callback(
    Output('incompletes', 'options'),
    [Input('wells', 'value')],
    [State('incompletes', 'value')]
)
def return_incomplete_wells(well_selection, incompletes):
    if not well_selection:
        return []
    well_keys = list(set([i[:4] for i in well_selection]))
    compl_df = WELLBORE[WELLBORE['wellkey'].isin(well_keys)]
    compl_df = compl_df.loc[compl_df['comp_flag'] == 0]
    compl_df = compl_df[compl_df.api_suffix != '0-1']
    wells = list(compl_df['NEW_WELL_NAME'])
    wells = wells + incompletes
    return _options(wells)


# Update graph and well_map upon user selection changes
@app.callback(
    [Output('subsurface viz', 'figure'),
     Output('surface viz', 'figure')],
    [Input('wells', 'value'),
     Input('incompletes', 'value'),
     Input('faults', 'value')],
    [State('subsurface viz', 'figure'),
     State('surface viz', 'figure')]
)
def update_3d_graph(well_names, incomplete_wells, fault, subsurface, surface):
    if well_names == 'STARTUP' or not well_names:
        "Already rendered above, then well_names gets a value"
        pass
    else:
        new_well_names = list(WELLBORE[WELLBORE['wellid'].isin(well_names)]['NEW_WELL_NAME'])
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
        #             FROM [BDA_RWI].[dbo].[surveys_markers_perfs_v]
        #             WHERE CONCAT(WELL_COMMON_NAME,'_',API_SUFFIX) in (%s)
        #             ORDER BY [MD] asc;''' % placeholders
        # df = pd.read_sql(SMP_QUERY, cnxn, params = new_well_names)

        df = WELLPATH[WELLPATH['NEW_WELL_NAME'].isin(new_well_names)]
        subsurface['data'] = plot_3d_wellbore(df, fault)
        surface['data'] = plot_2d_map(df)
    return subsurface, surface


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)
