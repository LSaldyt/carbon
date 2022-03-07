import plotly.express as px
import pandas as pd
import numpy     as  np

from subprocess import call
from pathlib import Path

from plotly.subplots import make_subplots
from pprint import pprint
import plotly.graph_objects as go
import plotly
from optimization import *

import white_theme

QUAL_COLORS = plotly.colors.qualitative.G10 + plotly.colors.qualitative.Dark24 + plotly.colors.qualitative.Light24

def heatmap(*compare, labels=None, title=None):
    fig = make_subplots(rows=1, cols=len(compare), subplot_titles=labels)
    for i, c in enumerate(compare):
        fig.add_trace(go.Heatmap(z=c, zmin=0, zmax=1), row=1, col=i + 1)
    fig.update_layout(title_text=title)
    save(fig, f'{title.replace(" ", "_")}_heatmap', w=1000, h=800)
    return fig

def save(fig, name, w=1000, h=600):
    fig.show()
    fig.write_image(f'data/{name}.svg', width=w, height=h)
    fig.write_image(f'data/{name}.png', width=w, height=h)
    call(f'rsvg-convert -f pdf -o data/{name}.pdf data/{name}.svg', shell=True)

def compare(filename, x='generation', y='loss', color='name', x_label='Generation',
         legend_title='Metrics', rename=None, title=''):
    df = pd.read_csv(filename)
    if rename is not None:
        df = df.replace(rename)
    fig = go.Figure()
    x = df[x]
    for age in (True, False):
        width = 3; mode = 'lines+markers';
        color = [] * len(x)
        filt  = df[df['age'] == age]
        yi = 0 if age else 1
        fig.add_trace(go.Scatter(x=x, y=filt[y], mode=mode,
            name=y.title(), line=dict( color=QUAL_COLORS[yi], width=width)))
    fig.update_layout(font_size=32, xaxis_title=x_label,
        yaxis_title=y.title() if isinstance(y, str) else 'Fitness',
        title_text=title)
    fig.update_xaxes(type='log', tickfont=dict(size=24))
    # fig.update_yaxes(type='log', tickfont=dict(size=24))
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=24),
                      legend_title_text=legend_title)
    save(fig, f'compare', w=1400, h=700)
    return df, fig

def visualize():
    ''' Visualize the objective function '''
    x = np.linspace(-0.5, 1.0, 1000)
    y = x * np.sin(10 * np.pi * x) + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color=QUAL_COLORS[0], width=3)))
    fig.update_layout(font_size=32, xaxis_title='Decision variable (x)',
                      yaxis_title='Objective function (f)')
    fig.add_annotation(x=0.85, y=1.85, text='Max',
                       showarrow=False, yshift=15)
    fig.add_annotation(x=0.95, y=0.005, text='Min',
                       showarrow=False, yshift=-10)

    save(fig, f'objective', w=1400, h=700)

def surface(X, Y, Z, title='', filename='surface'):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis')])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
        project_z=True))
    w = 1080; h = 1080
    fig.update_layout(title='', autosize=False,
                      width=w, height=h,
                      margin=dict(l=0, r=0, b=0, t=0))
                      # margin=dict(l=65, r=50, b=65, t=90))

    save(fig, filename, w=w, h=h)

def plot_all_funcs():
    n = 1001
    r = 10
    nc = complex(n)
    X, Y = np.mgrid[-r:r:nc, -r:r:nc]

    for name, single in single_obj_functions.items():
        Z = single(X, Y)
        surface(X, Y, Z, title=name.replace('_', ' ').title(), filename=name)

    for name, multi in multi_obj_functions.items():
        Z = multi(X, Y)
        for i, Zi in enumerate(Z):
            surface(X, Y, Zi, title=(name.replace('_', ' ').title() + ' Objective ' + str(i)), filename=name)

def plot_specific(name):
    n = 101
    r = 10
    nc = complex(n)
    X, Y = np.mgrid[-r:r:nc, -r:r:nc]
    func = all_functions[name]

    Z = func(X, Y)
    surface(X, Y, Z, title=name.replace('_', ' ').title(), filename=name)

if __name__ == '__main__':
    filename = 'metrics.csv'
    compare(filename, y='fitness')
    # plot_specific('rastrigrin')

