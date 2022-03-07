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

def contour(X, Y, Z, p0, p1):
    fig = go.Figure(data = go.Contour(x=X[:, 0], y=Y[0], z=Z, colorscale='viridis'))
    fig.add_trace(go.Scatter(x=p0, y=p1, mode='markers',
        name='Path', line=dict(color=QUAL_COLORS[1], width=3)))
    fig.show()

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

def markers(l, legend_title='Variants', title='', filename='markers'):
    fig = go.Figure()
    width = 2; mode = 'markers'
    for yi, (name, x, y) in enumerate(l):
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode,
            name=name, line=dict(color=QUAL_COLORS[yi], width=width)))
    fig.update_layout(font_size=32, xaxis_title='X',
        yaxis_title='Y', title_text=title)
    # fig.update_xaxes(type='log', tickfont=dict(size=24))
    # fig.update_yaxes(type='log', tickfont=dict(size=24))
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=32),
                      legend_title_text=legend_title)
    save(fig, filename, w=1400, h=700)
    return fig

def error_bars(x, bars, x_label='Generation', legend_title='Variants', title='', filename='error_bars'):
    fig = go.Figure()
    width = 2; mode = 'lines+markers'
    for yi, (name, y, err) in enumerate(bars):
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode, error_y=dict(
            type='data', array=err, visible=True),
            name=name, line=dict(color=QUAL_COLORS[yi], width=width)))
    fig.update_layout(font_size=32, xaxis_title=x_label,
        yaxis_title='Fitness', title_text=title)
    # fig.update_xaxes(type='log', tickfont=dict(size=24))
    # fig.update_yaxes(type='log', tickfont=dict(size=24))
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=24),
                      legend_title_text=legend_title)
    save(fig, filename, w=1400, h=700)

def compare(filename, x='generation', y='loss', color='name', x_label='Generation',
         legend_title='Variants', rename=None, title=''):
    df = pd.read_csv(filename)
    print(df)
    if rename is not None:
        df = df.replace(rename)
    fig = go.Figure()
    x = df[x]
    for age_enabled in (True, False):
        width = 2; mode = 'lines+markers';
        filt  = df[df['age_enabled'] == age_enabled]
        if age_enabled:
            yi = 0; name = 'AFPO'
        else:
            yi = 1; name = 'PO'
        fig.add_trace(go.Scatter(x=x, y=filt[y], mode=mode,
            name=name, line=dict(color=QUAL_COLORS[yi], width=width)))
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

def split_by(df, field):
    mask = df[field]
    return df[mask == True], df[mask == False]

def get_ablations(df):
    afpo, po = split_by(df, 'age_enabled')
    afpo_exact, afpo = split_by(afpo, 'exact')
    po_exact, po     = split_by(po,   'exact')
    ablations = dict(afpo_exact=afpo_exact, afpo=afpo, po_exact=po_exact, po=po)
    return ablations

def analyze(filename, n=128):
    df = pd.read_csv(filename)
    x = list(range(n))
    ablations = get_ablations(df)
    bars = []
    for k, v in ablations.items():
        means = []; devs = [];
        for g in range(n):
            filt = v[v['generation'] == g - 1]['f0']
            means.append(filt.mean()); devs.append(filt.std())
        bars.append((k, means, devs))
    error_bars(x, bars, x_label='Generation', legend_title='Variants', title='', filename=filename[5:-4])

def pareto_progress(df):
    fig = px.scatter(df, x='x', y='y', color='age',
        color_continuous_scale='Viridis')
    fig['data'][0]['marker']['opacity'] = 0.5
    save(fig, 'progress', w=1400, h=700)

def progress(filename):
    df = pd.read_csv(filename)
    df = df[df['generation'] > 32]
    afpo, po = split_by(df, 'age_enabled')
    pareto_progress(afpo)
    pareto_progress(po)

def apply_layout(fig, xaxis_title='X', yaxis_title='Y'):
    fig.update_layout(font_size=48, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    legend_title = 'Variants'
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=48),
                      legend_title_text=legend_title)

def duration(filename):
    df = pd.read_csv(filename)
    ablations = get_ablations(df)
    fig = go.Figure()
    for i, (name, ablation) in enumerate(ablations.items()):
        try:
            title, exact = name.split('_')
        except:
            title = name; exact = ''
        title = title.upper(); exact = exact.title()
        title = title + ' ' + exact
        fig.add_trace(go.Bar(
            name=title,
            marker_color=QUAL_COLORS[i],
            x=ablation['generation'],
            y=[ablation['duration'].mean()],
            error_y=dict(type='data', array=[ablation['duration'].std()])))
    apply_layout(fig, yaxis_title='Duration', xaxis_title='Variant')
    save(fig, filename[5:-4] + '_duration', 1920, 1080)

def frontier(filename):
    df = pd.read_csv(filename)
    for g in [2**n-1 for n in range(6)] + [47]:
        for p0, p1 in (('x', 'y'), ('f0', 'f1')):
            final = df[df['generation'] == g]
            final['tag'] = (final['age_enabled'].apply(lambda age : 'age' if age else 'single')
                    + final['exact'].apply(lambda exact : ' exact' if exact else ' approx'))
            fig = px.scatter(final, x=p0, y=p1, color='tag', marginal_x='violin',
                    marginal_y='violin', color_discrete_sequence=QUAL_COLORS)
            fig['data'][0]['marker']['opacity'] = 0.5
            apply_layout(fig)
            save(fig, filename[5:-4] + f'_final_{g}_' + ''.join((p0, p1)), 1920, 1080)

def contour_optimization(f_name, filename):
    n = 101
    r = 10
    nc = complex(n)
    X, Y = np.mgrid[-r:r:nc, -r:r:nc]
    func = all_functions[f_name]

    df = pd.read_csv(filename)

    Z = func(X, Y)
    contour(X, Y, Z, df['x'], df['y'])

if __name__ == '__main__':
    # plot_specific('rastrigrin')
    tag = 'approx'
    # tag = ''
    # names = list(all_functions.keys())
    names = ['rastrigrin']
    # names = ['fonseca_fleming']
    if tag != '': tag += '_'
    for name in names:
        metrics = f'data/{name}_{tag}metrics.csv'
        long    = f'data/{name}_{tag}long.csv'
        # frontier(long)
        # analyze(metrics, n=64)
        # progress(long)
        # duration(metrics)
        contour_optimization(name, long)

