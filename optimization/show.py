import plotly.graph_objects as go
import pandas as pd
import numpy as np
import single_objective
import multi_objective

single_obj_functions = [(name, getattr(single_objective, name))
                        for name in dir(single_objective)
                        if '__' not in name and name != 'np']
multi_obj_functions = [(name, getattr(multi_objective, name))
                       for name in dir(multi_objective)
                       if '__' not in name and name != 'np']

def surface(X, Y, Z, title=''):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis')])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
        project_z=True))
    fig.update_layout(title=title, autosize=False,
                      width=1920, height=1080,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()

def main():
    print(single_obj_functions)
    print(multi_obj_functions)

    n = 101
    r = 10
    nc = complex(n)
    X, Y = np.mgrid[-r:r:nc, -r:r:nc]

    for name, single in single_obj_functions:
        Z = single(X, Y)
        surface(X, Y, Z, title=name.replace('_', ' ').title())

    for name, multi in multi_obj_functions:
        Z = multi(X, Y)
        for i, Zi in enumerate(Z):
            surface(X, Y, Zi, title=(name.replace('_', ' ').title() + ' Objective ' + str(i)))

if __name__ == '__main__':
    main()

