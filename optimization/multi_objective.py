import numpy as np
''' The following functions are pulled from:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization '''

def bihn_korn(x, y):
    return ((4*x**2 + 4*y**2),
            ((x-5)**2 + (y-5)**2))

def chankong_haimes(x, y):
    return ((2 + (x - 2)**2 + (y-1)**2),
            (9*x - (y-1)**2))

def fonseca_fleming(x, y):
    sn = np.sqrt(2)
    return ((1 - np.exp(-((x - 1/sn)**2 + (y - (1/sn))**2))),
            (1 - np.exp(-((x + 1/sn)**2 + (y + (1/sn))**2))))

def test(x, y):
    return ((x**2 - y),
            (-0.5 * x - y - 1.))

def kursawe(x, y):
    return ((-10 * np.exp(-0.2 * np.sqrt(x**2 + y**2))),
            (abs(x)**0.8 + 5.*np.sin(x**3) +
             abs(y)**0.8 + 5.*np.sin(y**3)))

def schaffer(x, y):
    return (x**2, (x-2)**2)

def poloni(x, y,
    A1=(0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)),
    A2=(1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2))):
    B1 = 0.5*np.sin(x) - 2*np.cos(x) + np.sin(y) - 1.5*np.cos(y)
    B2 = 1.5*np.sin(x) - np.cos(x) + 2*np.sin(y) - 0.5*np.cos(y)
    return ((1 + (A1 - B1)**2 + (A2 - B2)**2),
            ((x + 3)**2 + (y + 1)**2))

def ctp1(x, y):
    return (x, (1+y)*np.exp(-(x/(1+y))))

def constr_ex(x, y):
    return (x, (1+y)/x)

def viennet(x, y): # Three objectives
    return ((0.5*(x**2+y**2) + np.sin(x**2 + y**2)),
            ((3*x-2*y+4.)**2/8. + (x-y+1.)**2/27. + 15.),
            ((1./(x**2 + y**2 + 1.) - 1.1*np.exp(-(x**2+y**2)))))

''' Optionally could add higher-dim functions '''

