import numpy as np
''' The following functions are pulled from:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization '''

def rastrigrin(x, y, A=10):
    return (A * 2 + x**2 - A * np.cos(2 * np.pi * x)
                  + y**2 - A * np.cos(2 * np.pi * y))

def ackley(x, y):
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y **2)))
            -np.exp(0.5*(np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + np.e + 20)

def sphere(x, y):
    return x ** 2 + y ** 2

def rosenbrock(x, y):
    return (100 * (y - x**2)**2 + (1 - x)**2)

def beale(x, y):
    return ((1.5 - x + x * y)**2
            + (2.25 - x + x * y**2)**2
            + (2.625 - x + x * y**3)**2)

def goldstein_price(x, y):
    return ((1 + (x + y + 1) ** 2 *
            (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2))
            * (30 + (2 * x - 3 * y)**2 *
               (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y * y)))

def bukin(x, y):
    return 100. * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10.)

def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def levi(x, y):
    return (np.sin(3 * np.pi * x)**2 + (x - 1)**2 *
            (1 + np.sin(3 * np.pi * y)**2)
            + (y - 1)**2 * (1+np.sin(2 * np.pi * y)**2))

def himmelblau(x, y):
    return ((x ** 2 + y - 11)**2 + (x + y**2 - 7)**2)

def camel(x, y):
    return 2 * x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

def easom(x, y):
    return (-np.cos(x)*np.cos(y)*np.exp(-((x - np.pi)**2 + (y - np.pi)**2)))

def cross_in_tray(x, y):
    return -1e-4 * (np.abs(np.sin(x) * np.sin(y)
                    * np.exp(np.abs(100 - np.sqrt(x**2 + y**2)/np.pi)))
                    + 1.)**0.1

def eggholder(x, y):
    return (-(y + 47.) * np.sin(np.sqrt(np.abs(x/2 + y + 47.)))
            -x * np.sin(np.sqrt(np.abs(x - (y + 47.)))))

def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(
        1 - np.sqrt(x**2 + y**2)/np.pi)))

def mccormick(x, y):
    return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*x + 1

def schaffer(x, y):

