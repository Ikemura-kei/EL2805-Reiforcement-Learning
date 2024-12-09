import numpy as numpy


def f(x):
    y = - x**3
    
    return y

x_root = 1.5
for i in range(10000000000):
    x_root = x_root + 1.0/(i+1) * f(x_root)
    print(x_root)