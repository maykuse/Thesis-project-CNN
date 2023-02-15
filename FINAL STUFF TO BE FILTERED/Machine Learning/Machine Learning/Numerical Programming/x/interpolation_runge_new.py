#!/usr/bin/python

#This is a python script visualizing the runge effect using uniform distributed interpolation points.
#It also shows a possible solution to this problem: using a chebyshev distribution for the points.
#You can modify N to increase the number of interpolation points (N+1).
#The interpolating polynomial is then evaluated at m+1 uniform points.
#Copyright Michael Obersteiner

import math
from matplotlib import pylab
import matplotlib.pyplot as plt
import numpy as np
import random



def f(x):
    return 1.0/(1.0+25.0*x*x)
row = []
rows = []
row_cheb = []
rows_cheb = []
element = 0

#we use n points for constructing the interpolation polynomial
N=10
n=N+1
delX = 2.0/N

#some initializations

b_uniform = np.zeros(n)
b_cheb = np.zeros(n)
x_vals_uniform = np.zeros(n)
x_vals_cheb = np.zeros(n)
A = np.zeros((n,n))
A_cheb = np.zeros((n,n))
#fillMatrix with powers of x (uniform, chebyshev)
for i in range(0,n):
    #row = []
    #row_cheb = []
    #calculate uniform x
    x_uniform = -1 + i*delX
    x_vals_uniform[i] = x_uniform
    x_cheb = math.cos(math.pi * i*1.0 / (N*1.0))
    x_vals_cheb[i] = x_cheb
    y_uniform = f(x_uniform)
    y_cheb = f(x_cheb)
    value = 1
    value_cheb = 1
    for j in range(0,n): 
        #row.append(value)
        A[i][j]=value
        #row_cheb.append(value_cheb)
        A_cheb[i][j]=value_cheb
        value *= x_uniform
        value_cheb *= x_cheb
        

    rows.append(row)
    rows_cheb.append(row_cheb)
    b_uniform[i] = y_uniform
    b_cheb[i] = y_cheb

#calculating coefficients

c_uniform = np.linalg.solve(np.array(A),np.array(b_uniform))
c_cheb = np.linalg.solve(np.array(A_cheb),np.array(b_cheb))


#evaluating the function at m+1 points
m=200
delM=2.0/m
value = []
rows = []
y_exact = np.zeros(m+1)
x_vals_test = np.zeros(m+1)
#constructing test matrix
A_test = np.zeros((m+1,n))
for i in range(0,m+1):
    row = []
    x = -1 + i*delM
    x_vals_test[i] = x
    y = f(x)
    value = 1
    for j in range(0,n):
        A_test[i][j] = value
        #row.append(value)
        value *=x

        

    #rows.append(row)

    y_exact[i] = y

#evaluating functions at all points using Matrix vector product
y_uniform = np.inner(A_test,c_uniform)
y_cheb = np.inner(A_test,c_cheb)


#create plot
fig = plt.figure()
plt.plot(x_vals_test,y_uniform,color='red',lw=2,ls='-',label='interpolated uniform points')
plt.plot(x_vals_test,y_cheb,color='green',lw=2,ls='-',label='interpolated Chebyshev')
plt.plot(x_vals_uniform,b_uniform,color='brown',lw=2,ls='-',label='interpolated piecewise linear')
plt.plot(x_vals_test,y_exact,color='blue',lw=3,ls='--',label='f(x)')
plt.xlabel('x')
plt.ylabel('y')
#plt.legend()
plt.legend(bbox_to_anchor=(0.9, 1))

plt.savefig('runge_solution.pdf')

plt.show()



