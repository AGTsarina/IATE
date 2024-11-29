import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

Num = 100

def trend1(x, y):
    sx = x.sum()
    sy = y.sum()
    sx2 = (x ** 2).sum()
    N = x.size
    k = ((x*y).sum()- sx * sy / N)/(sx2 - sx ** 2 / N)
    b = (sy - k * sx) / N
    return (k, b)

def trend2(x, y):
    s4 = (x ** 4).sum()
    s3 = (x ** 3).sum()
    s2 = (x ** 2).sum()
    s1 = x.sum()

    A = np.array([[s4, s3, s2],[s3, s2, s1],[s2, s1, x.size]])
    print(A)
    A1 = np.linalg.inv(A)
    print(A1.dot(A))
    B = np.array([((x**2) * y).sum(), (x * y).sum(), y.sum()])
    a, b, c = list(np.dot(A1, B))
    return (a, b, c)

def R(x, y):
    ax = x.sum() / x.size
    ay = y.sum() / y.size
    dx = x - ax
    dy = y - ay
    R = (dx * dy).sum() / ((dx ** 2).sum() * (dy ** 2).sum()) ** 0.5 
    return R

x1 = np.random.uniform(0, 10, Num)
y1 = np.random.uniform(0, 10, Num)

x2 = np.random.normal(0, 2, Num)
y2 = np.random.normal(0, 3, Num)

x3 = np.random.uniform(0, 10, Num)
y3 = np.random.normal(0, 4, x3.size) + x3**2

x4 = np.abs(np.random.normal(0, 3, Num))
y4 = np.abs(np.random.normal(0, 1, x4.size)) * x4**2




data = [(x1, y1), (x2,y2), (x3, y3), (x4, y4)]
for d in data:
    x, y = d
    print(R(x, y))
colors = ["b", "r", "g", "m"]

fig, ax = plt.subplots(2, 2)
fig1, ax1 = plt.subplots(2, 2)
axes = list(ax[0])
axes.extend(list(ax[1]))
axes1 = list(ax1[0])
axes1.extend(list(ax1[1]))
for a, d, c, a1 in zip(axes, data, colors, axes1):
    x, y = d
    k, b = trend1(x, y)
    A, B, C = trend2(x, y)
    # print(k, b)
    a.plot(x, y, "." + c)
    x_ = np.arange(x.min(), x.max(), 0.1)
    y_ = x_ * k + b
    # a.plot(x_, y_)
    y_1 = A * x_ ** 2 + B * x_ + C
    y_2 = A * x ** 2 + B * x + C
    a.plot(x_, y_1)
    a1.plot(y, y_2, '.b')
    a.grid(True)
    print(f'R = {R(y, y_2)}')
fig.savefig('1.png')
fig1.savefig('2.png')
# plt.show()


#ax.plot(x2, y2, 'ob')
#ax.grid(True)
#ax.plot(x3, y3, 'ob')
#ax.grid(True)
