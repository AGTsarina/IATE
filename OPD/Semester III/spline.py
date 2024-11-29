import numpy as np
import matplotlib.pyplot as plt

Dx = 0.01
x = np.arange(5, 10 + Dx / 2, Dx)
#y = np.random.uniform(5,10, x.size)
y = np.sin(x) + np.cos(2 * x) + x * 0.5

def spline1(x1, y1, x2, y2):
    A = np.array([[x1, 1], [x2, 1]])
    A1 = np.linalg.inv(A)
    B = np.array([y1, y2])
    return A1.dot(B)

def spline2_base(x, y):
    x1, x2, x3 = x
    A = np.array([[x1 ** 2, x1, 1],
                  [x2 ** 2, x2, 1],
                  [x3 ** 2, x3, 1]])
    A1 = np.linalg.inv(A)
    B = y
    return A1.dot(B)

def spline2(x, y, coeff):
    x1, x2 = x
    a0, b0, c0 = coeff
    A = np.array([[x1 ** 2, x1, 1],
                  [x2 ** 2, x2, 1],
                  [2 * x1, 1, 0]])
    A1 = np.linalg.inv(A)
    B = np.array([y[0], y[1], 2 * a0 * x1 + b0])
    return A1.dot(B)

def I(x1, x2, coeff):
    a, b, c = coeff
    return (a * (x2 ** 3 - x1 ** 3) / 3 +
            b * (x2 ** 2 - x1 ** 2) / 2 +
            c * (x2 - x1))

fig, ax = plt.subplots()
ax.plot(x,y, 'or')
dim = 2
coeff = spline2_base(x[:3], y[:3])
a, b, c = coeff
x1 = x[0]
xN = x[2]
x_ = np.arange(x1, xN + 0.01, 0.1)
    # print(x_[-1])
y_ = a * x_ **2 + b * x_ + c
ax.plot(x_, y_)
res = I(x1, xN, coeff) 
for i in range(dim, x.size - 1, 1):
    x_slice = x[i:i + 2]
    y_slice = y[i:i + 2]
    coeff = spline2(x_slice, y_slice, coeff)
    a, b, c = coeff
    dx = 0.1
    x1 = x_slice[0]
    xN = x_slice[1]
    x_ = np.arange(x1, xN + 0.05, 0.1)
    # print(x_[-1])
    y_ = a * x_ **2 + b * x_ + c
    res += I(x1, xN, coeff)
    ax.plot(x_, y_)
print(res)
print((y * Dx).sum())
plt.show()