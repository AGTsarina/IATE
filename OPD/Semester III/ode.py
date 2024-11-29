import numpy as np
import matplotlib.pyplot as plt
from math import exp

# коэффициенты С1 и С2 для задачи с граничными условиями (краевая задача)
def find_coeff_bnd(L, a, b):
    A = np.array([[1, 1], [exp(L), exp(2 * L)]])
    B = np.array([a - 2.75, b - 1.5* L - 2.75])
    return np.dot(np.linalg.inv(A), B)

# коэффициенты С1 и С2 для задачи с начальными условиями (задача Коши)
def find_coeff_init(L, a, b):
    A = np.array([[1, 1], [1, 2]])
    B = np.array([a - 2.75, b - 1.5])
    return np.dot(np.linalg.inv(A), B)

# точное решение задачи
def y(x, C1, C2):      
    return C1 * exp(x) + C2 * exp(2 * x)  + 1.5 * x + 2.75

# векторизация
Y_acc = np.vectorize(y)


# краевая задача (с заданными граничными условиями)
def solve_bnd(L, a, b, h):
    X = np.arange(0, L+h, h)
    n = len(X)
    
    d_low = 1 / h ** 2 + 3 / (2*h)
    d = 2 - 2 / h ** 2
    d_up = 1 / h ** 2 - 3 / (2*h)
    
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n-1, n-1] = 1
    for i in range(1, n-1):
        A[i, i - 1] = d_low
        A[i, i] = d
        A[i, i + 1] = d_up
    B = np.array([a] + [3 * x + 1 for x in X[1:-1]] + [b])
    Y = np.dot(np.linalg.inv(A), B)
    return Y

# задача Коши (с заданными начальными условиями)
# матричный /нерациональный вариант метода Эйлера
def solve_init(L, a, b, h): 
    X = np.arange(0, L+h, h)
    n = len(X)    
    d_low = 1 / h ** 2 + 3 / (2*h)
    d = 2 - 2 / h ** 2
    d_up = 1 / h ** 2 - 3 / (2*h)
    
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[1, 0], A[1, 2] = -1, 1    
    for i in range(1, n-1):
        A[i+1, i - 1] = d_low
        A[i+1, i] = d
        A[i+1, i + 1] = d_up
    B = np.array([a, 2 *b*h] + [3 * x + 1 for x in X[2:]])
    Y = np.dot(np.linalg.inv(A), B)
    return Y

# задача Коши (с заданными начальными условиями)
# модифицированный метод Эйлера
def solve_init_modif(L, a, b, h): 
    X = np.arange(0, L+h, h)
    n = len(X)    
    Y = np.zeros(n)
    Z = np.zeros(n)
    y = Y[0] = a
    z = Z[0] = b
    for i, x in zip(range(1, n), X[1:]):
        zs = z + h * (3 * z - 2 * y + 3 * x + 1)
        ys = y + 0.5 * h * (z + zs)
        z = z + 0.5 * h * (3 * (z + zs) - 2 * (y + ys) + 3 * (x + x - h) + 2)
        Y[i] = y = ys
    return Y

def task(problem, coeff_finder, L=1, a=0, b=1, n=100):    
    h = L / n
    # дискретизация про-ва
    X = np.arange(0, L+h, h)
    # точное решение
    C1, C2 = coeff_finder(L, a, b)
    Y1 = Y_acc(X, C1, C2)
    # решение граничной задачи
    Y2 = problem(L, a, b, h)
    # вычисление ошибки
    dY = np.abs(Y2 - Y1).max()
    print(dY)
    # отрисовка
    plt.plot(X, Y1, 'r-')
    plt.plot(X, Y2, 'b.')
    plt.show()   


L, a, b, n = 1, 0, 1, 10

methods = {
    'bnd': (solve_bnd, find_coeff_bnd),
    'init': (solve_init, find_coeff_init),
    'mod': (solve_init_modif, find_coeff_init)
}

while(True):
    print('Choose "bnd" for boundary problem or "init" for Cauchy problem or "mod" for modified Cauchy method:')
    method_name = input()
    if method_name in methods.keys():
        print('Input n')
        n = int(input())
        problem, coeff_finder = methods[method_name]
        task(problem, coeff_finder, L, a, b, n)        
    else: break
    