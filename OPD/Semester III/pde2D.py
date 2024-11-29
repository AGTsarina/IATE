import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from math import cos, exp

def f(x:float, y: float) -> float:
    return 0
        
def mu2(x:float, t=0.0) -> float:
    return 300 + 10 * x

def mu1(y:float, t=0.0) -> float:
    return 300 - 10 * y

def q(x, y):
    if x < 6 and x > 4 and y < 6 and y > 4:
        return 1000
    return 0

class BaseAxis:
    def __init__(self, length:float) -> None:
        self.L = 0       
        self.R = length
        
class SpaceAxis(BaseAxis):
    def __init__(self, length: float, Ndiv=11) -> None:
        super().__init__(length)            
        self.N = Ndiv  
        self.h = (self.R - self.L) / (self.N - 1)
        self.x = np.arange(self.L,  self.R + self.h * 0.5, self.h)
        
class TimeAxis(BaseAxis):
    def __init__(self, length: float, maxSpaceDiv:float, a:float) -> None:
        super().__init__(length)
        self.dt = 0.1 * (maxSpaceDiv / a) ** 2
        self.t = np.arange(self.L, self.R + self.dt * 0.5, self.dt)
        self.N = len(self.t)

class PDESolver:
    def __init__(self, a:float, q, mu1, mu2, f, Lx, Ly, T_) -> None:
        X = SpaceAxis(Lx, 101)
        Y = SpaceAxis(Ly, 101)
        T = TimeAxis(T_, max(X.h, Y.h), a)
        self.X = X
        self.Y = Y
        self.T = T
        
        self.cx = a ** 2 * T.dt / X.h ** 2
        self.cy = a ** 2 * T.dt / Y.h ** 2  
        self.cur = 1 - 2 * a ** 2 * T.dt *(1/X.h ** 2 + 1/ Y.h ** 2)       
        
             
        self.N = X.N * Y.N
        self.u = PDESolver.create_u(self.N, f, X, Y)
        self.A = PDESolver.create_A(self.N, X, Y, self.cx, self.cy)
        self.B = np.zeros(self.N)
       
        
        self.res = [self.u[0].copy()]
        self.res_max = 0
        self.invA = np.linalg.inv(self.A)

    def create_u(N, f, X, Y):
        u = [np.zeros(N), np.zeros(N)]
        # начальные условия
        for i, x in zip(range(X.N), X.x):
            for j, y in zip(range(Y.N), Y.x):
                u[0][i * Y.N + j] = f(x, y)
        return u
    
    def create_A(N:int, X:SpaceAxis, Y:SpaceAxis, cx:float, cy:float):
        A = np.zeros((N, N))
        # создание матрицы
        # граничные условия
        for i, x in zip(range(0, X.N), X.x):
            A[i * Y.N, i * X.N] = 1
            A[i * Y.N + Y.N - 1, i * Y.N + Y.N - 1] = 1
            A[i * Y.N + Y.N - 1, i * Y.N + Y.N - 2] = -1            
        # граничные условия
        for j, y in zip(range(0, Y.N), Y.x):
            A[j][j] = 1
            A[(X.N - 1) * Y.N + j, (X.N - 1) * Y.N + j] = 1
            A[(X.N - 1) * Y.N + j, (X.N - 2) * Y.N + j] = -1            
        # диффузия
        for i, x in zip(range(1, X.N - 1), X.x):
            for j, y in zip(range(1, Y.N - 1), Y.x):
                ind = i * Y.N + j
                A[ind, ind] = 1.0
                A[ind, ind - Y.N] = -cx
                A[ind, ind - 1] = -cy
        return A
    
    def count_B(B, u, X:SpaceAxis, Y:SpaceAxis, dt, mu1, mu2, q, cur, cx, cy):
         # граничные услович
        for i, x in zip(range(0, X.N), X.x):  
            B[i * Y.N] = mu2(x)          
            B[i * Y.N + Y.N - 1] = 0
        
        for j, y in zip(range(0, Y.N), Y.x):            
            B[j] = mu1(y)
            B[(X.N - 1) * Y.N + j] = 0
        
        for i, x in zip(range(1, X.N - 1), X.x[1:-1]):
            for j, y in zip(range(1, Y.N - 1), Y.x[1:-1]):
                ind = i * Y.N + j                          
                B[ind] = u[ind] * cur + cx * u[ind - X.N] + cy * u[ind + 1] + q(x, y) * dt
    
    
    def calc(self):      
        # начинаем расчет
        for i, t in zip(range(self.T.N), self.T.t):
            self.step(t)
            if (i % 5 == 0): 
                print(f"{t}")            
                self.res.append(self.u[0].copy())
                m =  max(self.u[0]) 
                if m > self.res_max: self.res_max = m
            
    

    def step(self, t):
        # расчет следующего слоя        
         
        PDESolver.count_B(self.B, self.u[0], self.X, self.Y, self.T.dt, mu1, mu2, q, self.cur, self.cx, self.cy)     
        self.u[1] = np.dot(self.invA, self.B)
        temp = self.u[1]
        self.u[1] = self.u[0]
        self.u[0] = temp
        
    def draw(self):       
        fig = plt.figure()                
        def init():
            data = np.array(self.res[0]).reshape(self.X.N, self.Y.N)
            sns.heatmap(data, vmax=self.res_max, vmin=0, square=True, cbar=False, cmap="seismic")
        def animate(i):
            data = np.array(self.res[i]).reshape(self.X.N, self.Y.N)
            sns.heatmap(data, vmax=self.res_max, vmin=0, square=True, cbar=False, cmap="seismic")
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.res), repeat = False)
        anim.save('plot.gif')
    
pde = PDESolver(1.0, q, mu1, mu2, f, 10, 10, 0.5)
pde.calc()
pde.draw()