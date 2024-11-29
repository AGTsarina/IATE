import numpy as np
from graph import Graph

from math import cos, exp

def f(x:float) -> float:
    return cos(4 * x)
        
def g(t:float) -> float:
    return exp(-t)

def s(t: float) -> float:
    return t ** 2

class PDESolver:
    def __init__(self) -> None:
        L = 0 
        R = 10       
        
        self.N = 100
        self.dx = (R - L) / self.N
        self.x = np.arange(L,  R + self.dx * 0.5, self.dx)
        
        self.c1 = 1.0 / 6.0
        self.c2 = 1.0 - 2.0 * self.c1
        
        self.t0 = 0
        self.T = 10
        self.dt = self.dx ** 2 / self.c1
        self.nT = int(self.T / self.dt  + 1)        
       
        self.u = [None] * self.nT
        self.t = 0

    def calc(self):       
        self.t = self.t0
        # начальные условия
        self.u[0] = np.vectorize(f)(self.x)
        self.u[1] = np.zeros(len(self.x))                 
        # начинаем расчет
        for i in range(1, self.nT):
            self.step(i)
    

    def step(self, i):
        # расчет следующего слоя
        self.t += self.dt
        print(f"{self.t}")
        if (self.t > self.T): return
        # граничные условия для следующего временного слоя       
        up = self.u[i - 1]
        u = [g(self.t)] + [self.c1 * (up[i + 1] + up[i - 1]) + self.c2 * up[i] for i in range(1, self.N)] + [0.0]
        u[self.N] = (u[self.N - 1] - self.dx * s(self.t)) / (1 + self.dx)
        self.u[i] = np.array(u)
        
    def draw(self):       
        gr = Graph(self.x, self.u, self.nT)
        gr.anim.save('data.gif')
    
pde = PDESolver()
pde.calc()
pde.draw()