import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import cos, exp
from mpl_toolkits.mplot3d import Axes3D

def f(x:float) -> float:
    return cos(4 * x)
        
def g(t:float) -> float:
    return 0#exp(-t)

def s(t: float) -> float:
    return 1 #t ** 1.1/100

class GraphPDE:
    def __init__(self, L:float, R:float, N:int) -> None:             
        self.N = N
        self.dx = (R - L) / self.N
        self.x = np.arange(L,  R + self.dx * 0.5, self.dx)
        
        self.c1 = 1.0 / 6.0
        self.c2 = 1.0 - 2.0 * self.c1
        
        self.t0 = 0
        self.T = 10
        self.dt = self.dx ** 2 / self.c1
        self.nT = int(self.T / self.dt  + 1)        
       
        # начальные условия
        self.u = [np.vectorize(f)(self.x), np.zeros(len(self.x))]
        self.t = 0        
       
        self.fig = plt.figure()
        self.axes = self.fig.subplots(ncols=1, nrows=2)
        
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=200, frames = self.nT,
                                          init_func=self.init)
        
    
    def color_field():
        y = np.repeat(np.arange(0, 10.05, 1), 11).reshape((11, 11))
        x = y.copy().T
        z = np.sqrt(x**2 + y ** 2)        
        # z = np.random.normal(0.0, 0.1, (11, 11))
        
        cm = plt.get_cmap('jet')
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.pcolormesh(x, y, z, shading='gouraud', cmap=cm)
        ax1 = fig.add_subplot(212, projection='3d')
        ax1.plot_surface(x, y, z, cmap='inferno')
        plt.show()
    
    def colored_line(self, u,  line_width=1, MAP='jet'):
        # use pcolormesh to make interpolated rectangles
        num_pts = len(self.x)
        if u is None:
            u = np.random.uniform(size=num_pts)
        [xs, ys, zs] = [
            np.array([self.x, self.x]),
            np.array([np.zeros(num_pts), line_width * np.ones(num_pts)]),
            np.array([u, u])
        ]
        
        cm = plt.get_cmap(MAP)
        self.colormap = self.axes[1].pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)
        return self.colormap

    def init(self):
        self.t = self.t0
        """Initial drawing of the scatter plot."""        
        self.plot = self.axes[0].plot(self.x, self.u[0], '-b')
        self.colormap = self.colored_line(self.u[0])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.plot, 

    def update(self, k):            
            if k == 0:
                 return self.plot,
              # расчет следующего слоя
            self.t += self.dt
            print(f"{k}, {self.t}, {self.u[0].min()}-{self.u[0].max()}")
            if (self.t > self.T): return self.plot,
            # граничные условия для следующего временного слоя       
            up = self.u[0]
            u = self.u[1]
            u[0] = g(self.t)
            u[self.N] = s(self.t)
            for i in range(1, self.N):
                u[i] = self.c1 * (up[i + 1] + up[i - 1]) + self.c2 * up[i]
            # u[self.N] = (u[self.N - 1] - self.dx * s(self.t)) / (1 + self.dx)           
            
            """Update the scatter plot."""       
            self.axes[0].clear() 
            self.plot = self.axes[0].plot(self.x, self.u[1], '-b')
            self.axes[0].set_ylim(-1, 1)
            self.colormap = self.colored_line(self.u[1])
            
            temp = self.u[0]
            self.u[0] = self.u[1]
            self.u[1] = temp
                
            return self.plot,

#pde = GraphPDE(0, 10, 100)
#pde.anim.save('pic.gif')
# plt.show()

GraphPDE.color_field()