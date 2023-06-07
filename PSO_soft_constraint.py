import random
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 有界软约束粒子群最优化算法：约束处理、停机条件
class PSO_Soft:
    def __init__(self, func, dim=1, upperBounds=[float('inf')], lowerBounds=[float('inf')]) -> None:
        self.D = dim                        #计算空间的维度
        self.func = func                    #目标函数
        assert(len(upperBounds)==dim)
        assert(len(lowerBounds)==dim)
        self.upperBounds = np.array(upperBounds)
        self.lowerBounds = np.array(lowerBounds)
        # self.N = int(pow(3, dim))
        self.N = 30
        # 初始化种群：在有界的维度上产生均匀随机数；无界的维度上按照标准高斯分布产生随机数
        self.c1 = 0.6
        self.c2 = 0.3
        self.w = 1.2
    def init(self):
        #群体的当前位置: N × D
        self.curPos = np.random.random([self.N, self.D])*(self.upperBounds-self.lowerBounds) + self.lowerBounds
        poss = []
        vals = []
        self.globalBestVal = float('inf')
        for pos in self.curPos:
            poss.append(pos)
            val = self.get_value(pos)
            if val<self.globalBestVal:
                self.globalBestPos = pos                #群体的最优历史位置: D
                self.globalBestVal = val                #群体的最优历史数值: 1
            vals.append(val)
        self.individualBestPos = np.array(poss)         #个体历史最优位置: N × D
        self.individualBestVal = np.array(vals)         #个体历史最优数值: N
        self.preVel = np.zeros([self.N, self.D])        #群体的上一步速度: N × D
        self.historyValues = []
  
    def update(self):
        r1 = random.random()
        r2 = random.random()
        vel = self.preVel + self.c1*r1*(self.individualBestPos - self.curPos) + self.c2*r2*(self.globalBestPos - self.curPos)
        self.curPos = self.curPos + vel
        self.preVel = vel
        vals = [self.get_value(pos) for pos in self.curPos]
        for i in range(self.N):
            if vals[i] < self.individualBestVal[i]:
                self.individualBestPos[i] = self.curPos[i]
                self.individualBestVal[i] = vals[i]
        self.globalBestVal = min(self.individualBestVal)
        self.globalBestPos = self.individualBestPos[np.argmax(self.individualBestVal)]
        self.historyValues.append(self.globalBestVal)

        # 可视化当前位置
        xx = np.arange(0,15,0.1)
        yy = np.arange(0,15,0.1)
        X, Y = np.meshgrid(xx, yy)
        z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                z[i,j] = f_Damavandi([X[i,j], Y[i,j]])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, z, alpha=0.6)
        for i in range(self.N):
            x, y = self.curPos[i]
            z = self.get_value(self.curPos[i])
            ax.scatter(x, y, z)
        plt.show()


    def get_value(self, pos):
        return self.func(pos)
    # solve: 搜索 value 最小的位置
    def solve(self):
        self.init()
        for _ in range(100):
            self.update()
            print(self.globalBestPos, ':   ', self.globalBestVal)
        return self.globalBestPos, self.globalBestVal

pso = PSO_Soft(f_Damavandi, 2, [0, 0], [10, 10])
pso.solve()