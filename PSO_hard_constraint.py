import random
import numpy as np
from utils import *
import matplotlib.pyplot as plt

# 有界软约束粒子群最优化算法：约束处理、停机条件
class PSO_Hard:
    def __init__(self, func, dim=1, upperBounds=[float('inf')], lowerBounds=[float('inf')]) -> None:
        self.D = dim                        #计算空间的维度
        self.func = func                    #目标函数
        assert(len(upperBounds)==dim)
        assert(len(lowerBounds)==dim)
        self.upperBounds = np.array(upperBounds)
        self.lowerBounds = np.array(lowerBounds)
        self.N = int(pow(3, dim))
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
        prePos = self.curPos
        self.curPos = prePos + vel
        self.curPos = max(self.lowerBounds, min(self.upperBounds, self.curPos))
        self.preVel = self.curPos - prePos
        vals = [self.get_value(pos) for pos in self.curPos]
        for i in range(self.N):
            if vals[i] < self.individualBestVal[i]:
                self.individualBestPos[i] = self.curPos[i]
                self.individualBestVal[i] = vals[i]
        self.globalBestVal = min(self.individualBestVal)
        self.globalBestPos = self.individualBestPos[np.argmax(self.individualBestVal)]
        self.historyValues.append(self.globalBestVal)
    def get_value(self, pos):
        return self.func(pos)
    # solve: 搜索 value 最小的位置
    def solve(self):
        self.init()
        for _ in range(100):
            self.update()
            print(self.globalBestPos, ':   ', self.globalBestVal)
        return self.globalBestPos, self.globalBestVal
    
# pso = PSO(io_error, 4, [-100, -100, -100, -100], [100, 100, 100, 100])
# for i in range(10):
#     pso.update()
#     print(pso.globalBestPos, ':   ', pso.globalBestVal)
#     # [ 19.52181922 -16.24915936   4.91321575   1.80168999]:    -56881.67205050674
#     # [-24.67409193  77.4303907   92.67414545 -44.23328517] :    -9.51
# plt.plot(pso.historyValues)
# plt.xlabel('Iterations')
# plt.ylabel('Values')
# plt.title('Problem 1')
# plt.show()
import random
import numpy as np
from utils import *
import matplotlib.pyplot as plt

# 有界软约束粒子群最优化算法：约束处理、停机条件
class PSO:
    def __init__(self, func, dim=1, upperBounds=[float('inf')], lowerBounds=[float('inf')]) -> None:
        self.D = dim                        #计算空间的维度
        self.func = func                    #目标函数
        assert(len(upperBounds)==dim)
        assert(len(lowerBounds)==dim)
        self.upperBounds = np.array(upperBounds)
        self.lowerBounds = np.array(lowerBounds)
        self.N = int(pow(3, dim))
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
        prePos = self.curPos
        self.curPos = prePos + vel
        self.curPos = max(self.lowerBounds, min(self.upperBounds, self.curPos))
        self.preVel = self.curPos - prePos
        vals = [self.get_value(pos) for pos in self.curPos]
        for i in range(self.N):
            if vals[i] < self.individualBestVal[i]:
                self.individualBestPos[i] = self.curPos[i]
                self.individualBestVal[i] = vals[i]
        self.globalBestVal = min(self.individualBestVal)
        self.globalBestPos = self.individualBestPos[np.argmax(self.individualBestVal)]
        self.historyValues.append(self.globalBestVal)
    def get_value(self, pos):
        return self.func(pos)
    # solve: 搜索 value 最小的位置
    def solve(self):
        self.init()
        for _ in range(100):
            self.update()
            print(self.globalBestPos, ':   ', self.globalBestVal)
        return self.globalBestPos, self.globalBestVal
    
# pso = PSO(io_error, 4, [-100, -100, -100, -100], [100, 100, 100, 100])
# for i in range(10):
#     pso.update()
#     print(pso.globalBestPos, ':   ', pso.globalBestVal)
#     # [ 19.52181922 -16.24915936   4.91321575   1.80168999]:    -56881.67205050674
#     # [-24.67409193  77.4303907   92.67414545 -44.23328517] :    -9.51
# plt.plot(pso.historyValues)
# plt.xlabel('Iterations')
# plt.ylabel('Values')
# plt.title('Problem 1')
# plt.show()