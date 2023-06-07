from math import *
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import *
MAX_VALUE = 1e9

# 1. 遗传算法
# 2. powell法 单纯形 坐标轮换 模式搜索 旋转方向法 Rosenbrock搜索法
class GA_Bin(object):
    '''
    遗传算法类

    属性：
    n_var:          变量个数
    bounds:         各变量的上下界
    precisions:     各变量的最低表示精度
    pop_size:       种群大小
    generations:    遗传代数
    Pc:             交叉概率
    Pm:             变异概率

    函数：

    '''
    Population = 1000
    Precision = 0.01
    def __init__(self, f, n_var, bounds, precisions, pop_size=20, generations=10, Pc=0.6, Pm=0.01):
        assert(pop_size%2==0)   # 后面的交叉操作要求种群大小为偶数
        self.func = f
        self.dna_size = 0
        self.n_var = n_var
        self.bounds = bounds
        self.geneSizes = []
        assert(n_var==len(bounds) and n_var==len(precisions))
        for i in range(n_var):
            l, u = bounds[i]
            p = 1.0/precisions[i]
            self.geneSizes.append(floor(log(p*(u-l), 2)) + 1)
            self.dna_size += self.geneSizes[i]
        self.pop_size = pop_size
        self.generations = generations
        self.Pc = Pc
        self.Pm = Pm
    def init(self):
        self.genes = []
        self.vals =  []
        self.fitness = []
        # 当前种群最优值
        self.curBestGene = None
        self.curBestValue = 0
        # 历史种群最优值
        self.bestGene = None
        self.bestValue = float('inf')
        for _ in range(self.pop_size):
            tmp = []
            for _ in range(self.dna_size):
                tmp.append(np.random.randint(0, 2))
            self.genes.append(tmp)
            self.vals.append(self.getVals(tmp))
        self.getFitness()
    # 选择
    def select(self):
        fitness_list = []
        sum_fit = 0
        minValue = float('inf')
        for i in range(self.pop_size):
            v = self.vals[i]
            if(v < minValue):
                minValue = v
                self.curBestGene = self.genes[i]
                self.curBestValue = minValue

            sum_fit += self.fitness[i]
            fitness_list.append(self.fitness[i])
        if(self.curBestValue < self.bestValue):
            self.bestGene = self.curBestGene
            self.bestValue = self.curBestValue
        genes = []
        for i in range(self.pop_size):
            rd = sum_fit*random.random()
            tmp_sum = 0
            for j in range(self.pop_size):
                tmp_sum += fitness_list[j]
                if(tmp_sum >= rd):
                    genes.append(self.genes[j])
                    break
        assert(len(genes) == self.pop_size)
        self.genes = genes
    # 变异
    def mutate(self):
        for i in range(self.pop_size):
            pos = random.randint(0, self.dna_size-1)
            self.genes[i][pos] = 1-self.genes[i][pos]
        self.vals.clear()
        for i in range(self.pop_size):
            self.vals.append(self.getVals(self.genes[i]))
        self.getFitness()
    # 交换
    def cross(self):
        # 奇偶个体发生交换
        for i in range(self.pop_size//2):
            g1 = self.genes[2*i]
            g2 = self.genes[2*i+1]
            pos = random.randint(0, self.dna_size)
            tmp = g2[pos: self.dna_size]
            g2[pos: self.dna_size] = g1[pos: self.dna_size]
            g1[pos: self.dna_size] = tmp
            self.genes[2*i] = g1
            self.genes[2*i+1] = g2
            
    def translate(self, gene):
        pos = 0
        vars = []
        for i in range(self.n_var):
            tmp = gene[pos: pos+self.geneSizes[i]]
            value = 0.0
            for j in range(len(tmp)):
                value += tmp[j]*pow(2, j)
            value = value/(self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            vars.append(value)
            pos += self.geneSizes[i]
        return vars
    
    def getFitness(self):
        self.fitness.clear()
        assert(len(self.vals) == self.pop_size)
        M = max(self.vals)
        for val in self.vals:
            if(M-val<1e-3):
                self.fitness.append(1e-3)
                continue
            self.fitness.append(M-val)
        
    def getVals(self, gene):
        res = self.func(self.translate(gene))
        if res>MAX_VALUE:
            res = MAX_VALUE
        return res
    def solve(self):
        self.init()
        for _ in range(self.generations):
            self.select()
            self.cross()
            self.mutate()
            print('best gene:', self.bestGene)
            print('best value:', self.bestValue)
        return self.translate(self.bestGene), self.bestValue
