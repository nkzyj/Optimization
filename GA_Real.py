from math import *
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import *


history_best_value = []

# 1. 遗传算法
# 2. powell法 单纯形 坐标轮换 模式搜索 旋转方向法 Rosenbrock搜索法
class GA(object):
    '''
    遗传算法类

    属性：
    D:              变量个数
    bounds:         各变量的上下界
    precisions:     各变量的最低表示精度
    pop_size:       种群大小
    generations:    遗传代数
    Pc:             交叉概率
    Pm:             变异概率

    函数：

    '''
    def __init__(self, n_var, bounds, pop_size=100, generations=20, Pc=0.6, Pm=0.01):
        assert(pop_size%2==0)   # 后面的交叉操作要求种群大小为偶数
        self.dna_size = 0
        self.n_var = n_var
        self.bounds = bounds
        assert(n_var==len(bounds))
        self.pop_size = pop_size
        self.generations = generations
        self.Pc = Pc
        self.Pm = Pm
        self.genes = []
        self.fitness = []
        # 当前种群最优值
        self.curBestGene = None
        self.curBestValue = 0
        # 历史种群最优值
        self.bestGene = None
        self.bestValue = 0
        for i in range(pop_size):
            tmp = []
            for _ in range(self.n_var):
                tmp.append(np.random.random()*(self.bounds[]))
            self.genes.append(tmp)
            self.fitness.append(self.getFitness(tmp))
    # 选择
    def select(self):
        fitness_list = []
        sum_fit = 0
        maxValue = 0
        for i in range(self.pop_size):
            v = self.fitness[i]
            
            if(v > maxValue):
                maxValue = v
                self.curBestGene = self.genes[i]
                self.curBestValue = maxValue

            sum_fit += v
            fitness_list.append(v)
        if(self.curBestValue > self.bestValue):
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
        self.fitness.clear()
        for i in range(self.pop_size):
            self.fitness.append(self.getFitness(self.genes[i]))
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
        return 0
    
    def getFitness(self, gene):
        return 1.0/f_rotate(self.translate(gene))
    
    def solve(self):
        for _ in range(self.generations):
            self.select()
            self.cross()
            self.mutate()
            history_best_value.append(self.bestValue)
        return self.bestGene, self.bestValue
