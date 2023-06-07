# 信赖域反射算法的实现，主要包括如下要点：
# 1. 梯度、海塞阵的计算
# 2. 子问题的求解（带约束）
# 3. 信赖域的更新策略
import numpy as np
def invMap(idmap):
    n = len(idmap)
    r = [0]*n
    for i in range(n):
        r[idmap[i]] = i
    return r

def swapid(idmap, id1, id2):
    tmp = idmap[id1]
    idmap[id1] = idmap[id2]
    idmap[id2] = tmp
    return idmap

def objective(x):
    # 定义目标函数，根据实际情况编写
    # 这里假设目标函数为 Rosenbrock 函数
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def hessian_numerical(f, x, h=1e-6):
    n = len(x)
    hess = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            # 对于每个元素hess[i, j]，计算两个方向上的一阶偏导数差异
            delta_i = np.zeros(n)
            delta_i[i] = h
            delta_j = np.zeros(n)
            delta_j[j] = h
            
            # 使用中心差分法近似计算二阶偏导数
            hess[i, j] = (f(x + delta_i + delta_j) - f(x + delta_i) - f(x + delta_j) + f(x)) / (h**2)
            
            # 填充对称元素
            if i != j:
                hess[j, i] = hess[i, j]
    
    return hess

def hessian_BFGS(f):
    return 0

def gradient(x):
    # 计算目标函数的梯度，根据实际情况编写
    # 这里假设目标函数为 Rosenbrock 函数
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def hessian(x):
    # 计算目标函数的Hessian矩阵，根据实际情况编写
    # 这里假设目标函数为 Rosenbrock 函数
    hess = np.zeros((2, 2))
    hess[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
    hess[0, 1] = -400 * x[0]
    hess[1, 0] = -400 * x[0]
    hess[1, 1] = 200
    return hess

def solve_subproblem(g, B, Delta):
    n = len(g)
    r = 0
    # 初始化点为左边界
    cHat = -Delta*np.ones_like(g)#边界
    aHat = -Delta*np.ones_like(g)#下界
    bHat =  Delta*np.ones_like(g)#上界
    dHat = cHat#当前位置
    A = set([i for i in range(n)])#边界维度集合
    idmap = [i for i in range(n)]#映射后的第 i 个元素对应于原始的第 idmap[i] 个元素
    BHat = B
    while(True):
        # 检测是否满足K-T条件，需要更新
        gHat = np.matmul(B, dHat) + g
        b_KT = True
        q = None
        for i in set(A):
            if (gHat[i] < 0 and cHat[i] == aHat[i]) or (gHat[i] > 0 and cHat[i] == bHat[i]):
                q = i
                b_KT = False
                break
        if(b_KT):#满足KT条件，直接返回
            # 将d变换为原始顺序
            d = np.zeros_like(dHat)
            for i in range(n):
                d[idmap[i]] = dHat[i]
            return d
        assert(q != None)

        
        # 更新 idmap, g, B, c, d, A, q; 该问题中所有元素上下界均相同
        for i in range(r, n):
            if not i in A:
                idmap = swapid(idmap, i, r)
                r += 1
        for i in range(n):
            gHat[i] = g[idmap[i]]
            dHat[i] = d[idmap[i]]
            cHat[i] = dHat[i]
            for j in range(n):
                BHat[i, j] = B[idmap[i], idmap[j]]
        A.clear()
        A = set([r for i in range(r, n)])
        q = 


    return None


def trust_region_reflective(x0, Delta=1.0, eta=0.1, max_iter=100):
    Delta_max = 5
    x = x0.copy()  # 当前点
    g = gradient(x)  # 当前点的梯度
    B = hessian(x)  # 当前点的Hessian矩阵
    
    for i in range(max_iter):
        # 解决信赖域子问题
        p, rho = solve_subproblem(g, B, Delta)
        
        if rho < 0.25:
            Delta *= 0.5
        elif rho > 0.75 and np.linalg.norm(p) == Delta:
            Delta = min(2 * Delta, Delta_max)
        
        if rho > eta:
            x += p
            g = gradient(x)
            B = hessian(x)
        
        if np.linalg.norm(g) < 1e-6:
            break
    
    return x

# 调用
x = trust_region_reflective(np.array([0.0, 0.0]))
print(x, ': ', objective(x))