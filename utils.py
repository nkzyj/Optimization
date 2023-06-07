import numpy as np
import matplotlib.pyplot as plt
import control as ct
import random
from math import *
import scipy
# ----------------------------------------分数阶系统辨识----------------------------------------
# 传递函数辨识 P(s) -> X(s)  阶次为 n，数据 m 条
# 1. 整数阶
# 2. 分数阶，固定阶次
# 3. 分数阶，限定阶次范围

# [7.08, 2.78, -5.07, 2.42, 2.44, 1.34, -0.24, 0.36]
# θ = [a_i, α_i,...] G_θ(s) = 1/(a_i×s^α_i) X(s) = G(s)P(s)   a_i D^{α_i}x(t) = p(t)
path = r'F:\zyj\Dataset\SCNT-Enucleation\Institute of Animal Sciences\002/data.txt'
npdata = np.loadtxt(path, skiprows=1)
t = npdata[:, 0]
p = npdata[:, 1]
x = npdata[:, 2]
if(len(t) > 1):
    h = t[1] - t[0]
else:
    h = 1.0
pass
# 计算分数阶微积分，对于不同的alpha，有特定的权重数组与之对应；
def glfdiff(y, alpha):
    m = len(y)
    w = [0.0]*m
    w[0] = 1
    dy = [0.0]*m
    dy[0] = y[0]/pow(h, alpha)
    for i in range(1, m):
        w[i] = w[i-1]*(1-(alpha+1)/i)
        dy[i] = 0
        for j in range(i+1):
            dy[i] += w[j]*y[i-j]
        dy[i]/=pow(h, alpha)
    return np.array(dy)

# 将 u(t) 通过传递函数变换为 y(t)：整数阶系统
def io_system_response(den, num=[1], t=t, u=p):
    sys = ct.tf(num, den)
    y = ct.forced_response(sys, t, u)
    return y.outputs
# 将 u(t) 通过传递函数变换为 y(t)：分数阶系统 paras: [a1, ..., an, α1, ..., αn]
def fo_system_response(paras, t=t, u=p):
    assert(len(paras)%2==0)
    n = len(paras)//2
    m = len(t)
    a = paras[0:n]
    alpha = paras[n:]

    w = np.ones([n, m], dtype=np.float32)
    y = np.zeros(m, dtype=np.float32)
    for i in range(n):
        for j in range(1, m):
            w[i, j] = w[i, j-1]*(1-(alpha[i]+1)/j)
    for k in range(m):
        c = 0.0
        for i in range(n):
            ci = a[i]/pow(h, alpha[i])
            c += ci
            for j in range(1, k+1):
                y[k] += ci*w[i, j]*y[k-j]
        y[k] = (u[k]-y[k])/c
    return y
# 对于参数 den, 整数阶辨识效果的误差平方和
def io_error(den, y=x):
    return np.sqrt(np.sum((y-io_system_response(den))**2)/len(y))
# 对于参数 paras, 分数阶辨识效果的误差平方和
def fo_error(paras, y=x):
    return np.sqrt(np.sum((y-fo_system_response(paras))**2)/len(y))

# para_io = [3.24, -0.32, 1.51, -0.22]
# para_fo = [7.08, -5.07, 2.44, -0.24, 2.78, 2.42, 1.34, 0.36]
# y_io = io_system_response(para_io)
# y_fo = fo_system_response(para_fo)
# e_io = io_error(para_io)
# e_fo = fo_error(para_fo)
# plt.plot(t, y_fo)
# plt.plot(t, x)
# plt.figure()
# plt.plot(t, y_io)
# plt.plot(t, x)
# plt.show()


# ----------------------------------------随机旋转优化----------------------------------------
N = 100 #随机模拟器的实验次数
eps = 5.0    #误差带为 5°
def f_rotate(vars):
    # 根据翻译后的自变量计算适应度
    sigma = vars[0]
    count = 0
    for i in range(N):
        p = init_position()
        while(abs(get_angleZ(p))>eps):
            k, delta, M = get_rotation_matrix(sigma)
            p = np.matmul(M, p)
            # viz(p)
            count += 1
    avg_count = 1.0*count/N
    return avg_count
# 初始化极体位置的函数
def init_position():
    # dS = r^2 dθ sin θ dφ = -r^2 dcosθ dφ
    phi = 2*pi*random.random() # 0..2pi 随机
    theta = acos(2*random.random()-1) # 0..pi 随机
    return np.array([[sin(theta)*cos(phi)], [sin(theta)*sin(phi)], [cos(theta)]])
# 获取旋转矩阵
def get_rotation_matrix(sigma, alpha=None):
    # 如果α == None，则α~U(0, π)  k = [sin(α), 0, cos(α)]T
    if(alpha==None):
        alpha = 2*pi*random.random() # 0..2π 随机
    kx = sin(alpha)
    ky = 0
    kz = cos(alpha)
    k = np.array([kx, ky, kz])
    delta = random.gauss(0, sigma)
    sd = sin(delta)#sin δ
    cd = cos(delta)#cos δ
    vd = 1-cd           #1-cos δ
    M = np.array([[kx*kx*vd+cd,     kx*ky*vd-kz*sd,     kx*kz*vd+ky*sd], 
                  [kx*ky*vd+kz*sd,  ky*ky*vd+cd,        ky*kz*vd-kx*sd],
                  [kx*kz*vd-ky*sd,  ky*kz*vd+kx*sd,     kz*kz*vd+cd]])
    return k, delta, M
# 计算向量与z轴的夹角
def get_angleZ(k):
    assert(k.shape==(3, 1))
    return 180*atan2(k[2], sqrt(k[0]*k[0]+k[1]*k[1]))/pi
def viz(p):
    ax = plt.figure()
    ax = plt.axes(projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='r')

    ax.scatter3D(p[0],p[1],p[2], cmap='Blues')
    plt.show()

# ----------------------------------------  测试函数  ----------------------------------------
def f_Langermann(x):
    a = [3,5,2,1,7]
    b = [5,2,1,4,9]
    c = [1,2,5,2,3]
    sum = 0.0
    for i in range(5):
        e = ((x[0]-a[i])*(x[0]-a[i])+(x[1]-b[i])*(x[1]-b[i]))/pi
        theta = pi*((x[0]-a[i])*(x[0]-a[i])+(x[1]-b[i])*(x[1]-b[i]))
        sum -= c[i]*cos(theta)*(exp(-e))
    return sum

def f_Damavandi(x):
    return (2.0+pow(x[0]-7, 2)+2*pow(x[1]-7, 2))*(1.0-pow(abs(np.sinc(x[0]-2)*np.sinc(x[1]-2)), 5))

def plot_function():
    xx = np.arange(0,15,0.1)
    yy = np.arange(0,15,0.1)
    X, Y = np.meshgrid(xx, yy)
    z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i,j] = f_Damavandi([X[i,j], Y[i,j]])
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, z, cmap='rainbow')
    plt.show()

# scipy.optimize.minimize(fun, x0)