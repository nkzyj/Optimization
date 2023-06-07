from GA_Bin import *
from PSO_soft_constraint import *
# from TrustRegion import *

# 各个问题的初始猜测值
# fit: [1, ...]
# rotate: 0
# Langermann: [1, 1]
# Damavandi: [1, 1]
x0_fit = np.zeros(8)
x0_rotate = 0
x0_Langermann = [1, 1]
x0_Damavandi = [1, 1]


a = np.array([])
# -----------------------------------信赖域算法测试-----------------------------------
# trust_vals = []
# for i in range(10):
#     # res = scipy.optimize.minimize(f_Langermann, [2, 2], method='trust-constr')
#     # max:  -1.49   min: -1.49  mean:  -1.49    std: 0.0
#     res = scipy.optimize.minimize(f_Damavandi, [1, 1], method='trust-constr')
#     # max:  0/2     min: 0/2    mean:   0/2     std: 0.0
#     # res = scipy.optimize.minimize(f_rotate, [0], method='trust-constr')
#     # res = scipy.optimize.minimize(f_Langermann, [1, 1], method='trust-constr')
#     trust_vals.append(res.fun)
# a = np.array(trust_vals)

# -----------------------------------粒子群测试-----------------------------------
# pso = PSO_Soft(io_error, 4, [-100, -100, -100, -100], [100, 100, 100, 100])  # max: 41.30 min: 17.05 mean: 0.55 std: 6.66
# pso = PSO_Soft(f_Langermann, 2, [0, 0], [10, 10]) # max:  -2.90  min:  -5.04  mean:  -4.29 std:  0.67
pso = PSO_Soft(f_Damavandi, 2, [0, 0], [10, 10])  # max:   2.30  min:  2.00   mean:  2.07  std:  0.09
# pso = PSO_Soft(f_rotate, 1, [0], [pi])            # max:   9.97  min:  8.31   mean:  9.19  std:  0.53
pso_vals = []
for i in range(10):
    _, v = pso.solve()
    pso_vals.append(v)
a = np.array(pso_vals)


# -----------------------------------遗传算法测试-----------------------------------
# ga = GA_Bin(f_rotate, 1, [[0, 3.14]], [0.01]) # max: 9.59  min:  8.27  mean:  8.97  std:  0.42
# ga = GA_Bin(f_Langermann, 2, [[0, 10], [0, 10]], [0.1, 0.1])  # max: -1.37  min:  -5.03  mean:  -2.70  std:  1.02
# ga = GA_Bin(f_Damavandi, 2, [[0, 15], [0, 15]], [0.1, 0.1])     # max: 4.03  min:  2.12  mean:  2.54  std:  0.57
# ga = GA_Bin(io_error, 4, [[-10, 10], [-10, 10], [-10, 10], [-10, 10]], [0.1, 0.1, 0.1, 0.1])
# max: 82.46 min: 41.15 mean: 63.56 std: 11.77
# ga_vals = []
# for _ in range(10):
#     _, v = ga.solve()
#     ga_vals.append(v)
# a = np.array(ga_vals)

print('max: ', a.max(), '\tmin: ', a.min(), '\tmean: ', a.mean(), '\tstd: ', a.std())
