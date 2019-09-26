from module import *
from NPZD_lib import *


def test_model(x):
    d1 = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
    d1 

    T = 100
    dt = 1/365

    x = fill_free_param(x,d2_init)
    x = normalize_columns(x)
        
    F_i = run_time_evo(euler_forward, T, dt,d1,x)[-1]
    F = np.sum(F_i) - 2*F_i[-1]

    return F

d2_init = np.genfromtxt("d2_init", delimiter='\t', skip_header=1)[:,1:]
x = d2_init.copy().flatten()

x = filter_free_param(x)[:,1]
y = np.array([0]) # i want the totel flow be zero 

constrains = np.ones((len(x),2))
constrains[:,0] = 0


X, F, L, A = gradient_decent(test_model,SGD_basic,x,y,
                             constrains=None,max_iter=100,mu=1,
                             pert_scale=0.0000001,grad_scale=0.000000001)

#print(X)
#print(F)
print(L)