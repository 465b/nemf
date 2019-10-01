from module import *
from NPZD_lib import *

# time steps in days NOT YEARS!
def NPZD_model(x):
    dt = 1/10
    T = 100

    constants = heinle_2013()
    method = euler_forward
    d1_weights_model = LLM_model

    F_i = run_time_evo(method, T,dt,x,d1_weights_model,weights_model_constants = constants)[-1]

    return F_i

names =      ['N','P','Z','F','D']
x = np.array([1e-1,  1,  1,  1e-1])
y = np.array([1, 1, 1, 1]) # i want the totel flow be zero 

#constrains = np.zeros((len(x),2))
#constrains[:,0] = -np.inf
#constrains[:,1] = np.inf
constrains = np.array([None])

X, F, L = gradient_decent(NPZD_model,SGD_basic,x,y,
                             constrains,max_iter=4,mu=0.0001,
                             pert_scale=0.0001,grad_scale=1e-1)

#print(X)
#print(F)
#print(L)

'''
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

print(X)
print(F)
print(L)
'''