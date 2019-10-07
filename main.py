from module import *
from NPZD_lib import *

flag = 'NPZD'

if flag == 'd15':
    def d15_model(x,y,z):
        return y

    def test_model(x):
        d1 = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
        d0 = d1
        T = 100
        dt = 1/36

        x = fill_free_param(x,d2_init)
        x = normalize_columns(x)
        method = euler_forward

        F_i = run_time_evo(method, T,dt,d0,d15_model,d1_weights=x)[-1]
        F = np.array(np.sum(F_i) - 2*F_i[-1])

        return F


    d2_init = np.genfromtxt("d2_init", delimiter='\t', skip_header=1)[:,1:]
    x = d2_init.copy().flatten()
    x = filter_free_param(x)[:,1]
    y = np.array([0]) # i want the totel flow be zero 

    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1
    #constrains = np.array([None])

    X, F, L = gradient_decent(test_model,SGD_basic,x,y,
                                 constrains,max_iter=40,mu=1e-5,
                                 pert_scale=1e-4,grad_scale=1e-9,
                                 seed=137)

if flag == 'NPZD':
    def NPZD_model(x):
        dt = 1/10
        T = 100

        constants = heinle_2013()
        method = euler_forward
        d1_weights_model = LLM_model

        F_i = run_time_evo(method, T,dt,x,d1_weights_model,weights_model_constants = constants)[-1]

        return F_i

    names =      ['N','P','Z','F','D']
    x = np.array([-2,  3e-1,  8e-1,  6e-1])
    y = np.array([0, 0, 0, 0]) # i want the totel flow be zero 

    constrains = np.zeros((len(x),2))
    constrains[:,0] = -1
    constrains[:,1] = 1
    #constrains = np.array([None])

    X, F, L = gradient_decent(NPZD_model,SGD_momentum,x,y,
                                 constrains,max_iter=200,mu=0.1,
                                 pert_scale=1e-2,grad_scale=1e-1,
                                 seed=141)

    
plt.figure()
plt.title('Free Input Parameter')
plt.plot(X[:-1])
plt.ylabel('Input value (arb. u.)')
plt.xlabel('Iteration Step')
plt.show()

plt.figure()
plt.title('Time Evolution Output')
plt.plot(F[:-1])
plt.ylabel('Output value (arb. u.)')
plt.xlabel('Iteration Step')
plt.show()

plt.figure()
plt.title('Loss function over time')
plt.plot(L[:-1])
plt.ylabel('Loss function (arb. u.)')
plt.xlabel('Iteration Step')
plt.show()