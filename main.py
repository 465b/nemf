from module import *
from NPZD_lib import *

flag = 'd5'

if flag == 'd5_monte_carlo':

    d0 = np.genfromtxt('initial_values/d1_D05_init.tsv')
    d2_init = np.genfromtxt("initial_values/d2_D05_init.tsv",skip_header=1)[:,1:]
    d1_index = np.where( (d2_init != 0) & (d2_init != -1) & (d2_init != 1) )
    d1 = d2_init
    x = construct_X_from_d0_d1(d1=d1,d1_indexes=d1_index)
    print(x)
    y = np.array([0]) # i want the totel flow be zero 

    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1
    #constrains = np.array([None])

    sample_sets = 5    
    max_iter = 300
    
    X,F,L = init_variable_space_for_adjoint(x,y,max_iter)
    X_stack = np.zeros((sample_sets,) + np.shape(X))
    F_stack = np.zeros((sample_sets,) + np.shape(F))
    L_stack = np.zeros((sample_sets,) + np.shape(L))

    for ii in np.arange(0,sample_sets):
        np.random.seed()
        x = monte_carlo_sample_generator(constrains)
        print(x)
    
        #constrains = np.array([None])

        X, F, L = gradient_decent(standard_fit_model,SGD_basic,euler_forward,
                                     d0,d1,standard_weights_model, y,
                                     d1_indexes = d1_index,
                                     constrains = constrains,mu=1e-6,
                                     gd_max_iter=max_iter,time_evo_max=100,dt_time_evo=1/5,
                                     pert_scale=1e-4,grad_scale=1e-12,
                                     tolerance=1e-5,tail_length_stability_check=10,
                                     start_stability_check=100,
                                     seed=137)
        
        X_stack[ii] = X
        F_stack[ii] = F
        L_stack[ii] = L
        
    stable_ratio = np.shape(X_stack[X_stack > 0])[0]/(
                    np.shape(X_stack)[0] * np.shape(X_stack)[1] * np.shape(X_stack)[2])
    print(stable_ratio)

if flag == 'd5':

    d0 = np.genfromtxt('initial_values/d1_D05_init.tsv')
    d2_init = np.genfromtxt("initial_values/d2_D05_init.tsv",skip_header=1)[:,1:]
    d1_index = np.where( (d2_init != 0) & (d2_init != -1) & (d2_init != 1) )
    d1 = d2_init
    x = construct_X_from_d0_d1(d1=d1,d1_indexes=d1_index)
    print(x)
    y = np.array([0]) # i want the totel flow be zero 

    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1
    #constrains = np.array([None])

    max_iter = 300
    
    X,F,L = init_variable_space_for_adjoint(x,y,max_iter)
    X, F, L = gradient_decent(standard_fit_model,SGD_basic,euler_forward,
                                     d0,d1,standard_weights_model, y,
                                     d1_indexes = d1_index,
                                     constrains = constrains,mu=1e-2,
                                     gd_max_iter=max_iter,time_evo_max=300,dt_time_evo=1/5,
                                     pert_scale=1e-4,grad_scale=1e-13,
                                     tolerance=1e-5,tail_length_stability_check=10,
                                     start_stability_check=100,
                                     seed=137)






if flag == 'd15':
    def d15_model(x,y,z):
        return y

    def test_model(x,
                   tail_length_stability_check=10,tolerance=1e-9):

        d1 = np.genfromtxt("d1_init",delimiter='\t', skip_header=1)[1:]
        d0 = d1
        T = 100
        dt = 1/36

        x = fill_free_param(x,d2_init)
        x = normalize_columns(x)
        method = euler_forward

        F_ij = run_time_evo(method, T,dt,d0,d15_model,d1_weights=x)
        is_stable = verify_stability_time_evolution(F_ij,tolerance)
        F_i = F_ij[-1]
        F = np.array(np.sum(F_i) - 2*F_i[-1])

        return F,is_stable


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
                                 tolerance=1e-9,tail_length_stability_check=10,
                                 seed=137)

if flag == 'NPZD':
    def NPZD_model(x,
                   tail_length_stability_check=10,tolerance=1e-9):
        dt = 1/10
        T = 100

        constants = heinle_2013()
        method = euler_forward
        d1_weights_model = LLM_model

        F_ij = run_time_evo(method, T,dt,x,d1_weights_model,weights_model_constants = constants)
        is_stable = verify_stability_time_evolution(F_ij,tolerance)


        return F_ij[-1],is_stable

    names =      ['N','P','Z','F','D']
    x = np.array([0,  3e-1,  8e-1,  6e-1])
    y = np.array([0, 0, 0, 0]) # i want the totel flow be zero 

    constrains = np.zeros((len(x),2))
    constrains[:,0] = -1
    constrains[:,1] = 1
    #constrains = np.array([None])

    X, F, L = gradient_decent(NPZD_model,SGD_momentum,x,y,
                                 constrains,max_iter=10,mu=1,
                                 pert_scale=1e-2,grad_scale=1e-2,
                                 tolerance=1e-9,tail_length_stability_check=10,
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