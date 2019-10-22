import numpy as np
from . import worker
from . import models
from . import decorators

import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)



def run_time_evo(integration_scheme, T, dt, d0, d1_weights_model, 
                 d1_weights=None,
                 tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
                 
    """ integrates first order ODE

        integration_scheme: {euler,runge_kutta}

        T: postive float, time span that is integrated
        dt: postive float, time step that is integrated

        d0: 1D-numpy.array, set of initial values
        d1_weights_model: function, updates d1 (ODE coupling coefficients).
                          necessaray becaus coupling coefficient are allowed 
                          to be 'time' dependent, i.e the coupling is dependent
                          on the last d0 values (i.e. NPZD)
        d1_weights: 2D-square-numpy.array with ODE coefficients 
        
        tolerance: positive float, max allowed fluctuation to decide if 
                   time evolution is stable 
        tail_length_stability_check: postive integer, number of entries used from the tail d0
                                     from which the stablity is checked
        start_stability_check: positive integer, amount of steps before stability is checked """

    # initialize data arrays
    n_obs = len(d0)
    n_steps = int(T/dt)
    
    d0_log = np.zeros( (n_steps,n_obs) )
    d0_log[0] = d0

    # calculate the time evolution
    time_step = 0
    is_stable = False
    for ii in np.arange(1,n_steps):
        d1_weights = d1_weights_model(d0_log[ii],d1_weights)
        d0_log[ii] = integration_scheme(d0_log[ii-1],d1_weights,time_step,dt)

        if ((ii >= start_stability_check) & (ii%tail_length_stability_check == 0) &
            (tolerance != 0)):
            is_stable = worker.verify_stability_time_evolution(d0_log[:ii+1],
                                            tolerance,tail_length_stability_check)
        
            if is_stable:
                return d0_log[:ii+1], is_stable
        
    
    return d0_log, is_stable


# Top level Routine

def gradient_decent(fit_model,gradient_method,integration_scheme,
                    d0,d1, d1_weights_model, y,
                    d0_indexes = None, d1_indexes = None,
                    constrains=np.array([None]), mu=1e-2,
                    gd_max_iter=100,time_evo_max=100,dt_time_evo=0.2, 
                    pert_scale=1e-5,grad_scale=1e-9,
                    tolerance=1e-9,tail_length_stability_check=10, 
                    start_stability_check = 100,seed = 137):
    """ framework for applying a gradient decent approach to a 
        a model, applying a certain method 
        
        x: initial guess for the parameter set
        y: expected outcome of the model F(x)"""    

    np.random.seed(seed)
    
    x = worker.construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)
    X,F,J = worker.init_variable_space_for_adjoint(x,y,gd_max_iter)
    
    # ii keeps track of the position in the output array
    # jj keeps track to not exceed max iterations
    ii = 0; jj = 0
    
    while jj < gd_max_iter-1:
        if ii == 0:
            X[ii] = worker.barrier_hard_enforcement(X[ii],ii,constrains,pert_scale)
            d0,d1 = worker.fill_X_into_d0_d1(X[ii],d0,d1,d0_indexes,d1_indexes)
            d1 = worker.normalize_columns(d1)
            X[ii] = worker.construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)

            F[ii],J[ii],is_stable = worker.prediction_and_costfunction(
                        X[ii],d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)
            if ((not is_stable) & (jj == 0)):
                warn_string = ('Initial conditions do not result in a stable model output. X: {}'.format(x))
                logging.debug(warn_string)
                X[ii] = worker.add_pertubation(X[ii],pert_scale)
            else:
                X[ii+1] = worker.add_pertubation(X[ii],pert_scale)
                ii += 1
        
        else:
            """ evaluate the system by iterating until a stable solution (F) is found
                and constructing the cost function (J) """
            X[ii] = worker.barrier_hard_enforcement(X[ii],ii,constrains,pert_scale)
            d0,d1 = worker.fill_X_into_d0_d1(X[ii],d0,d1,d0_indexes,d1_indexes)
            d1 = worker.normalize_columns(d1)
            X[ii] = worker.construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)
            
            F[ii],J[ii],is_stable = worker.prediction_and_costfunction(
                        X[ii],d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)
            if not is_stable:
                X[ii] = worker.add_pertubation(X[ii],pert_scale)
            else:
                """ applying a decent model to find a new ( and better)
                    input variable """
                gradient,is_stable = worker.local_gradient(X[:ii+1],y,fit_model,integration_scheme,
                                          d0,d1,d0_indexes,d1_indexes,
                                          d1_weights_model,constrains,mu,
                                          time_evo_max, dt_time_evo,
                                          tolerance=1e-6,tail_length_stability_check=10,
                                          start_stability_check=100)
                if not is_stable:
                    X[ii] = worker.add_pertubation(X[ii],pert_scale)
                else:
                    X[ii+1] = gradient_method(X[:ii+1],gradient,grad_scale)
                    ii += 1
        jj += 1

            
    return X, F, J


## monte carlo methods

@decorators.log_input_output
def dn_monte_carlo(path_d0_init,path_d1_init,y,
                    fit_model = models.standard_fit_model,
                    gradient_method = models.SGD_basic,
                    integration_method = models.euler_forward,
                    d1_weights_model = models.standard_weights_model,
                    mu=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    d0 = np.genfromtxt(path_d0_init)
    d0_indexes = None
    d1 = np.genfromtxt(path_d1_init)
    d1_index = np.where( (d1 != 0) & (d1 != -1) & (d1 != 1) )


    x = worker.construct_X_from_d0_d1(d1=d1,d1_indexes=d1_index)
    
    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    X,F,L = worker.init_variable_space_for_adjoint(x,y,gd_max_iter)
    X_stack = np.zeros((sample_sets,) + np.shape(X))
    F_stack = np.zeros((sample_sets,) + np.shape(F))
    #L_stack = np.zeros((sample_sets,) + np.shape(L))

    if sample_sets == 0 :
        X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                d0,d1,d1_weights_model, y,
                                d0_indexes, d1_index,constrains,mu,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            x = worker.monte_carlo_sample_generator(constrains)
            X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                        d0,d1,d1_weights_model, y,
                                        d0_indexes, d1_index,constrains,mu,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            X_stack[ii] = X
            F_stack[ii] = F
        
    return X_stack,F_stack


@decorators.log_input_output
def NPZD_monte_carlo(path_d0_init,path_d1_init,y,
                    fit_model = models.direct_fit_model,
                    gradient_method = models.SGD_basic,
                    integration_method = models.euler_forward,
                    d1_weights_model = models.LLM_model,
                    mu=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    d0 = np.genfromtxt(path_d0_init)
    d0_indexes = np.where(d0 == d0)
    d1 = d1_weights_model
    d1_index = None
    
    x = worker.construct_X_from_d0_d1(d0=d0,d0_indexes=d0_indexes)
    
    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    X,F,L = worker.init_variable_space_for_adjoint(x,y,gd_max_iter)
    X_stack = np.zeros((sample_sets,) + np.shape(X))
    F_stack = np.zeros((sample_sets,) + np.shape(F))
    #L_stack = np.zeros((sample_sets,) + np.shape(L))

    if sample_sets == 0 :
        X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                d0,d1,d1_weights_model, y,
                                d0_indexes, d1_index,constrains,mu,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)
        
        return X,F,L

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            x = worker.monte_carlo_sample_generator(constrains)
            X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                        d0,d1,d1_weights_model, y,
                                        d0_indexes, d1_index,constrains,mu,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            X_stack[ii] = X
            F_stack[ii] = F
        
    return X_stack,F_stack