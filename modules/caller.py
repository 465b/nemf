import numpy as np
from . import worker
from . import models
from . import decorators

import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)



def run_time_evo(integration_scheme, time_evo_max, dt_time_evo, ODE_state, ODE_coeff_model, 
                 ODE_coeff=None,
                 stability_rel_tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
    
    """ integrates first order ODE

        integration_scheme: {euler,runge_kutta}

        time_evo_max: postive float, time span that is integrated
        dt_time_evo: postive float, time step that is integrated

        ODE_state: 1D-numpy.array, set of initial values
        ODE_coeff_model: function, updates ODE_coeff (ODE coupling coefficients).
                          necessaray becaus coupling coefficient are allowed 
                          to be 'time' dependent, i.e the coupling is dependent
                          on the last ODE_state values (i.e. NPZD)
        ODE_coeff: 2D-square-numpy.array with ODE coefficients 
        
        stability_rel_tolerance: positive float, max allowed fluctuation to decide if 
                   time evolution is stable 
        tail_length_stability_check: postive integer, number of entries used from the tail ODE_state
                                     from which the stablity is checked
        start_stability_check: positive integer, amount of steps before stability is checked """

    # initialize data arrays
    n_obs = len(ODE_state)
    n_steps = int(time_evo_max/dt_time_evo)
    
    ODE_state_log = np.zeros( (n_steps,n_obs) )
    ODE_state_log[0] = ODE_state

    # calculate the time evolution
    time_step = 0
    is_stable = False
    for ii in np.arange(1,n_steps):
        ODE_coeff = ODE_coeff_model(ODE_state_log[ii],ODE_coeff)
        ODE_state_log[ii] = integration_scheme(ODE_state_log[ii-1],ODE_coeff,time_step,dt_time_evo)

        if ((ii >= start_stability_check) & (ii%tail_length_stability_check == 0) &
            (stability_rel_tolerance != 0)):
            is_stable = worker.verify_stability_time_evolution(ODE_state_log[:ii+1],
                                            stability_rel_tolerance,tail_length_stability_check)
        
            if is_stable:
                return ODE_state_log[:ii+1], is_stable
        
    
    return ODE_state_log, is_stable


# Top level Routine

def gradient_decent(fit_model,gradient_method,integration_scheme,
                    ODE_state,ODE_coeff, ODE_coeff_model, y,
                    ODE_state_indexes = None, ODE_coeff_indexes = None,
                    constrains=np.array([None]), barrier_slope=1e-2,
                    gd_max_iter=100,time_evo_max=100,dt_time_evo=0.2, 
                    pert_scale=1e-5,grad_scale=1e-9,
                    stability_rel_tolerance=1e-9,tail_length_stability_check=10, 
                    start_stability_check = 100,seed = 137):
    """ framework for applying a gradient decent approach to a 
        a model, applying a certain method """
        

    np.random.seed(seed)
    
    free_param = worker.filter_free_param(ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
    free_param,prediction,cost = worker.init_variable_space(free_param,y,gd_max_iter)
    
    # ii keeps track of the position in the output array
    # jj keeps track to not exceed max iterations
    ii = 0; jj = 0
    
    while jj < gd_max_iter-1:

        """ makes sure that all points in the parameter set are inside of the search space
            and if not moves them back into it """
        free_param[ii] = worker.barrier_hard_enforcement(free_param[ii],ii,constrains,pert_scale)
        ODE_state,ODE_coeff = worker.fill_free_param(free_param[ii],ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        ODE_coeff = worker.normalize_columns(ODE_coeff)
        free_param[ii] = worker.filter_free_param(ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        

        """ evaluate the system by iterating the time evolution until a stable solution (prediction) is found
            and constructing the cost function (cost) """
        prediction[ii],cost[ii],is_stable = worker.prediction_and_costfunction(
                    free_param[ii],ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
                    integration_scheme, time_evo_max, dt_time_evo,constrains,barrier_slope,
                    stability_rel_tolerance,tail_length_stability_check, start_stability_check)
        if not is_stable:
            """ moves the parameter set randomly in the hope to find a stable solution """
            free_param[ii] = worker.perturb(free_param[ii],pert_scale)
        else:
            """ calculates the local gradient by evaluation the time evolution at surrounding
                free parameters set points """
            gradient,is_stable = worker.local_gradient(free_param[:ii+1],y,fit_model,integration_scheme,
                                        ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes,
                                        ODE_coeff_model,constrains,barrier_slope,
                                        pert_scale,
                                        time_evo_max, dt_time_evo,
                                        stability_rel_tolerance,tail_length_stability_check,
                                        start_stability_check)
            if not is_stable:
                """ moves the originial set of free parameters in case that any(!)
                    of the surrounding points used in the calculation of the
                    local gradient is unstable """
                free_param[ii] = worker.perturb(free_param[ii],pert_scale)
            else:
                """ applying a decent model to find a new and hopefully
                    better set of free parameters """
                free_param[ii+1] = gradient_method(free_param[:ii+1],gradient,grad_scale)
                ii += 1
        jj += 1

            
    return free_param, prediction, cost


## monte carlo methods

@decorators.log_input_output
def dn_monte_carlo(path_ODE_state_init,path_ODE_coeff_init,y,
                    fit_model = models.standard_fit_model,
                    gradient_method = models.SGD_basic,
                    integration_method = models.euler_forward,
                    ODE_coeff_model = models.standard_weights_model,
                    barrier_slope=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    stability_rel_tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    ODE_state = np.genfromtxt(path_ODE_state_init)
    ODE_state_indexes = None
    ODE_coeff = np.genfromtxt(path_ODE_coeff_init)
    ODE_coeff_index = np.where( (ODE_coeff != 0) & (ODE_coeff != -1) & (ODE_coeff != 1) )


    free_param = worker.filter_free_param(ODE_coeff=ODE_coeff,ODE_coeff_indexes=ODE_coeff_index)
    
    constrains = np.zeros((len(free_param),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    free_param,prediction,cost = worker.init_variable_space(free_param,y,gd_max_iter)
    optim_free_param = np.zeros((sample_sets,) + np.shape(free_param))
    F_stack = np.zeros((sample_sets,) + np.shape(prediction))
    #L_stack = np.zeros((sample_sets,) + np.shape(cost))

    if sample_sets == 0 :
        free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                ODE_state,ODE_coeff,ODE_coeff_model, y,
                                ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                stability_rel_tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            free_param = worker.monte_carlo_sample_generator(constrains)
            free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                        ODE_state,ODE_coeff,ODE_coeff_model, y,
                                        ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        stability_rel_tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            optim_free_param[ii] = free_param
            F_stack[ii] = prediction
        
    return optim_free_param,F_stack


@decorators.log_input_output
def NPZD_monte_carlo(path_ODE_state_init,path_ODE_coeff_init,y,
                    fit_model = models.direct_fit_model,
                    gradient_method = models.SGD_basic,
                    integration_method = models.euler_forward,
                    ODE_coeff_model = models.LLM_model,
                    barrier_slope=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    stability_rel_tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    ODE_state = np.genfromtxt(path_ODE_state_init)
    ODE_state_indexes = np.where(ODE_state == ODE_state)
    ODE_coeff = ODE_coeff_model
    ODE_coeff_index = None
    
    free_param = worker.filter_free_param(ODE_state=ODE_state,ODE_state_indexes=ODE_state_indexes)
    
    constrains = np.zeros((len(free_param),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    free_param,prediction,cost = worker.init_variable_space(free_param,y,gd_max_iter)
    optim_free_param = np.zeros((sample_sets,) + np.shape(free_param))
    F_stack = np.zeros((sample_sets,) + np.shape(prediction))
    #L_stack = np.zeros((sample_sets,) + np.shape(cost))

    if sample_sets == 0 :
        free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                ODE_state,ODE_coeff,ODE_coeff_model, y,
                                ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                stability_rel_tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)
        
        return free_param,prediction,cost

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            free_param = worker.monte_carlo_sample_generator(constrains)
            free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                        ODE_state,ODE_coeff,ODE_coeff_model, y,
                                        ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        stability_rel_tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            optim_free_param[ii] = free_param
            F_stack[ii] = prediction
        
    return optim_free_param,F_stack