import numpy as np
from . import worker
from . import models
from . import decorators

import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',
                    level=logging.DEBUG)



def run_time_evo(integration_scheme, time_evo_max, dt_time_evo, ODE_state,
                 ODE_coeff_model, ODE_coeff=None, stability_rel_tolerance=1e-5,
                 tail_length_stability_check=10, start_stability_check=100):
    
    """ integrates first-order coupled ordinary-differential-equations (ODEs)

    Parameters
    ----------

    integration_scheme : function
        {euler,runge_kutta}
    time_evo_max : positive float
        time span that is integrated
    dt_time_evo : positive float
        time step that is integrated
    ODE_state : numpy.array
        one dimensional set of initial values
    ODE_coeff_model : function
        updates ODE coupling coefficients
        necessary because coupling coefficient are allowed 
        to be 'time' dependent, i.e the coupling is dependent
        on the last ODE_state values (i.e. NPZD)
    ODE_coeff : numpy.array
        2D-square-matrix with ODE coefficients 
    stability_rel_tolerance : positive float
        max allowed fluctuation
        to decide if time evolution is stable 
    tail_length_stability_check : positive integer
        number of entries used from the tail ODE_state from
        which the stability is checked
    start_stability_check : positive integer
        amount of steps before stability is checked 

    Returns
    -------
    ODE_state_log : numpy.array
        two dimensional array containing the results of the 
        time integration for each iteration step
    is_stable : bool
        True if time integration was stable
        False else
    """

    # initialize data arrays
    n_obs = len(ODE_state)
    n_steps = int(time_evo_max/dt_time_evo)
    
    ODE_state_log = np.zeros( (n_steps,n_obs) )
    ODE_state_log[0] = ODE_state

    # calculate the time evolution
    is_stable = False
    for ii in np.arange(1,n_steps):
        # updates ODE coefficients
        ODE_coeff = ODE_coeff_model(ODE_state_log[ii],ODE_coeff)
        # calcu
        ODE_state_log[ii] = integration_scheme(ODE_state_log[ii-1],ODE_coeff,dt_time_evo)

        # repeatedly checks if the solution is stable, if so returns, if not continous
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
        a model, applying a certain method 
        
    Parameters
    ----------
    fit_model : function
        {net_flux_fit_model, direct_fit_model}
        defines how the output of the time evolution get accounted for.
        i.e. the sum of the output is returned or all its elements
    gradient_method : function
        {SGD_basic,SGD_momentum}
        Selects the method used during the gradient descent.
        They differ in their robustness and convergence speed
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    ODE_state : numpy.array
        1D array containing the initial state of the observed quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    y : numpy array
        1D-array containing the desired output of the model in the form
        defined by the fit-model
    ODE_state_indexes : numpy.array
        1D-array containing sets of indices used to select which elements
        of the ODE_state array are optimized. 'None' if none are optimized.
    ODE_coeff_indexes : numpy.array
        1D-array containing sets of indices used to select which elements  
        of the ODE_coeff array are optimized. 'None' if none are optimized.
    constrains : numpy.array
        2D-array containing the upper and lower limit of every free input
        parameter in the shape (len(free_param),2).
    barrier_slope : positive-float
        Defines the slope of the barrier used for the soft constrain.
        Lower numbers, steeper slope. Typically between (0-1].
    gd_max_iter : positive integer
        Maximal amount of iterations allowed in the gradient descent
        algorithm.
    time_evo_max
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    pert_scale : positive float
        Maximal value which the system can be perturbed if necessary
        (i.e. if instability is found). Actual perturbation ranges
        from [0-pert_scal) uniformly distributed.
    grad_scale : positive float
        Scales the step size in the gradient descent. Often also
        referred to as learning rate. Necessary to compensate for the
        "roughness" of the objective function field.
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative fluctuation range in the tail
        of the time evolution. If below, system is called stable.
    tail_length_stability_check : positive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned
    seed : positive integer
        Initializes the random number generator. Used to recreate the
        same set of pseudo-random numbers. Helpfull when debugging.

    Returns
    -------
    free_param : numpy.array
        2D-array containing the set of optimized free parameter,
        stacked along the first axis.
    prediction : numpy.array
        2D-array containing the output of the time evolution,
        stacked along the first axis.
    cost
        1D-array containing the corresponding values of the cost
        function calculated based on the free_param at the same
        first-axis index.   """

    np.random.seed(seed)
    
    free_param_init = worker.filter_free_param(ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
    free_param,prediction,cost = worker.init_variable_space(free_param_init,y,gd_max_iter)
    
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
                """ moves the original set of free parameters in case that any(!)
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
                    fit_model = models.net_flux_fit_model,
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
    
    """ Optimizes a set of randomly generated free parameters and returns
        their optimized values and the corresponding fit-model and cost-
        function output


    Parameters
    ----------
    path_ODE_state_init : string
        Path to a file containing the initial values for the time evolution
        Expects them to be in tab-seperated-format.
    path_ODE_coeff_init : string
        Path to a file containing the coupling coefficients used in 
        the time evolution. Expects them to be in tab-seperated-format.
    y : numpy array
        1D-array containing the desired output of the model in the form
        defined by the fit-model
    fit_model : function
        {net_flux_fit_model, direct_fit_model}
        defines how the output of the time evolution get accounted for.
        i.e. the sum of the output is returned or all its elements
    gradient_method : function
        {SGD_basic,SGD_momentum}
        Selects the method used during the gradient descent.
        They differ in their robustness and convergence speed
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    barrier_slope : positive-float
        Defines the slope of the barrier used for the soft constrain.
        Lower numbers, steeper slope. Typically between (0-1].
    sample_sets : positive integer
        Amount of randomly generated sample sets used as initial free
        parameters
    gd_max_iter : positive integer
        Maximal amount of iterations allowed in the gradient descent
        algorithm.
    time_evo_max
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    pert_scale : positive float
        Maximal value which the system can be perturbed if necessary
        (i.e. if instability is found). Actual perturbation ranges
        from [0-pert_scal) uniformly distributed.
    grad_scale : positive float
        Scales the step size in the gradient descent. Often also
        referred to as learning rate. Necessary to compensate for the
        "roughness" of the objective function field.
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative fluctuation range in the tail
        of the time evolution. If below, system is called stable.
    tail_length_stability_check : positive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned
    seed : positive integer
        Initializes the random number generator. Used to recreate the
        same set of pseudo-random numbers. Helpfull when debugging.

    Returns
    -------
    free_param : numpy.array
        2D-array containing the set of optimized free parameter,
        stacked along the first axis.
    prediction : numpy.array
        2D-array containing the output of the time evolution,
        stacked along the first axis.
    cost
        1D-array containing the corresponding values of the cost
        function calculated based on the free_param at the same
        first-axis index.
    """

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
    prediction_stack = np.zeros((sample_sets,) + np.shape(prediction))
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
            ODE_state,ODE_coeff = worker.fill_free_param(free_param,ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_index)
            free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                        ODE_state,ODE_coeff,ODE_coeff_model, y,
                                        ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        stability_rel_tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            optim_free_param[ii] = free_param
            prediction_stack[ii] = prediction
        
    return optim_free_param,prediction_stack


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
    ODE_coeff = ODE_coeff_model(ODE_state,None)
    ODE_coeff_index = None
    
    free_param = worker.filter_free_param(ODE_state=ODE_state,ODE_state_indexes=ODE_state_indexes)
    
    constrains = np.zeros((len(free_param),2))
    constrains[:,0] = 0
    constrains[:,1] = np.sum(ODE_state)    
    
    free_param,prediction,cost = worker.init_variable_space(free_param,y,gd_max_iter)
    optim_free_param = np.zeros((sample_sets,) + np.shape(free_param))
    prediction_stack = np.zeros((sample_sets,) + np.shape(prediction))
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
            ODE_state,ODE_coeff = worker.fill_free_param(free_param,ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_index)
            free_param, prediction, cost = gradient_decent(fit_model,gradient_method,integration_method,
                                        ODE_state,ODE_coeff,ODE_coeff_model, y,
                                        ODE_state_indexes, ODE_coeff_index,constrains,barrier_slope,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        stability_rel_tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            optim_free_param[ii] = free_param
            prediction_stack[ii] = prediction
        
    return optim_free_param,prediction_stack