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
                    idx_source, idx_sink,
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
    idx_source : list of integers
        list containing the integers of compartments which are constructed
        to be a carbon source 
    idx_sink : list of integers
        list containing the integers of compartments which are designed
        to be a carbon sink 
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
                    integration_scheme, time_evo_max, dt_time_evo, idx_source, idx_sink,
                    constrains,barrier_slope,
                    stability_rel_tolerance,tail_length_stability_check, start_stability_check)
        if not is_stable:
            """ moves the pararun_timemeter set randomly in the hope to find a stable solution """
            free_param[ii] = worker.perturb(free_param[ii],pert_scale)
        else:
            """ calculates the local gradient by evaluation the time evolution at surrounding
                free parameters set points """
            gradient,is_stable = worker.local_gradient(free_param[:ii+1],y,fit_model,integration_scheme,
                                        ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes,
                                        ODE_coeff_model,
                                        idx_source, idx_sink, constrains,barrier_slope,
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
def dn_monte_carlo(path_model_configuration,
					gradient_method = models.SGD_basic,
					barrier_slope=1e-6,
					sample_sets = 3,
					gd_max_iter=10,
					pert_scale=1e-4,
					grad_scale=1e-12,
					seed=137):

	""" Optimizes a set of randomly generated free parameters and returns
		their optimized values and the corresponding fit-model and cost-
		function output 
	
	Parameters
	----------
	path_model_configuration : string
		Path to a file containing the coupling coefficients used in 
		the time evolution. Expects them to be in tab-seperated-format.
	gradient_method : function
		{SGD_basic,SGD_momentum}
		Selects the method used during the gradient descent.
		They differ in their robustness and convergence speed
	barrier_slope : positive-float
		Defines the slope of the barrier used for the soft constrain.
		Lower numbers, steeper slope. Typically between (0-1].
	sample_sets : positive integer
		Amount of randomly generated sample sets used as initial free
		parameters
	gd_max_iter : positive integer
		Maximal amount of iterations allowed in the gradient descent
		algorithm.
	pert_scale : positive float
		Maximal value which the system can be perturbed if necessary
		(i.e. if instability is found). Actual perturbation ranges
		from [0-pert_scal) uniformly distributed.
	grad_scale : positive float
		Scales the step size in the gradient descent. Often also
		referred to as learning rate. Necessary to compensate for the
		"roughness" of the objective function field.
	seed : positive integer
		Initializes the random number generator. Used to recreate the
		same set of pseudo-random numbers. Helpfull when debugging.

	Returns
	-------
	model_configuration.log : dict
		contains the results for every run
		(parameters, prediction, cost)
	"""


	# seeds random generator to create reproducible runs
	np.random.seed(seed)

	# initializes model configuration 
	# (contains information about the optimized model)
	model_configuration = models.model_class(path_model_configuration)
	model_configuration.initialize_log(sample_sets, gd_max_iter)
	
	# runs the optimization with the initial values read from file
	if sample_sets == 0:	
	
		# fetches the parameters and their constraints from model config
		constraints = model_configuration.to_grad_method()[1]
		parameters = worker.monte_carlo_sample_generator(constraints)
		
		# runs the gradient descent for the generated sample set 
		parameters, prediction, cost = gradient_descent(model_configuration,
										parameters, constraints, gradient_method,
										barrier_slope, gd_max_iter,
										pert_scale, grad_scale)
		
		# updates log with the generated results
		log_dict = {'parameters': parameters,
					'prediction': prediction,
					'cost': cost}
		model_configuration.log = log_dict
	
	# runs the optimization with randomly chosen values
	# values are picked from inside the allowed optimization range
	else:
		for ii in np.arange(0,sample_sets):
			print('Monte Carlo Sample #{}'.format(ii))
	
			# fetches the parameters and their constraints from model config
			constraints = model_configuration.to_grad_method()[1]
			parameters = worker.monte_carlo_sample_generator(constraints)
			
			# runs the gradient descent for the generated sample set 
			param_stack, prediction_stack, cost_stack = gradient_descent(
				model_configuration, parameters,
				constraints, gradient_method,
				barrier_slope, gd_max_iter,
				pert_scale, grad_scale) 
			
			# updates log with the generated results
			model_configuration.to_log(param_stack, prediction_stack, cost_stack)
			
			# updates the state of the optimization run
			model_configuration.log['monte_carlo_idx'] = ii
			model_configuration.log['gradient_idx'] = 0

	return model_configuration.log