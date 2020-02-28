import numpy as np
from . import worker
from . import models
from . import decorators

import logging
import warnings
logging.basicConfig(filename='carbonflux_inverse_model.log',
					level=logging.DEBUG)



def run_time_evo(model_configuration, integration_scheme, time_evo_max,
				 dt_time_evo, ode_state, ode_coeff_model, ode_coeff=None,
				 stability_rel_tolerance=1e-5, tail_length_stability_check=10,
				 start_stability_check=100):
	
	""" integrates first-order coupled ordinary-differential-equations (odes)

	Parameters
	----------
	model_configuration : object
			contains all the information and necessary methods
			of the optimized model
	integration_scheme: function
			{euler_forward, runge_kutta}
			Selects which method is used in the integration of the time evolution.
			Euler is of first order, Runge-Kutta of second
	time_evo_max
		Maximal amount of iterations allowed in the time evolution.
		Has the same unit as the one used in the initial ODE_state
	dt_time_evo
		Size of time step used in the time evolution.
		Has the same unit as the one used in the initial ODE_state
	ode_state : numpy.array
		1D array containing the initial state of the observed quantities
		in the ODE. Often also referred to as initial conditions.
	ode_coeff : numpy.array
		2d-square-matrix containing the coefficients of the ODE
	ode_coeff_model : function
		selects the function used for the calculation of the ODE
		coefficients. I.e. if dependencies of the current state are present.
		If no dependency is present use 'standard_weights_model'
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

	Returns
	-------
	ode_state_log : numpy.array
		two dimensional array containing the results of the 
		time integration for each iteration step
	is_stable : bool
		True if time integration was stable
		False else
	"""

	# initialize data arrays
	n_obs = len(ode_state)
	n_steps = int(time_evo_max/dt_time_evo)
	
	ode_state_log = np.zeros( (n_steps,n_obs) )
	ode_state_log[0] = ode_state

	# calculate the time evolution
	is_stable = False
	for ii in np.arange(1,n_steps):
		# updates ode coefficients
		ode_coeff = ode_coeff_model(model_configuration)
		# calculates next time step
		ode_state_log[ii] = integration_scheme(ode_state_log[ii-1],ode_coeff,dt_time_evo)
		model_configuration.from_ode(ode_state_log[ii])
		# repeatedly checks if the solution is stable, if so returns, if not continuos
		if ( (ii >= int(start_stability_check/dt_time_evo))
			& (ii%tail_length_stability_check == 0) 
			& (stability_rel_tolerance != 0)):
		
			is_stable = worker.verify_stability_time_evolution(ode_state_log[:ii+1],
											stability_rel_tolerance,tail_length_stability_check)
		
			if is_stable:
				return ode_state_log[:ii+1], is_stable
		
	
	return ode_state_log, is_stable


# Top level Routine
def gradient_descent(model_configuration, parameters, constraints,
					gradient_method,barrier_slope=1e-2,
					gd_max_iter=100, pert_scale=1e-5,grad_scale=1e-9):

	""" framework for applying a gradient decent approach to a 
		a model, applying a certain method 
		
		Parameters
		----------
		model_configuration : object
			contains all the information and necessary methods
			of the optimized model			
		parameters: numpy.array (1D)
			contains the values which are going to be optimized
		constraints: numpy.array (2D)
			contains the value ranges of the optimized parameters 
		gradient_method : function
			{SGD_basic,SGD_momentum}
			Selects the method used during the gradient descent.
			They differ in their robustness and convergence speed
		integration_scheme: function
			{euler_forward, runge_kutta}
			Selects which method is used in the integration of the time evolution.
			Euler is of first order, Runge-Kutta of second
		barrier_slope : positive-float
			Defines the slope of the barrier used for the soft constrain.
			Lower numbers, steeper slope. Typically between (0-1].
		gd_max_iter : positive integer
			Maximal amount of iterations allowed in the gradient descent
			algorithm.
		pert_scale : positive float
			Maximal value which the system can be perturbed if necessary
			(i.e. if instability is found). Actual perturbation ranges
			from [0-pert_scale) uniformly distributed.
		grad_scale : positive float
			Scales the step size in the gradient descent. Often also
			referred to as learning rate. Necessary to compensate for the
			"roughness" of the objective function field.

		Returns
		-------
		parameter_stack : numpy.array
			2D-array containing the set of optimized free parameter,
			stacked along the first axis.
		prediction_stack : numpy.array
			2D-array containing the output of the time evolution,
			stacked along the first axis.
		cost_stack
			1D-array containing the corresponding values of the cost
			function calculated based on the free_param at the same
			first-axis index.   """

	param_stack = np.zeros((gd_max_iter,len(parameters)))
	param_stack[0] = parameters
	cost_stack = np.zeros((gd_max_iter))
	model_output_shape = model_configuration.configuration['model_output_shape']
	prediction_output_shape = model_configuration.configuration['prediction_shape']
	model_stack = np.zeros( (gd_max_iter,) + model_output_shape )
	prediction_stack = np.zeros( (gd_max_iter,) + prediction_output_shape )
	# ii keeps track of the position in the output array
	# jj keeps track to not exceed max iterations
	ii = 0; jj = 0
	
	while jj < gd_max_iter-1:

		""" makes sure that all points in the parameter set are inside
			of the search space and if not moves them back into it """
		param_stack[ii] = worker.barrier_hard_enforcement(
			param_stack[ii],constraints,pert_scale)
		""" fetch prediction and cost at given point """
		model_log,prediction_stack[ii], cost_stack[ii] = \
			model_configuration.calc_cost(param_stack[ii],barrier_slope)[0:2+1]
		# deals with shorter model output
		model_stack[ii,:len(model_log)] = model_log
		
		if cost_stack[ii] == None:
			# in case the current parameter stack are unstable
			param_stack[ii] = worker.perturb(param_stack[ii],pert_scale)
		else:
			""" calculate the local gradient at at the current point """
			# maybe implement a local_gradient method as well.
			# this would make it more consistent with the other methods
			gradient = worker.local_gradient(model_configuration,
				param_stack[:ii+1], constraints, barrier_slope, pert_scale)[0]
			if gradient is None:
				""" moves the original set uof free parameters_stack in case
					that any(!) of the surrounding points used in the calculation
					of the local gradient is unstable """
				param_stack[ii] = worker.perturb(param_stack[ii],pert_scale)
			else:
				""" applying a decent model to find a new and hopefully
					better set of free parameters_stack """
				param_stack[ii+1] = gradient_method(param_stack[ii],
										gradient,grad_scale)
				ii += 1
		jj += 1

	return param_stack, model_stack, prediction_stack, cost_stack


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
	model_configuration : dict
		contains the configuration of the model including the output log,
		containing all the intermediate outputs
		(parameters, model, prediction, cost)
	"""

	if sample_sets == -1:
		sample_sets = 0
		optim_flag = True
	else: 
		optim_flag = False
	# seeds random generator to create reproducible runs
	np.random.seed(seed)

	# initializes model configuration 
	# (contains information about the optimized model)
	model_configuration = models.model_class(path_model_configuration)
	model_configuration.initialize_log(sample_sets, gd_max_iter)
	
	# runs the optimization with the initial values read from file
	if optim_flag:
		""" This option is run if there is no optimization desired,
			of if no parameters to optimized are provided """
		sample_sets = 0
		model_log, prediction, is_stable = \
			model_configuration.calc_prediction()
		cost = worker.cost_function(prediction,
			model_configuration.configuration['fit_target'])
		print('Is stable? {}'.format(is_stable))
		model_configuration.to_log(np.array([]),model_log,prediction,cost)

	elif sample_sets == 0:	
		""" This option runs the optimization with the initial parameters
			presented in the model configuration """
		
		# fetches the parameters and their constraints from model config
		constraints = model_configuration.to_grad_method()[1]
		if len(constraints) == 0:
			warnings.warn('Monte Carlo optimization method called with '
							+'no parameters to optimise. '
							+'Falling back to running model without '
							+'optimization.')
			return dn_monte_carlo(path_model_configuration,
				gradient_method=gradient_method,
				barrier_slope=barrier_slope,
				sample_sets=-1,gd_max_iter=gd_max_iter,
				pert_scale=pert_scale,grad_scale=grad_scale,seed=seed)
		else:
			parameters = worker.monte_carlo_sample_generator(constraints)
			
			# runs the gradient descent for the generated sample set 
			parameters, model_data, prediction, cost = \
				gradient_descent(model_configuration, parameters, constraints,
				gradient_method, barrier_slope, gd_max_iter, pert_scale, grad_scale)
			
			# updates log with the generated results

			model_configuration.to_log(
					parameters,model_data,prediction, cost)
	
	# runs the optimization with randomly chosen values
	# values are picked from inside the allowed optimization range
	else:
		for ii in np.arange(0,sample_sets):
			print('Monte Carlo Sample #{}'.format(ii))
			
			# updates the state of the optimization run
			model_configuration.log['monte_carlo_idx'] = ii
			
			# fetches the parameters and their constraints from model config
			constraints = model_configuration.to_grad_method()[1]
			if len(constraints) == 0:
				warnings.warn('Monte Carlo optimization method called with '
								+'no parameters to optimise. '
								+'Falling back to running model without '
								+'optimization.')
				return dn_monte_carlo(path_model_configuration,
					gradient_method=gradient_method,
					barrier_slope=barrier_slope,
					sample_sets=-1,gd_max_iter=gd_max_iter,
					pert_scale=pert_scale,grad_scale=grad_scale,seed=seed)
			else:
				parameters = worker.monte_carlo_sample_generator(constraints)
				
				# runs the gradient descent for the generated sample set 
				param_stack, model_stack, prediction_stack, cost_stack = \
					gradient_descent(model_configuration, parameters,constraints,
					gradient_method, barrier_slope, gd_max_iter, pert_scale, grad_scale) 
				
				# updates log with the generated results
				model_configuration.to_log(
					param_stack,model_stack,prediction_stack, cost_stack)
				

	return model_configuration