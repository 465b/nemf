import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from gemf import worker
from gemf import models
from gemf import decorators

#import logging
import warnings
#logging.basicConfig(filename='carbonflux_inverse_model.log',
#					level=logging.DEBUG)


def forward_model(model,method='RK45',verbose=False,t_eval=None):

	""" Runs the time integration for a provided model configuration.
			
		Parameters
		----------
		model : model_class object
			class object containing the model configuration
			and its related methods. See load_configuration()
		verbose : bool
			Flag for extra verbosity during runtime
		t_eval : 1d-array
			contains time stamps in posix time for which a solution shall be 
			found and returned.

		Returns
		-------
		model : model_class object
			class object containing the model configuration, model run results,
			and its related methods
	"""

	[initial_states,args] = model.fetch_param()
	differential_equation = model.de_constructor()
	model.initialize_log(maxiter=1)	

	if t_eval is None:
		t_start = 0
		t_stop = model.configuration['time_evo_max']
		dt = model.configuration['dt_time_evo']
		t = np.arange(t_start,t_stop,dt)
	else:
		t_start = min(t_eval)
		t_stop = max(t_eval)
		t = t_eval
	
	sol = solve_ivp(differential_equation,[t_start,t_stop],initial_states,
					method=method,args=[args], dense_output=True)
	y_t = sol.sol(t).T

	if verbose:
		print(f'ode solution: {sol}')
		print(f't_events: {sol.t_events}')

	t = np.reshape(t,(len(t),1))
	time_series = np.concatenate( (t,y_t),axis=1)
	model.log['time_series'] = time_series

	return model


def inverse_model(model,method='SLSQP',
					sample_sets = 3,
					maxiter=1000,
					seed=137,
					verbose=False,
					debug=False):

	""" Optimizes a set of randomly generated free parameters and returns
		their optimized values and the corresponding fit-model and cost-
		function output 
	
		Parameters
		----------
		model_configuration : model_class object
			class object containing the model configuration
			and its related methods. See load_configuration()
		gradient_method : function
			{SGD_basic,SGD_momentum}
			Selects the method used during the gradient descent.
			They differ in their robustness and convergence speed
		sample_sets : positive integer
			Amount of randomly generated sample sets used as initial free
			parameters
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
		seed : positive integer
			Initializes the random number generator. Used to recreate the
			same set of pseudo-random numbers. Helpfull when debugging.
		convergence_tail_length : int
			number of values counted from the end up that are used to check
			for convergence of the gradient descent iteration.
		convergence_tolerance : float
			maximal allowed relative fluctuation range in the tail of the
			cost function to test positive for convergence
		verbose : bool
			Flag for extra verbosity during runtime

		Returns
		-------
		model_configuration : model_class object
			class object containing the model configuration, 
			model run results (parameters, model, prediction, cost),
			and its related methods
	"""

	# seeds random generator to create reproducible runs
	np.random.seed(seed)

	[fit_param, bnd_param] = model.fetch_to_optimize_args()[0][1:3]
	objective_function = worker.construct_objective(model,debug=debug)
	logger = model.construct_callback(method=method,debug=debug)
	model.initialize_log(maxiter=maxiter)


	if len(fit_param) == 0:
		warnings.warn('Monte Carlo optimization method called with '
						+'no parameters to optimise. '
						+'Falling back to running model without '
						+'optimization.')
		return forward_model(model)

	else:
		cons = model.fetch_constraints()
		if cons ==  None:
			out = minimize(objective_function,fit_param,method=method,
							bounds=bnd_param,callback=logger,
							options={'disp': verbose, 'maxiter': maxiter})
		else:
			out = minimize(objective_function,fit_param,method=method,
							bounds=bnd_param,constraints=cons,callback=logger,tol=1e-6,
							options={'disp': verbose,'maxiter': maxiter})
		
		model.update_system_with_parameters(out.x)
		if verbose:
			print(out)
		
	
	return model