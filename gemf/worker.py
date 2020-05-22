import numpy as np
from scipy.integrate import solve_ivp 
import logging
import yaml
from copy import deepcopy
from termcolor import colored, cprint

from gemf import models

logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)


# Data reading

def read_coeff_yaml(path):
	"Reads a yaml file and returns its as a dictionary"
	
	fo = open(path,'r')
	stream = fo.read()
	data_dict = yaml.safe_load(stream)
	
	return data_dict


def import_fit_data(path):
	# add switches for file type parser
	return np.genfromtxt(path)


# Parsers

def initialize_ode_system(path_config):
	""" Initializes the dictionary containing the 'state' and 'interactions'

		Parameters:
		-----------
		path_config : string
			Path to the yaml file containing the ode model configuration.
			The configuration contains the compartments, initial values
			and optimization constrains - as well as - interactions paths, 
			interaction directions (sign),interaction functions,
			initial parameter values and optimization constrains.

		Returns:
		--------
		ode_system_configuration : dict
			Contains the state and interaction configuration
			in a similar structure as provided by the yaml files."""

	system_config = read_coeff_yaml(path_config)

	return system_config


# Gradient Decent

## Cost function related Methods
def cost_function(prediction,y):
	""" normalized squared distance between model prediction and
		desired model behavior 
		
		Parameters
		----------
		prediction : numpy.array
			1D-array contains the output of the fit model which
			ran the time integration
		y : numpy.array
			1D-array,same length as prediction containing the desired
			model output.
			
		Returns
		-------
		cost : float
			contains the value of the cost function for the given prediction
			and desired state"""


	cost = (1/len(y))*np.sum( (prediction - y)**2 )
	return cost


def barrier_function(free_param,constrains=np.array([None]),barrier_slope=1):
	""" Constructs the additional cost that is added closer to constrains.
		This is helpfull to keep the optimization process from trying to 
		exceed the search space. However, also distorts the optimal output
		slightly.

		Parameters
		----------
		free_param : numpy.array
			1D-array of length n containing the current set of optimized free
			paratmers
		constrains : numpy.array
			2D-array of shape (n,2) containing the constraints limits
		barrier_slope : positive-float
			Defines the slope of the barrier used for the soft constrain.

		Returns
		-------
		cost_barrier : float
			additional cost created from beeing close to a constraints
			barrier. """

	# checks if contains are defined. else sets them to +/- inf
	if (constrains == None).all():
		constrains = np.zeros((len(free_param),2))
		constrains[:,0] = -np.inf
		constrains[:,1] = +np.inf

	# checks if constrains has the correct length
	if len(constrains) != len(free_param):
		raise ValueError('List of constrains must have'+
						 'the same length as free_param ({},{})'.format(
							 len(constrains),(len(free_param))))

	# checks if barrier_slope is positive
	if barrier_slope <= 0:
		raise ValueError('barrier_slope must be a positive value.')

	# initializes arrays containing the additional cost
	cost_barrier_left = np.zeros(len(free_param))
	cost_barrier_right = np.zeros(len(free_param))
	cost_barrier = np.zeros(len(free_param))
	
	# calculates the additional cost
	""" note that we always apply the cost of both sides.
		We assume that if one of them is non-negligible,
		the other one is """
	for ii,[left,right] in enumerate(constrains):
		#left constrain
		if (left == -np.inf):
			cost_barrier_left[ii] = 0
		else:
			cost_barrier_left[ii] = -np.log(free_param[ii]-left)

		# right constrain
		if (right == np.inf):
			cost_barrier_right[ii] = 0
		else:
			cost_barrier_right[ii] = -np.log(-free_param[ii]+right)

	cost_barrier = cost_barrier_left + cost_barrier_right
	cost_barrier = np.sum(cost_barrier)*barrier_slope
	
	return cost_barrier


def barrier_hard_enforcement(free_param,constrains=None,
							 pert_scale=1e-2,seed=137):
	""" if outside the search space we enforce a 'in-constrain'
		search space by ignoring the recommended step and
		moving it back into the search space

	Parameters
	----------
	free_param : numpy.array
		1D-array containing the set of optimized free parameter.
	jj : positive integer
		Index of the iteration step in which the function is called.
		This is logged for debugging purposes.
	constrains : numpy.array
		2D-array containing the upper and lower limit of every free input
		parameter in the shape (len(free_param),2)
	pert_scale : positive float
		Maximal value which the system can be perturbed if necessary
		(i.e. if instability is found). Actual perturbation ranges
		from [0-pert_scale) uniformly distributed.
	seed : positive integer
		Initializes the random number generator. Used to recreate the
		same set of pseudo-random numbers. Helpfull when debugging.
	
	Returns
	-------
	free_param : numpy.array
		1D-array containing the set of optimized free parameter.
		Now, guaranteed to be inside of the search space.
	"""

	# initializes constrains from plus to minus inf if not provied
	if (constrains == None).all():
		constrains = np.zeros((len(free_param),2))
		constrains[:,0] = -np.inf
		constrains[:,1] = +np.inf

	# tests if constrains is provided for all free_parameters
	if len(constrains) != len(free_param):
		raise ValueError('List of constrains must have'+
						 'the same length as free_param')

	# applies the actual enforcement
	for ii,[left,right] in enumerate(constrains):
		""" we add a small perturbation to avoid 
			that the we remain on the boarder """
		if (free_param[ii] <= left):
			buffer = left + np.random.rand()*pert_scale
			warn_string = ( 'Left  barrier enforcement'+
							'Value shifted from {:+8.2E} to {:+8.2E}'.format(
								free_param[ii],buffer))
			logging.debug(warn_string)
			free_param[ii] = buffer

		if (free_param[ii] >= right):
			buffer = right - np.random.rand()*pert_scale
			warn_string = ( 'Right barrier enforcement'+
							'Value shifted from {:+8.2E} to {:+8.2E}'.format(
								free_param[ii],buffer))
			logging.debug(warn_string)
			free_param[ii] = buffer
			
	return free_param


# Time Evolution

## Stability

def convergence_event_constructor(model,threshold=1e-6,):
	epsilon = threshold
	differential_equation = model.de_constructor()

	def convergence(t,y,*args):
		derivative = differential_equation(t,y,*args)

		if (sum(abs(derivative)) < epsilon) and (t != 0):
			print('Convergence reached')
			return 0
		else:
			return 1

	return convergence


def check_convergence(prediction, stability_rel_tolerance=1e-6,
									tail_length=10):
	""" checks if the current solution is stable by 
		comparing the relative fluctuations in the 
		last tail_length model outputs to a stability_rel_tolerance value
		returns true if all values are below that threshold.
		
	Parameters
	----------
	prediction : numpy.array
		1D-array containing the output of the time evolution for every time
		integration step.
	stability_rel_tolerance : positive float
		Defines the maximal allowed relative fluctuation range in the tail
		of the time evolution. If below, system is called stable.
	tail_length_stability_check : positive integer
		Defines the length of the tail used for the stability calculation.
		Tail means the amount of elements counted from the back of the
		array.
	
	Returns
	-------
	is_stable : bool
		true if stability conditions are met
	"""

	# gets relevant data slice    
	prediction_tail = prediction[-tail_length-1:-1]

	# computes local average and spread
	average = np.average(prediction_tail,axis=0)
	spread = np.max(prediction_tail,axis=0) -np.min(prediction_tail,axis=0)
	
	# compares spread to the provided conditions 
	rel_spread = spread/average
	rel_condition = (rel_spread <= float(stability_rel_tolerance)).all() 
	# also evaluates positive if the spread is smaller in absolute values
	# then the provided conditions. This avoid instability due to float
	# rounding errors
	abs_condition = (spread <= float(stability_rel_tolerance)).all()
	
	is_stable = rel_condition or abs_condition
	
	return is_stable


# Helper Functions

## General Helper Function

def division_scalar_vector_w_zeros(a,b):
	""" Divides an an array (a/b) while catching divisions by zero
		and sets them to zero """
	free_param = np.divide(a, b, out=np.zeros_like(b), where=b!=0)
	
	return free_param


## PDE-weights related Helper Functions
def normalize_columns(A):
	""" Enforces that all columns of A are normalized to 1.
		Otherwise an carbon mass transport amplification would occur.
		Does so by perserving the ratios between the values in a single column. 
	
	Parameters:
	-----------
	A : numpy.array
		can be any square-matrix containing scalars.
	
	Returns:
	--------
	A : numpy.array
		normalized version of the input """
	
	overshoot = np.sum(A,axis=0)    
	for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
		if truth_val : 
			diag_val = A[ii,ii]
			if diag_val == overshoot[ii]:
				# only the diagonal entry is filled
				# hence, its the dump.
				pass
			elif (diag_val == 0): 
				overshoot[ii] -= 1
				if (abs(overshoot[ii]) > 1e-16):
					A[:,ii] -= A[:,ii]/(overshoot[ii]+1)*overshoot[ii]
					A[ii,ii] = diag_val
			else:
				A[:,ii] -= A[:,ii]/(overshoot[ii]-diag_val)*overshoot[ii]
				A[ii,ii] = diag_val

	return A


## Gradient Decent related Helper Functions

def update_param(initial_param, fit_param, fit_idx):

	# states
	fit_states = fit_param[:len(fit_idx[0])]
	states_t0 = deepcopy(initial_param[0])
	for [idx,item] in zip(fit_idx[0],fit_states):
		states_t0[idx] = item
	
	# args
	fit_args = fit_param[len(fit_idx[0]):]
	args = deepcopy(initial_param[1])
	
	for [idx,item] in zip(fit_idx[1],fit_args):
		args[idx[0]][idx[1]][idx[2]] = item
	
	return [states_t0,args]


def construct_objective(model,time_series_events=None,debug=False):
	
	differential_equation = model.de_constructor()
	fit_indices = model.fetch_to_optimize_args()[0][0]
	initial_param = model.fetch_param()
	ref_data = model.reference_data
	t_eval = ref_data[:,0]
	

	# steady-state solution
	if t_eval[0] == np.inf:
		if debug: cprint('steady_state objective','magenta')		
		t_min = 0
		t_max = model.configuration['time_evo_max']
		t_span = [t_min,t_max]
		if debug: 
			cprint(f't_span: ','magenta')
			print(t_span)
			cprint(f't_eval: ','magenta')
			print(t_eval)
		y = ref_data[:,1:]

		def objective(fit_param):
			
			updated_param = update_param(initial_param,fit_param,fit_indices)
			y0 = updated_param[0]
			args = [updated_param[1]]
			if debug:
				cprint(f'initial_param: ','magenta')	
				print(initial_param)
				cprint(f'updated_param: ','magenta')
				print(updated_param)
			ode_sol = solve_ivp(differential_equation,t_span,y0,
								args=args,dense_output=True,
								events=time_series_events)
		   

			x = ode_sol.sol(t_span[1]).T
			x = np.reshape(x,(1,len(x)))
			res = np.linalg.norm(x-y)**2
			if debug:
				cprint(f'solution(time integration): ','magenta')
				print(x)
				cprint(f'solution(reference data): ','magenta')
				print(y)
				cprint(f'objective function: ','magenta')
				print(res)
			return res

	# non-steady-state solution
	else:
		if debug: cprint('non_steady_state objective','magenta')		

		t_min = t_eval[0]
		t_max = t_eval[-1]
		t_span = [t_min,t_max]
		if debug: 
			cprint(f't_span: ','magenta')
			print(t_span)
			cprint(f't_eval: ','magenta')
			print(t_eval)
		y = ref_data[:,1:]
		
		def objective(fit_param):
			
			updated_param = update_param(initial_param,fit_param,fit_indices)
			y0 = updated_param[0]
			args = [updated_param[1]]
			if debug:
				cprint(f'fit_param: ','magenta')
				print(fit_param)
				cprint(f'fit_indices: ','magenta')
				print(fit_indices)
				cprint(f'initial_param: ','magenta')
				print(initial_param)
				cprint(f'updated_param: ','magenta')
				print(updated_param)
	
			ode_sol = solve_ivp(differential_equation,t_span,y0,
								args=args,dense_output=True, t_eval=t_eval)
		   
			x = ode_sol.sol(t_eval).T
			res = np.linalg.norm(x-y)**2
			if debug:
				cprint(f'solution(time integration): ','magenta')
				print(x)
				cprint(f'solution(reference data): ','magenta')
				print(y)
				cprint(f'objective function: ','magenta')
				print(res)
				
			return res
		
	return objective


def perturb(x,pert_scale=1e-4):
	"Adds a small perturbation to input (1D-)array"
	
	delta_x = (np.random.rand(len(x)) - 0.5)*pert_scale
	
	return x+delta_x


def read_coeff_constrains(path):
	""" reads the constrains of the ODE coefficients from file """
	constrains = np.genfromtxt(path)
	lower = constrains[:,::2]
	upper = constrains[:,1::2]

	free_coeff = np.where( ((lower != 0) + (upper != 0)) *
			  ((lower != -1) + (upper != -1)) *
			  ((lower != 1) + (upper != 1)) )

	lower = lower[free_coeff]
	upper = upper[free_coeff]

	lower = np.reshape(lower, (len(lower),))
	upper = np.reshape(upper, (len(upper),))

	constrains = np.stack((lower,upper),axis=1)
	
	return constrains,free_coeff


def monte_carlo_sample_generator(constrains):
	""" Constructs a set of homogenously distributed random values 
		in the value range provided by 'constrains'.
		Returns an array of the length of 'constrains' 
		Caution: Samples the FULL float search space if an inf value is provided! 
		
	Parameter:
	----------
	constrains : numpy.array
		2D-array containing the upper and lower limit of every free input
		parameter in the shape (len(free_param),2).
		
	Returns:
	--------
	sample_set : numpy.array 
		1D-array containing a random vector in the range of constrains """

	""" returns min/max possible value if a +/- infinite value is pressent """
	constrains[constrains==np.inf] = np.finfo(float).max
	constrains[constrains==-np.inf] = np.finfo(float).min

	constrains_width = constrains[:,1] - constrains[:,0]
	sample_set = constrains_width*np.random.rand(len(constrains))+constrains[:,0]

	return sample_set
