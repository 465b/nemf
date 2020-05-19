import numpy as np
from gemf import caller
from gemf import worker
from copy import deepcopy

# Time Evolution

## Integration Schemes
def euler_forward(ode_state,ode_coeff,dt_time_evo):
	""" integration scheme for the time evolution 
		based on a euler forward method
		
		Parameters:
		-----------
		ode_state : numpy.array
			1D array containing the state of the observed quantities
			in the ODE
		ode_coeff : numpy.array
			2d-square-matrix containing the coefficients of the ODE
		dt_time_evo
			Size of time step used in the time evolution.
			Has the same unit as the one used in the initial ode_state
		
		Returns:
		--------
		ode_state : numpy.array
			1D array containing the state of the observed quantities
			in the ODE. Now at the next iteration step  """
	
	ode_state = ode_state + np.matmul(ode_coeff,ode_state)*dt_time_evo 
	
	return ode_state

def runge_kutta(ode_state,ode_coeff,dt_time_evo):
	""" integration scheme for the time evolution 
		based on a euler forward method 
		
		Parameters:
		-----------
		ode_state : numpy.array
			1D array containing the state of the observed quantities
			in the ODE
		ode_coeff : numpy.array
			2d-square-matrix containing the coefficients of the ODE
		dt_time_evo
			Size of time step used in the time evolution.
			Has the same unit as the one used in the initial ode_state
		
		Returns:
		--------
		ode_state : numpy.array
			1D array containing the state of the observed quantities
			in the ODE. Now at the next iteration step  """
	
	ode_state_half = ode_state + dt_time_evo/2*np.matmul(ode_coeff,ode_state)
	ode_state = ode_state_half + np.matmul(ode_coeff,ode_state_half)*dt_time_evo
	
	return ode_state


# Compartment Coefficient models
""" all coefficient models should follow the form:
	model(model_config) """

## ODE coefficient models
""" all weights model have the form: model(ode_state,ode_coeff).
	no ode_state dependence (so far necessary) and all required
	constants should be called in the model through a function 
	(i don't know if thats very elegant, but saves an unnecessary) """


def interaction_model_generator(model_config):
	""" uses the system configuration to compute the interaction_matrix """

	interaction_index = model_config.fetch_index_of_interaction()
	# we will rewrite alpha_ij every time after it got optimized.
	alpha = model_config.create_empty_interaction_matrix()
	
	#interactions
	for kk,(ii,jj) in enumerate(interaction_index):
		interaction = list(model_config.interactions)[kk]
		#functions
		for item in model_config.interactions[interaction]:
			parameters = item['parameters'].copy()
			for nn,entry in enumerate(parameters):
				if (type(entry) == str) & (entry in list(model_config.compartment)):
					parameters[nn] = model_config.compartment[entry]['value']

			# updates the matrix
			## nondiagonal elements
			alpha[ii,jj] += int(item['sign']) \
								*globals()[item['fkt']](*parameters)
			## diagonal elements
			alpha[jj,jj] -= int(item['sign']) \
								*globals()[item['fkt']](*parameters)
	return alpha


# Fit models
""" Fit models define how the output of the time evolution (if stable)
	is used  further processing. Then, the optimization routine uses
	this processed time evolution output to fit the model to.
	Hence, the name fit_model. """
	
def direct_fit_model(model_config,integration_scheme, 
					   time_evo_max, dt_time_evo,
					   idx_source, idx_sink,
					   ode_state, ode_coeff=None,
					   ode_coeff_model=interaction_model_generator,
					   stability_rel_tolerance=1e-5,
					   tail_length_stability_check=10,
					   start_stability_check=100):
	""" Returns the last step in the time evolution. Hence, it uses the values
		of the time evolution *directly*, without any further processing 
	
		Parameters:
		-----------
		integration_scheme: function
			{euler_forward, runge_kutta}
			Selects which method is used in the integration
			of the time evolution. Euler is of first order,
			Runge-Kutta of second
		time_evo_max
			Maximal amount of iterations allowed in the time evolution.
			Has the same unit as the one used in the initial ode_state
		dt_time_evo
			Size of time step used in the time evolution.
			Has the same unit as the one used in the initial ode_state
		ode_state : numpy.array
			1D array containing the initial state of the observed quantities
			in the ODE. Often also referred to as initial conditions.
		ode_coeff : numpy.array
			2d-square-matrix containing the coefficients of the ODE
		ode_coeff_model : function
			selects the function used for the calculation of the ODE
			coefficients. I.e. if dependencies of the current 
			state are present. If no dependency is present
			use 'standard_weights_model'
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

		Returns:
		--------
		F_i : numpy.array
			2D-array containing the output of the time evolution
			after the fit_model has been applied, stacked along the first axis.
		is_stable : bool
			true if stability conditions are met. 
			See check_convergence() for more details. """
	
	F_ij, is_stable = caller.time_evolution(model_config,
										integration_scheme, time_evo_max,
										dt_time_evo,ode_state,
										ode_coeff_model,ode_coeff,
										stability_rel_tolerance,
										tail_length_stability_check,
										start_stability_check)
	F_i = F_ij[-1]

	return F_ij,F_i,is_stable 


def net_flux_fit_model(model_config,integration_scheme, 
					   time_evo_max, dt_time_evo,
					   idx_source, idx_sink,
					   ode_state, ode_coeff=None,
					   ode_coeff_model=interaction_model_generator,
					   stability_rel_tolerance=1e-5,
					   tail_length_stability_check=10,
					   start_stability_check=100):

	""" Takes the last step in the time evolution and calculates the sum
		of its entries. Counts the last entry negatively.
		This is done to represent a 'dump' through which the positive
		net-flux of the system is compensated.
		
		Parameters:
		-----------
		integration_scheme: function
			{euler_forward, runge_kutta}
			Selects which method is used in the integration of 
			the time evolution. Euler is of first order,
			Runge-Kutta of second
		time_evo_max : float
			Maximal amount of iterations allowed in the time evolution.
			Has the same unit as the one used in the initial ode_state
		dt_time_evo : float
			Size of time step used in the time evolution.
			Has the same unit as the one used in the initial ode_state
		b
		ode_state : numpy.array
			1D array containing the initial state of the observed quantities
			in the ODE. Often also referred to as initial conditions.
		ode_coeff : numpy.array
			2d-square-matrix containing the coefficients of the ODE
		ode_coeff_model : function
			selects the function used for the calculation of the ODE
			coefficients. I.e. if dependencies of the current state 
			are present. If no dependency is present 
			use 'standard_weights_model'
		stability_rel_tolerance : positive float
			Defines the maximal allowed relative fluctuation range in the 
			tail of the time evolution. If below, system is called stable.
		tail_length_stability_check : positive integer
			Defines the length of the tail used for the stability calculation.
			Tail means the amount of elements counted from the back of the
			array.
		start_stability_check : positive integer
			Defines the element from which on we repeatably check if the
			time evolution is stable. If stable, iteration stops and last
			value is returned

		Returns:
		--------
		F_i : numpy.array
			2D-array containing the output of the time evolution
			after the fit_model has been applied, stacked along the first axis.
		is_stable : bool
			true if stability conditions are met. 
			See check_convergence() for more details. """

	F_ij, is_stable = caller.time_evolution(model_config,
										  integration_scheme, time_evo_max,
										  dt_time_evo,ode_state,
										  ode_coeff_model,ode_coeff,
										  stability_rel_tolerance,
										  tail_length_stability_check,
										  start_stability_check)
	F_i = F_ij[-1]
	prediction = np.array(np.sum(F_i[idx_source]) - np.sum(F_i[idx_sink]))

	return F_ij, prediction, is_stable


# Gradient Decent

## Gradient Decent Methods

def SGD_basic(free_param,gradient,grad_scale):
	""" Stochastic Gradient Descent (SGD) implementation to minimize
		the cost function 
		
		Parameters:
		-----------
		free_param : numpy.array
			2D-array containing the set of optimized free parameter,
			stacked along the first axis.
		gradient : numpy.array
			Gradient at the center point calculated by the randomly chosen 
			local environment. The gradient always points in the direction
			of steepest ascent.
		grad_scale : positive float
			Scales the step size in the gradient descent. Often also
			referred to as learning rate. Necessary to compensate for the
			"roughness" of the objective function field.

		Returns:
		-------
		free_param_next : numpy.array
			1D-array containing the set of optimized free parameter
			for the next iteration step. """
	
	
	free_param_next = free_param[-1] - grad_scale*gradient
	
	return free_param_next


def SGD_momentum(free_param,gradient,grad_scale):
	""" Stochastic Gradient Descent (SGD) implementation plus and additional
		momentum term to minimize the cost function.
	
		Parameters:
		-----------
		free_param : numpy.array
			2D-array containing the set of optimized free parameter,
			stacked along the first axis.
		gradient : numpy.array
			Gradient at the center point calculated by the randomly chosen 
			local environment. The gradient always points in the direction
			of steepest ascent.
		grad_scale : positive float
			Scales the step size in the gradient descent. Often also
			referred to as learning rate. Necessary to compensate for the
			"roughness" of the objective function field.

		Returns:
		-------
		free_param_next : numpy.array
			1D-array containing the set of optimized free parameter
			for the next iteration step. """
	
	if len(free_param) >= 3:
		previous_delta = free_param[-2]-free_param[-3]
	else:
		previous_delta = 0

	alpha = 1e-1 # fix function to accept alpha as an input
	free_param_next = free_param[-1] - grad_scale*gradient + alpha*previous_delta
	
	return free_param_next

	
# interaction functions

## Grazing Models
def J(N,k_N,mu_m):
	""" Nutrition saturation model"""

	""" we are currently assuming constant, perfect 
		and homogeneous illumination. Hence, the 
		f_I factor is currently set to 1 """

	f_N = N/(k_N+N)
	f_I = 1 #I/(k_I+I)
	cost_val = mu_m*f_N*f_I
	return cost_val


def holling_type_0(coefficient):
	""" constant respone (implicit linear),
		compartment size independent """
	return coefficient


def holling_type_I(value,coefficient):
	""" (co-)linear response """
	return value*coefficient

def holling_type_II(prey_population,food_processing_time,hunting_rate):
	""" (co-)linear + saturation response """
	consumption_rate = ((hunting_rate * prey_population)/
			(1+hunting_rate * food_processing_time * prey_population))
	return consumption_rate


def holling_type_III(P,epsilon,g):
	""" quadratic (+ implicit linear) + saturation response """
	G_val = (g*epsilon*P**2)/(g+(epsilon*P**2))
	return G_val


def sloppy_feeding(holling_type,coeff,*args):
	""" calls holling_type functions with an extra "efficiency" coefficient.
		the inverse of the efficiency is then supposed to flow into
		a different compartment """
		
	if holling_type == '0':
		return coeff*holling_type_0(*args)
	elif holling_type == 'I':
		return coeff*holling_type_I(*args)
	elif holling_type == 'II':
		return coeff*holling_type_II(*args)
	elif holling_type == 'III':
		return coeff*holling_type_III(*args)
	else:
		raise ValueError("The defined holling_type is not available. "
			+"Use one of the following: ['0','I','II','III']")
	

def nutrition_limited_growth(N,growth_rate,half_saturation):
	""" reparameterization of holling_II """
	return growth_rate*(N/(half_saturation+N))

## referencing interaction functions
""" It is helpfull from a mechanistic point of view to not only
	represent the relation between parameters but also its 'context'.
	Hence, a linear mortality might be represented with a lin_mort()
	function which then maps to a standard linear function to better
	represent its usage and increase readability """

linear_mortality = holling_type_0
remineralisation = holling_type_0
exudation = holling_type_0
excretion = holling_type_0
stress_dependant_exudation = holling_type_I

# Model_Classes 

class model_class:
	def __init__(self,path):
		self.init_sys_config = worker.initialize_ode_system(path)
		self.sanity_check_input()
		self.compartment = deepcopy(
			self.init_sys_config['compartment'])
		self.interactions = deepcopy(
			self.init_sys_config['interactions'])
		self.configuration = deepcopy(
			self.init_sys_config['configuration'])
		self.fetch_index_of_source_and_sink()
		
		self.configuration['model_output_shape'] = \
			(int(self.configuration['time_evo_max']/
				self.configuration['dt_time_evo']),
				len(list(self.init_sys_config['compartment'])))
		self.configuration['prediction_shape'] = \
			(len(self.configuration['fit_target']),)


	def sanity_check_input(self):
		""" 
			checks for obvious errors in the configuration file.
			passing this test doesn't guarante a correct configuration. 
		"""
		unit = self.init_sys_config

		# checks if compartment is well defined
		name = "Model configuration "
		assert_if_exists_non_empty('compartment',unit,reference='compartment')
		assert (len(list(unit['compartment'])) > 1), \
			name + "only contains a single compartment"
		## check if all compartment are well defined
		for item in list(unit['compartment']):
			assert_if_non_empty(item,unit['compartment'],item,reference='state')
			assert_if_exists('optimise',unit['compartment'][item],item)
			assert_if_exists_non_empty('value',unit['compartment'][item],
									   item,'value')
			assert_if_exists('optimise',unit['compartment'][item],item)
			
		# checks if interactions is well defined
		assert_if_exists_non_empty('interactions',unit,'interactions')
		## check if all interactions are well defined
		for item in list(unit['interactions']):
			for edge in unit['interactions'][item]:
				assert edge != None, \
					name + "interaction {} is empty".format(item)
				assert_if_exists_non_empty('fkt',edge,item,)
				assert_if_exists_non_empty('sign',edge,item)
				assert_if_exists('parameters',edge,item)
				assert_if_exists('optimise',edge,item)

		# checks if configuration is well defined
		#assert (unit[]) 
		assert_if_exists_non_empty('configuration', unit)
		required_elements = ['integration_scheme','time_evo_max',
			'dt_time_evo','ode_coeff_model', 'stability_rel_tolerance',
			'tail_length_stability_check','start_stability_check',
			'fit_model','fit_target']
		for element in required_elements:
			assert_if_exists_non_empty(element,unit['configuration'])


	def initialize_log(self,n_monte_samples,max_gd_iter):
		parameters = self.to_grad_method()[0]
		param_log = np.zeros((n_monte_samples,
									max_gd_iter, len(parameters)))
		prediction_log = np.zeros( (n_monte_samples, max_gd_iter,
								len(self.configuration['fit_target'])))
		cost_log = np.zeros( (n_monte_samples, max_gd_iter) )
		model_log = np.zeros( (n_monte_samples, max_gd_iter,
							int(self.configuration['time_evo_max']/
							self.configuration['dt_time_evo']),
							len(list(self.compartment))))
	
		log_dict = {'parameters': param_log,
					'predictions': prediction_log,
					'cost': cost_log,
					'model': model_log,
					'monte_carlo_idx': 0, 'gradient_idx': 0}
		
		self.log = log_dict


	def fetch_index_of_interaction(self):
		""" gets the indices in the interaction matrix """
		## separate row & column
		interactions = list(self.interactions)
		compartments = list(self.compartment)
		
		interaction_index = interactions.copy()
		for index,item in enumerate(interactions):
			interaction_index[index] = item.split(':')
		## parse them with the index
		for index, item in enumerate(interaction_index):
			interaction_index[index][0] = compartments.index(item[0])
			interaction_index[index][1] = compartments.index(item[1])

		return interaction_index


	def fetch_index_of_source_and_sink(self):
		if self.configuration['sources'] == None:
			sources = None
			idx_sources = []
		else:
			sources = list(self.configuration['sources'])
			idx_sources = sources.copy()
			for ii, item in enumerate(idx_sources):
				idx_sources[ii] = list(self.compartment).index(item)
		
		if self.configuration['sinks'] == None:
			sinks = None
			idx_sinks = []
		else:
			sinks = list(self.configuration['sinks'])
			idx_sinks = sinks.copy()
			for ii, item in enumerate(idx_sinks):
				idx_sinks[ii] = list(self.compartment).index(item)
			
		self.configuration['idx_sources'] = idx_sources
		self.configuration['idx_sinks'] = idx_sinks


	def to_log(self,parameters,model_data,predictions,cost):
		#current monte sample
		ii = self.log['monte_carlo_idx']

		if np.shape(predictions) == ():
			# reshape prediction in case they are zero-dim
			predictions = np.reshape(predictions,(1,1))
		
		if np.shape(self.log['parameters'])[0] == 0:
			# in case there is only one parameter set
			self.log['parameters'] = parameters
			self.log['predictions'] = predictions
			self.log['model'] = model_data
			self.log['cost'] = cost
		else:
			self.log['parameters'][ii] = parameters
			self.log['predictions'][ii] = predictions
			self.log['model'][ii] = model_data
			self.log['cost'][ii] = cost
		
		self.log['gradient_idx'] += 1
		

	def from_ode(self,ode_states):
		""" updates self with the results provided by the ode solver """
		for ii, item in enumerate(self.compartment):
			self.compartment[item]['value'] = ode_states[ii]


	def to_ode(self):
		""" fetches the parameters necessary for the ode solver 
 	       Returns: ode_state,ode_coeff_model,ode_coeff """
		ode_state = np.array([self.compartment[ii]['value'] for ii in self.compartment])
		ode_coeff_model = interaction_model_generator
		ode_coeff = ode_coeff_model(self)
		
		return ode_state,ode_coeff_model, ode_coeff


	def to_grad_method(self):
		""" fetches the parameters necessary for the gradient descent method
 	       Returns: free_parameters, constraints """
		
		free_parameters = []
		constraints = []
		labels = []
		for ii in self.compartment:
			if self.compartment[ii]['optimise'] is not None:
				labels.append('{}'.format(ii))
				value = self.compartment[ii]['value']
				lower_bound = self.compartment[ii]['optimise']['lower']
				upper_bound = self.compartment[ii]['optimise']['upper']
				free_parameters.append(value)
				constraints.append([lower_bound,upper_bound])

		for ii in self.interactions:
			#function
			for item in self.interactions[ii]:
				#parameters
				if item['optimise'] is not None:
					for jj,elements in enumerate(item['optimise']):
						labels.append('{},fkt: {} #{}'.format(
							ii,item['fkt'],elements['parameter_no']))
						value = item['parameters'][jj]
						lower_bound = elements['lower']
						upper_bound = elements['upper']

						free_parameters.append(value)
						constraints.append([lower_bound,upper_bound])
			
		free_parameters = np.array(free_parameters)
		constraints = np.array(constraints)

		return free_parameters, constraints, labels


	def refresh_to_initial(self):
		self.compartment = deepcopy(self.init_sys_config['compartment'])


	def create_empty_interaction_matrix(self):
		""" initializes an returns empty interaction matrix """
		size = len(self.compartment)
		alpha = np.zeros((size,size))
		return alpha


	def update_system_with_parameters(self, parameters):
		values = list(parameters)
		
		for ii in self.compartment:
			if self.compartment[ii]['optimise'] is not None:
				self.compartment[ii]['value'] = values.pop(0)
	
		for ii in self.interactions:
			#function
			for item in self.interactions[ii]:
				#parameters
				if item['optimise'] is not None:
					for element in item['optimise']:
							item['parameters'][element['parameter_no']-1] = \
								values.pop(0)

	
	def calc_prediction(self):
		ode_states,ode_coeff_model, ode_coeff = self.to_ode()
		fit_model = globals()[self.configuration['fit_model']]
		
		model_log, prediction,is_stable = fit_model(self,
			globals()[self.configuration['integration_scheme']], 
			self.configuration['time_evo_max'],
			self.configuration['dt_time_evo'],
			self.configuration['idx_sources'],
			self.configuration['idx_sinks'],
			ode_states,
			ode_coeff,	
			ode_coeff_model,
			float(self.configuration['stability_rel_tolerance']),
			self.configuration['tail_length_stability_check'],
			self.configuration['start_stability_check'])
		
		return model_log, prediction, is_stable
	
	
	def calc_cost(self, parameters, barrier_slope):

		self.refresh_to_initial()
		self.update_system_with_parameters(parameters)	
		constrains = self.to_grad_method()[1]		
		model_log, prediction, is_stable = self.calc_prediction()
		cost = worker.cost_function(
			prediction,self.configuration['fit_target'])
		cost += worker.barrier_function(
			parameters,constrains,barrier_slope)

		return model_log, prediction, cost, is_stable


def assert_if_exists(unit,container,item='',reference='',
								name="Model configuration "):
	assert (unit in container), \
		name + reference + " {} lacks definition of {}".format(item,unit)

def assert_if_non_empty(unit,container,item='',reference='',
								name="Model configuration "):
	assert (container[unit] != None), \
		name + reference + " {} {} is empty".format(item,unit)

def assert_if_exists_non_empty(unit,container,item='',reference='',
								name="Model configuration "):
	assert_if_exists(unit,container,item,reference=reference,name=name)
	assert_if_non_empty(unit,container,item,reference=reference,name=name)