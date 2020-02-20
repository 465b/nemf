import numpy as np
from . import caller
from . import worker

# Time Evolution

## Integration Schemes
def euler_forward(ODE_state,ODE_coeff,dt_time_evo):
    """ integration scheme for the time evolution 
        based on a euler forward method
        
        Parameters:
        -----------
        ODE_state : numpy.array
            1D array containing the state of the observed quantities
            in the ODE
        ODE_coeff : numpy.array
            2d-square-matrix containing the coefficients of the ODE
        dt_time_evo
            Size of time step used in the time evolution.
            Has the same unit as the one used in the initial ODE_state
        
        Returns:
        --------
        ODE_state : numpy.array
            1D array containing the state of the observed quantities
            in the ODE. Now at the next iteration step  """
    
    ODE_state = ODE_state + np.matmul(ODE_coeff,ODE_state)*dt_time_evo 
    
    return ODE_state

def runge_kutta(ODE_state,ODE_coeff,dt_time_evo):
    """ integration scheme for the time evolution 
        based on a euler forward method 
        
        Parameters:
        -----------
        ODE_state : numpy.array
            1D array containing the state of the observed quantities
            in the ODE
        ODE_coeff : numpy.array
            2d-square-matrix containing the coefficients of the ODE
        dt_time_evo
            Size of time step used in the time evolution.
            Has the same unit as the one used in the initial ODE_state
        
        Returns:
        --------
        ODE_state : numpy.array
            1D array containing the state of the observed quantities
            in the ODE. Now at the next iteration step  """
    
    ODE_state_half = ODE_state + dt_time_evo/2*np.matmul(ODE_coeff,ODE_state)
    ODE_state = ODE_state_half + np.matmul(ODE_coeff,ODE_state_half)*dt_time_evo
    
    return ODE_state


# Compartment Coefficient models
""" all coefficient models should follow the form:
    model(system_configuration) """

## ODE coefficient models
""" all weights model have the form: model(ODE_state,ODE_coeff).
    no ODE_state dependence (so far necessary) and all required
    constants should be called in the model through a function 
    (i don't know if thats very elegant, but saves an unnecessary)"""


def interaction_model_generator(system_configuration):
    """ uses the system configuration to compute the interaction_matrix """

    interaction_index = system_configuration.fetch_index_of_interaction()
    # we will rewrite alpha_ij every time after it got optimized.
    alpha = worker.create_empty_interaction_matrix(system_configuration)
    
    #interactions
    for kk,(ii,jj) in enumerate(interaction_index):
        interaction = list(system_configuration.interactions)[kk]
        #functions
        for item in system_configuration.interactions[interaction]:
            # adds everything up
            alpha[ii,jj] += int(item['sign'])*globals()[item['fkt']](*item['parameters'])

    return alpha


# Fit models
""" Fit models define how the output of the time evolution (if stable) is used
    for further processing. Then, the optimization routine uses this processed
    time evolution output to fit the model to. Hence, the name fit_model. """
    
def direct_fit_model(model_configuration,
                     integration_scheme, time_evo_max, dt_time_evo,
                     ODE_state, ODE_coeff=None, 
                     ODE_coeff_model=interaction_model_generator,
                     stability_rel_tolerance=1e-5,tail_length_stability_check=10,
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
            Has the same unit as the one used in the initial ODE_state
        dt_time_evo
            Size of time step used in the time evolution.
            Has the same unit as the one used in the initial ODE_state
        ODE_state : numpy.array
            1D array containing the initial state of the observed quantities
            in the ODE. Often also referred to as initial conditions.
        ODE_coeff : numpy.array
            2d-square-matrix containing the coefficients of the ODE
        ODE_coeff_model : function
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

        Returns:
        --------
        F_i : numpy.array
            2D-array containing the output of the time evolution
            after the fit_model has been applied, stacked along the first axis.
        is_stable : bool
            true if stability conditions are met. 
            See verify_stability_time_evolution() for more details. """
    
    F_ij, is_stable = caller.run_time_evo(model_configuration,
                                        integration_scheme, time_evo_max,
                                        dt_time_evo,ODE_state,
                                        ODE_coeff_model,ODE_coeff,
                                        stability_rel_tolerance,
                                        tail_length_stability_check,
                                        start_stability_check)
    F_i = F_ij[-1]

    return F_ij,F_i,is_stable 


def net_flux_fit_model(model_configuration,integration_scheme, 
                       time_evo_max, dt_time_evo,
                       idx_source, idx_sink,
                       ODE_state, ODE_coeff=None,
                       ODE_coeff_model=interaction_model_generator,
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
            Has the same unit as the one used in the initial ODE_state
        dt_time_evo : float
            Size of time step used in the time evolution.
            Has the same unit as the one used in the initial ODE_state
        b
        ODE_state : numpy.array
            1D array containing the initial state of the observed quantities
            in the ODE. Often also referred to as initial conditions.
        ODE_coeff : numpy.array
            2d-square-matrix containing the coefficients of the ODE
        ODE_coeff_model : function
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

        Returns:
        --------
        F_i : numpy.array
            2D-array containing the output of the time evolution
            after the fit_model has been applied, stacked along the first axis.
        is_stable : bool
            true if stability conditions are met. 
            See verify_stability_time_evolution() for more details. """

    F_ij, is_stable = caller.run_time_evo(model_configuration,
                                          integration_scheme, time_evo_max,
                                          dt_time_evo,ODE_state,
                                          ODE_coeff_model,ODE_coeff,
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

def holling_type_0(value):
    return value

def holling_type_II(food_processing_time,hunting_rate,prey_population):
    """ Holling type II function """
    consumption_rate = ((hunting_rate * prey_population)/
            (1+hunting_rate * food_processing_time * prey_population))
    return consumption_rate


def holling_type_III(epsilon,g,P):
    """ Holling type III function """
    G_val = (g*epsilon*P**2)/(g+(epsilon*P**2))
    return G_val


# Model_Classes 

class model_class:
	def __init__(self,path):
		self.initial_system_configuration = worker.initialize_ode_system(path)
		self.compartments = self.initial_system_configuration['compartments']
		self.states = self.initial_system_configuration['states']
		self.interactions = self.initial_system_configuration['interactions']
		self.configuration = self.initial_system_configuration['configuration']
		self.fetch_index_of_source_and_sink()	
		
		
	def initialize_log(self,n_monte_samples,max_gd_iter):
		parameters = self.to_grad_method()[0]
		param_log = np.zeros( (n_monte_samples,
									max_gd_iter, len(parameters)))
		prediction_log = np.zeros( (n_monte_samples, max_gd_iter,
								len(self.configuration['fit_target'])) )
		cost_log = np.zeros( (n_monte_samples, max_gd_iter) )
		model_log = np.zeros( (n_monte_samples, max_gd_iter,
							int(self.configuration['time_evo_max']/
							self.configuration['dt_time_evo']),
							len(list(self.compartments))))
	
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
		compartments = list(self.compartments)
		
		interaction_index = interactions.copy()
		for index,item in enumerate(interactions):
			interaction_index[index] = item.split(':')
		## parse them with the index
		for index, item in enumerate(interaction_index):
			interaction_index[index][0] = compartments.index(item[0])
			interaction_index[index][1] = compartments.index(item[1])

		return interaction_index


	def fetch_index_of_source_and_sink(self):
		sources = list(self.configuration['sources'])
		sinks = list(self.configuration['sinks'])
			
		idx_sources = sources.copy()
		idx_sinks = sinks.copy()

		for ii, item in enumerate(idx_sources):
			idx_sources[ii] = self.compartments.index(item)
		for ii, item in enumerate(idx_sinks):
			idx_sinks[ii] = self.compartments.index(item)
		
		self.configuration['idx_sources'] = idx_sources
		self.configuration['idx_sinks'] = idx_sinks


	def to_log(self,parameters,model_data,predictions,cost):
		#current monte sample
		ii = self.log['monte_carlo_idx']
		
		# reshape prediction in case they are zero-dim
		predictions = np.reshape(predictions,(len(predictions),1))
	
		self.log['parameters'][ii] = parameters
		self.log['predictions'][ii] = predictions
		self.log['model'][ii] = model_data
		self.log['cost'][ii] = cost
		
		self.log['gradient_idx'] += 1
		

	def to_ode(self):
		ODE_state = np.array([self.states[ii]['value'] for ii in self.states])
		ODE_coeff_model = interaction_model_generator
		ODE_coeff = ODE_coeff_model(self)
		
		return ODE_state,ODE_coeff_model, ODE_coeff


	def to_grad_method(self):
		""" Add docstring """
		
		free_parameters = []
		constraints = []
		for ii in self.states:
			if self.states[ii]['optimise'] is not None:
				value = self.states[ii]['value']
				lower_bound = self.states[ii]['optimise']['lower']
				upper_bound = self.states[ii]['optimise']['upper']
				free_parameters.append(value)
				constraints.append([lower_bound,upper_bound])

		for ii in self.interactions:
			#function
			for item in self.interactions[ii]:
				#parameters
				if item['optimise'] is not None:
					for jj,elements in enumerate(item['optimise']):
						value = item['parameters'][jj]
						lower_bound = elements['lower']
						upper_bound = elements['upper']

						free_parameters.append(value)
						constraints.append([lower_bound,upper_bound])
			
		free_parameters = np.array(free_parameters)
		constraints = np.array(constraints)

		return free_parameters, constraints


	def update_states(self, parameters):
		values = list(parameters)
		
		for ii in self.states:
			if self.states[ii]['optimise'] is not None:
				self.states[ii]['value'] = values.pop(0)
	
		for ii in self.interactions:
			#function
			for item in self.interactions[ii]:
				#parameters
				if item['optimise'] is not None:
					for jj,_ in enumerate(item['optimise']):
							item['parameters'][jj] = values.pop(0)

	
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
									#globals()[ode_coeff_model],
									# ISSUE - fix the hardcoding of interatcion model
									interaction_model_generator,
									float(self.configuration['stability_rel_tolerance']),
									self.configuration['tail_length_stability_check'],
									self.configuration['start_stability_check'])
		
		return model_log, prediction, is_stable