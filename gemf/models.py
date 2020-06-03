import numpy as np
from gemf import caller
from gemf import worker
from gemf.interaction_functions import *
from copy import deepcopy


# Model_Classes 
class model_class:

	# initialization methods
	# they are only used when a new model_class is created

	def __init__(self,model_path,fit_data_path=None):
		self.init_sys_config = worker.initialize_ode_system(model_path)
		self.sanity_check_input()
		self.compartment = deepcopy(
			self.init_sys_config['compartment'])
		self.interactions = deepcopy(
			self.init_sys_config['interactions'])
		self.configuration = deepcopy(
			self.init_sys_config['configuration'])
		self.reference_data = self.load_reference_data(fit_data_path)
		if ('sinks' in self.configuration) and ('sources' in self.configuration):
			self.fetch_index_of_source_and_sink()
		if ('alternative_interaction_names' in self.configuration):
			self.init_alternative_interaction_names()
		if 'constraints_path' in self.configuration:
			self.load_constraints(self.configuration['constraints_path'])


	def load_reference_data(self,fit_data_path=None):
		""" Loads reference data used in model optimization from file 

		Either, the path to the reference data is provided in the yaml 
		configuration file or passed to this function.
		Latter overwrites the path in the configuration file.

		Parameters
		----------

		fit_data_path : string (optional)
			path to the file containing the reference data

		Returns
		-------

		reference_data : np.array or None
			if reference data was found returns it in a numpy array or if not
			or no path provided returns None

		"""

		if fit_data_path != None:
			reference_data = worker.import_fit_data(fit_data_path)
			if len(np.shape(reference_data)) == 1:
				reference_data = np.reshape(reference_data,(1,len(reference_data)))
			return reference_data
		elif 'fit_data_path' in self.configuration:
			fit_data_path = self.configuration['fit_data_path']
			reference_data = worker.import_fit_data(fit_data_path)
			if len(np.shape(reference_data)) == 1:
				reference_data = np.reshape(reference_data,(1,len(reference_data)))
			return reference_data
		else:
			print('No reference data has been provided')
			return None


	def load_constraints(self,path):
		""" Loads constraints from python file 
		
		Parameters
		----------
		path : string
			path to python file
			
		"""

		self.configuration['constraints'] = \
			worker.import_constraints(path)


	def sanity_check_input(self):
		""" 
			checks for obvious errors in the configuration file.
			passing this test doesn't guarante a correct configuration. 
		"""
		unit = self.init_sys_config

		# checks if compartment is well defined
		name = "Model configuration "
		worker.assert_if_exists_non_empty(
			'compartment',unit,reference='compartment')
		assert (len(list(unit['compartment'])) > 1), \
			name + "only contains a single compartment"
		## check if all compartment are well defined
		for item in list(unit['compartment']):
			worker.assert_if_non_empty(
				item,unit['compartment'],item,reference='state')
			worker.assert_if_exists('optimise',unit['compartment'][item],item)
			worker.assert_if_exists_non_empty('value',unit['compartment'][item],
									   item,'value')
			worker.assert_if_exists('optimise',unit['compartment'][item],item)
			
		# checks if interactions is well defined
		worker.assert_if_exists_non_empty('interactions',unit,'interactions')
		## check if all interactions are well defined
		for item in list(unit['interactions']):
			for edge in unit['interactions'][item]:
				assert edge != None, \
					name + "interaction {} is empty".format(item)
				worker.assert_if_exists_non_empty('fkt',edge,item,)
				worker.assert_if_exists('parameters',edge,item)

		# checks if configuration is well defined
		worker.assert_if_exists_non_empty('configuration', unit)
		required_elements = ['time_evo_max']
		for element in required_elements:
			worker.assert_if_exists_non_empty(element,unit['configuration'])


	def init_alternative_interaction_names(self):
		""" add renamed copies defined in the yaml config to globals() """
		alt_names = self.configuration['alternative_interaction_names']
		for new_func in list(alt_names):
			orig_func = globals()[alt_names[new_func]]
			globals()[new_func] = orig_func


	def initialize_log(self,maxiter):

		max_iter = maxiter + 1	
		fit_parameter = self.fetch_to_optimize_args()[0][1]

		param_log = np.full((max_iter,len(fit_parameter)), np.nan)
		cost_log = np.full( (max_iter), np.nan )
		
		log_dict = {'parameters': param_log,
					'cost': cost_log,
					'iter_idx': 0}
		
		self.log = log_dict

	
	# Logging

	def construct_callback(self,method='SLSQP',debug=False):
		model = self

		if method == 'trust-constr':
			def callback(xk, opt):# -> bool
				if debug: print(f'xk: \n{xk}')
				model.to_log(xk,cost=opt.fun)
		elif method == 'SLSQP':
			def callback(xk):# -> bool
				if debug: print(f'xk: \n{xk}')
				model.to_log(xk)
		else:
			raise Exception
			
		return callback



	def to_log(self,parameters,cost=None):
		#current monte sample
		idx = self.log['iter_idx']
		self.log['parameters'][idx] = parameters
		self.log['cost'][idx] = cost
		self.log['iter_idx'] += 1


	# Parsing

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
				if 'optimise' in item:
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


	def update_system_with_parameters(self, parameters):
		values = list(parameters)
		
		for ii in self.compartment:
			if self.compartment[ii]['optimise'] is not None:
				self.compartment[ii]['value'] = values.pop(0)
	
		for ii in self.interactions:
			#function
			for item in self.interactions[ii]:
				#parameters
				if 'optmise' in item:
					if item['optimise'] is not None:
						for element in item['optimise']:
							item['parameters'][element['parameter_no']-1] = \
								values.pop(0)


	# Fetching
	# they retrieve some information from the model and output them in a certain
	# form. They mostly look through dicts to extract a subset of them
	
	def fetch_constraints(self):
		# placeholder for constraints generator
		if 'constraints' in self.configuration:
			return self.configuration['constraints']
		else:
			return None


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


	def fetch_index_of_compartment(self,parameters):
		# add something to make sure all stanges to parameters stay internal
		compartments = list(self.compartment)
		for nn,entry in enumerate(parameters):
			if (type(entry) == str) & (entry in list(self.compartment)):
				#print(entry, compartments.index(entry))
				parameters[nn] = compartments.index(entry)
		return parameters


	def fetch_to_optimize_args(self):
		""" fetches the parameters necessary for the gradient descent method
			Returns: free_parameters, constraints """

		labels = []
		idx_state = []; val_state = []; bnd_state = []
		
		for ii,entry in enumerate(self.compartment):
			if self.compartment[entry]['optimise'] is not None:
				labels.append('{}'.format(entry))
				
				idx_state.append(ii)
				
				val_state.append(self.compartment[entry]['value'])
				
				lower_bound = self.compartment[entry]['optimise']['lower']
				upper_bound = self.compartment[entry]['optimise']['upper']
				bnd_state.append([lower_bound,upper_bound])
				
		
		idx_args = []; val_args = []; bnd_args = []
		
		for ii,interaction in enumerate(self.interactions):
			for jj,function in enumerate(self.interactions[interaction]):
				if 'optimise' in function:
					if function['optimise'] is not None:
						for kk,parameter in enumerate(function['optimise']):
							labels.append('{},fkt: {} #{}'.format(
								interaction,function['fkt'],
								parameter['parameter_no']))
							
							current_idx_args = \
								[ii,jj,parameter['parameter_no']-1]
							current_val_args = \
								self.fetch_arg_by_idx(current_idx_args)				
							lower_bound = parameter['lower']
							upper_bound = parameter['upper']
							
							idx_args.append(current_idx_args)
							val_args.append(current_val_args)
							bnd_args.append( (lower_bound,upper_bound) )


		fit_indices = [idx_state,idx_args]
		fit_param = val_state + val_args
		bnd_param = bnd_state + bnd_args
			
		return [fit_indices,fit_param,bnd_param], labels
		

	def fetch_states(self):
		states = []
		compartment = self.compartment
		for item in compartment:
			states.append(compartment[item]['value'])
		return states

	
	def fetch_args(self):
		args = []
		for interactions in self.interactions:
			args_edge = []
			for edges in self.interactions[interactions]:
				indexed_args = self.fetch_index_of_compartment(edges['parameters'])
				args_edge.append(indexed_args)

			args.append(args_edge)
		return args

	
	def fetch_param(self):
		states = self.fetch_states()
		args = self.fetch_args()
		return [states,args]


	def fetch_arg_by_idx(self,index):
		args = self.fetch_args()
		idx = index
		arg = args[idx[0]][idx[1]][idx[2]]
		return arg


	def de_constructor(self):
		# the benefit of constructing it like this is that:
		#    * we are able to get the signature f(x,args)
		#    * all non-(x,args) related objects are only evaluated once.
		# however, this for looping is still super inefficient and a more
		# vectorized object should be intended
	
		# args is expected to have the same shape as set_of_functions
		set_of_function = self.interactions
		idx_interactions = self.fetch_index_of_interaction()
		n_compartments = len(self.compartment)
	
		def differential_equation(t,x,args): #(t,x) later
		
			y = np.zeros((n_compartments,n_compartments))
	
			for ii,functions in enumerate(set_of_function):
				interaction = set_of_function[functions]          
				for jj,edge in enumerate(interaction):
					kk,ll = idx_interactions[ii]
					#print(f'{edge["fkt"]} \t\t flows into '+'
					# {list(self\.compartment)[kk]} outof {list(self\.compartment)[ll]}')
	
					# flows into kk (outof ll)
					y[kk,ll] += globals()[edge['fkt']](x,kk,*args[ii][jj])
					# flow outof ll (into kk)
					y[ll,kk] -= globals()[edge['fkt']](x,kk,*args[ii][jj])
	
			return np.sum(y,axis=1)
	
		return differential_equation


def import_interaction_functions(func):
	""" Adds the functions from 'func' to the globals in models 

	func : list
		List containing the functions that will be added to globals

	"""
	for item in func:
		name = item.__name__
		if name in globals():
			print(f'Warning! A function with the same name as interaction ' 
				+ 'function {name} is already known in globals!\n'
				+ 'Function will be overwritten!')
		globals()[name] = item