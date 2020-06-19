
# Main package for NEMF the network-based ecosystem modelling framework 

import nemf.caller
import nemf.models
import nemf.plot
import nemf.interaction_functions

from nemf.models import model_class as model_class

def load_model(model_path,ref_data_path=None):
	""" Loads a model by reading its model configuration from from file

	Parameters
	----------
	
	model_path : string
		path to yaml file containing the model
	ref_data_path : string (optional)
		path to file, plain text or xls(x), containing reference data to the 
		model.
		See: https://nemf.readthedocs.io/en/latest/README_reference_data.html

	Returns
	-------

	model : nemf_model class
		class objects that contains the model
	
	"""

	return model_class(model_path,ref_data_path)


from nemf.caller import forward_model as forward_model 
from nemf.caller import inverse_model as inverse_model

from nemf.plot import output_summary as output_summary
from nemf.plot import interaction_graph as interaction_graph



