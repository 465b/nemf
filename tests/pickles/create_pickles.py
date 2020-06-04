import nemf
import pickle


# helper
def load_model(pkl,folder='tests/pickles/'):
	path = folder+pkl
	with open(path,'rb') as f:
		model = pickle.load(f)
	return model

def write_pickle(model,name):
	with open('tests/pickles/test_'+name+'.pkl','wb') as f:
		pickle.dump(model,f)

def model_pickler(model,name):
	if type(model) == str:
		path = model
	model = nemf.model_class(path)
	write_pickle(model,name)

def forward_pickler(path,name):
	model = load_model(path)
	model = nemf.forward_model(model)
	write_pickle(model,name)
	
	
def inverse_pickler(path,name,method):
	model = load_model(path)
	model = nemf.inverse_model(model,method=method)
	write_pickle(model,name)
	

#def pickle_model():
#	model = nemf.model_class('test_models/minimal_model_example.yml')
#	with open('test_module.pkl','wb') as f:
#		pickle.dump(model,f)
#
#
#def pickle_forward():
#	model = load_model_from_pickle()
#	model = nemf.forward_model(model)
#	with open('forward_out.pkl','wb') as f:
#		pickle.dump(model,f)
#
#
#def pickle_inverse():
#	model = load_model_from_pickle()
#	model = nemf.inverse_model(model,gd_max_iter=3+1,sample_sets=1)
#	with open('inverse_out.pkl','wb') as f:
#		pickle.dump(model,f)


if __name__ == "__main__":
	
	model_pickler('tests/test_models/minimal_model_example.yml',
		'minimial_model')
	model_pickler('tests/test_models/NPZD_model_oscillating.yml',
		'NPZD_osci')
	model_pickler('tests/test_models/NPZD_model_stable.yml',
		'NPZD_stable')
	model_pickler('tests/test_models/NPZD_model_oscillating_refed.yml',
		'NPZD_osci_refed')
	model_pickler('tests/test_models/NPZD_model_stable_refed.yml',
		'NPZD_stable_refed')

	forward_pickler('test_NPZD_osci.pkl',
		'NPZD_osci_forward')
	
	inverse_pickler('test_NPZD_osci_refed.pkl',
		'NPZD_osci_inverse_SLSQP',method='SLSQP')
	inverse_pickler('test_NPZD_osci_refed.pkl',
			'NPZD_osci_inverse_trust',method='trust-constr')