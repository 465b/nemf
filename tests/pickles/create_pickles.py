import gemf
import pickle


# helper
def load_model_from_pickle(path='pickles/test_module.pkl'):
	with open(path,'rb') as f:
		model = pickle.load(f)
	return model


# pickler
def pickle_model():
	model = gemf.model_class('test_models/minimal_model_example.yml')
	with open('test_module.pkl','wb') as f:
		pickle.dump(model,f)


def pickle_forward():
	model = load_model_from_pickle()
	model = gemf.forward_model(model)
	with open('forward_out.pkl','wb') as f:
		pickle.dump(model,f)


def pickle_inverse():
	model = load_model_from_pickle()
	model = gemf.inverse_model(model,gd_max_iter=3+1,sample_sets=1)
	with open('inverse_out.pkl','wb') as f:
		pickle.dump(model,f)


if __name__ == "__main__":
	pickle_model()
	pickle_forward()
	pickle_inverse()
	print('pickled')