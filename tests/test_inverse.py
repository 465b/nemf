import gemf
import pickle
#import matplotlib.pyplot as plt

def load_model_from_pickle(path='tests/pickles/test_module.pkl'):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model



def test_inverse_model_v1():
    model = load_model_from_pickle()
    gemf.inverse_model(model,gd_max_iter=3+1,sample_sets=1)

def test_inverse_model_v2():
    model = load_model_from_pickle()
    gemf.inverse_model(model,gd_max_iter=3+1,sample_sets=0)

def test_inverse_model_v3():
    model = load_model_from_pickle()
    gemf.inverse_model(model,gd_max_iter=3+1,sample_sets=2)