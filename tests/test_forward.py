import gemf
import pickle
#import matplotlib.pyplot as plt

def load_model_from_pickle(path='tests/pickles/test_module.pkl'):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model


def test_forward_model_v1():
    model = load_model_from_pickle()
    gemf.forward_model(model)


def test_forward_model_v2():
    model = load_model_from_pickle()
    model.configuration['integration_scheme'] = 'runge_kutta'
    gemf.forward_model(model)