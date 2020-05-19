import gemf
import pickle

def load_model_from_pickle(path='tests/pickles/test_module.pkl'):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model


def test_import():
    path = 'tests/test_models/minimal_model_example.yml'
    yaml_read_model = gemf.model_class(path)
    pickle_read_model = load_model_from_pickle()
    print()
    assert yaml_read_model.configuration==pickle_read_model.configuration

