import gemf
import pickle

def test_import(model_minimal_yml,model_minimal_pkl):
    assert model_minimal_yml.compartment == model_minimal_pkl.compartment
    assert model_minimal_yml.interactions == model_minimal_pkl.interactions
    assert model_minimal_yml.configuration == model_minimal_pkl.configuration

