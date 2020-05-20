import gemf
import pickle
import pytest

def load_model(pkl,folder='tests/pickles/'):
	path = folder+pkl
	with open(path,'rb') as f:
		model = pickle.load(f)
	return model


## YAML

@pytest.fixture
def model_minimal_yml():
	return gemf.model_class('tests/test_models/minimal_model_example.yml')


## PICKLES

@pytest.fixture
def model_npzd_osci_pkl():
	return load_model('test_NPZD_osci.pkl')

@pytest.fixture
def model_npzd_stable_pkl():
	return load_model('test_NPZD_stable.pkl')

@pytest.fixture
def model_npzd_osci_refed_pkl():
	return load_model('test_NPZD_osci_refed.pkl')

@pytest.fixture
def model_npzd_stable_refed_pkl():
	return load_model('test_NPZD_stable_refed.pkl')

@pytest.fixture
def model_minimal_pkl():
	return load_model('test_minimial_model.pkl')

@pytest.fixture
def forward_osci_pkl():
	return load_model('test_NPZD_osci_forward.pkl')

@pytest.fixture
def inverse_osci_SLSQP_pkl():
	return load_model('test_NPZD_osci_inverse_SLSQP.pkl')

@pytest.fixture
def inverse_osci_trust_pkl():
	return load_model('test_NPZD_osci_inverse_trust.pkl')

