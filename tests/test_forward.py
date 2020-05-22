import gemf
import pickle

#from . import conftest
#import matplotlib.pyplot as plt

def test_forward_model_v1(model_minimal_pkl):
    gemf.forward_model(model_minimal_pkl)


def test_forward_model_v2(model_npzd_osci_pkl):
    gemf.forward_model(model_npzd_osci_pkl)


def test_forward_model_v3(model_npzd_osci_pkl):
    gemf.forward_model(model_npzd_osci_pkl)


def test_forward_model_v4(model_npzd_stable_pkl):
    gemf.forward_model(model_npzd_stable_pkl)


def test_forward_model_v5(model_npzd_osci_refed_pkl):
    gemf.forward_model(model_npzd_osci_refed_pkl)


def test_forward_model_v6(model_npzd_stable_refed_pkl):
    gemf.forward_model(model_npzd_stable_refed_pkl)