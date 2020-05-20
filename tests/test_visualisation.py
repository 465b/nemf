import gemf
import pickle

def load_model_from_pickle(path='tests/pickles/test_module.pkl'):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model


def test_graph_visualization(model_npzd_osci_pkl):
    gemf.plot.draw_interaction_graph(model_npzd_osci_pkl)


def test_forward_visualization(forward_osci_pkl):
    gemf.plot.draw_output_summary(forward_osci_pkl)


def test_inverse_visualization_v1(inverse_osci_SLSQP_pkl):
    gemf.plot.draw_output_summary(inverse_osci_SLSQP_pkl)


def test_inverse_visualization_v2(inverse_osci_trust_pkl):
    gemf.plot.draw_output_summary(inverse_osci_trust_pkl)
