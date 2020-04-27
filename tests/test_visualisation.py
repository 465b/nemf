import gemf
import pickle

def load_model_from_pickle(path='tests/pickles/test_module.pkl'):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model


def test_graph_visualization():
    model = load_model_from_pickle('tests/pickles/test_module.pkl')
    gemf.plot.draw_interaction_graph(model)


def test_forward_visualization():
    model = load_model_from_pickle('tests/pickles/forward_out.pkl')
    gemf.plot.draw_output_summary(model)


def test_inverse_visualization():
    model = load_model_from_pickle('tests/pickles/inverse_out.pkl')
    gemf.plot.draw_output_summary(model)

