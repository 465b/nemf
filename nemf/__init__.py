
""" Main package for NEMF the network-based ecosystem modelling framework """

import nemf.caller
import nemf.models
import nemf.plot

from nemf.models import model_class as model_class
from nemf.models import model_class as load_model_configuration

from nemf.caller import forward_model as forward_model 
from nemf.caller import inverse_model as inverse_model

from nemf.plot import output_summary as output_summary
from nemf.plot import interaction_graph as interaction_graph



