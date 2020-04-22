# Example D5 Model
import modules.models as models
import modules.worker as worker
import modules.caller as caller
import modules.plot as plot

import matplotlib.pyplot as plt


# path of model/system configuration
path = ('configuration_files/NPZD_model_config.yml')

model_config = models.model_class(path)
plot.interaction_graph(model_config)

# calls the top level executable
output_dict = caller.dn_monte_carlo(path,
    sample_sets=1, gd_max_iter=100, grad_scale=1e-1)

# simple plot of some results
plot.output_summary(output_dict)