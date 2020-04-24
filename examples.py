# NPZD Example 

# import the module
import gemf

# provide the path of model/system configuration
path = ('configuration_files/NPZD_model_config.yml')

# load the model configuration
model_config = gemf.model_class(path)
# visualise the model configuration to check for errors
gemf.interaction_graph(model_config)


# for a simple time evolution of the model call:
output_dict = gemf.forward_model(model_config)
# the results of the time evolution can be visualized with:
gemf.output_summary(output_dict)


# if the model shall be fitted as well call:
output_dict = gemf.inverse_model(model_config,
    sample_sets=1, gd_max_iter=100, grad_scale=1e-1)
# to plot the results:
gemf.output_summary(output_dict)