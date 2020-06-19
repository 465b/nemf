# NPZD Example 

# import the module
import nemf

# provide the path of model/system configuration
model_path = 'example_files/NPZD/NPZD_model.yml'
ref_data_path = 'example_files/NPZD/NPZD_ref_oscillating.csv'

# load the model configuration
model_config = nemf.model_class(model_path,ref_data_path)

# visualise the model configuration to check for errors
nemf.interaction_graph(model_config)


# for a simple time evolution of the model call:
output_dict = nemf.forward_model(model_config)
# the results of the time evolution can be visualized with:
nemf.output_summary(output_dict)


# if the model shall be fitted as well call:
model = nemf.inverse_model(model_config)
# to plot the results:
nemf.output_summary(model)
# writes optimized  model to file
model.export_to_yaml(path='optimized_model.yml')