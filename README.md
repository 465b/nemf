# NEMF: network-based ecosystem modeling framework


<div class="row">

<a href=https://nemf.readthedocs.io/en/latest/introduction.html#network-and-forward>
<img src="doc/figures/model_graph_small_square.png" height="250" width="250">
</a>

<a href=https://nemf.readthedocs.io/en/latest/introduction.html#network-and-forward>
<img src="doc/figures/model_forecast_small_square.png" height="250" width="250">
</a>

<a href=https://nemf.readthedocs.io/en/latest/introduction.html#inverse-modelling>
<img src="doc/figures/model_fit_small_square.png" height="250" width="250">
</a>

</div>

---

[![build & publish](https://github.com/465b/nemf/workflows/build%20&%20publish/badge.svg)](https://pypi.org/project/nemf/)
[![Documentation Status](https://readthedocs.org/projects/nemf/badge/?version=latest)](https://nemf.readthedocs.io/en/latest/?badge=latest)

NEMF is a ecosystem modelling framework written in python.
It is designed to offer an easy to use method for modelling ecosystems with low- to intermediate complexity.  
The framework offers the functionality to handle non-equilibrium, non-linear interactions.
For the typical use-cases, user do need to write any code but only provide a model configuration.
The use-cases can easily be extended with simple user written functions without needing to change any of the framework code.  
The framework offers an easy to use method that fits any parameter of the model to mimic the studied system.
For simplicity, the current version of the framework is limited to non-spatially resolved models (box-models).

A detailed introduction and many more resources can be found at: nemf.readthedocs.io

## Installation

The package can be installed using pythons integrated package manager *pip*.
``` bash
pip install nemf
```
For a detailed description how it can be installed, see 
[install notes](https://nemf.readthedocs.io/en/latest/installation.html)

## Quick Start

See examples.py. For a set of exemplary framework configurations see 
[configuration_files](example_files/)


## Usage


``` python
import nemf as 

# A model configuration is defined in a yaml file. To read this file:
model = nemf.load_model('path/to/model_configuration.yml')

# To draw a graph visualizing the model configuration:
nemf.interaction_graph(model)

# To solve the time integration of this model call:
forward_results = nemf.forward_model(model)
# To also solve the inverse problem as defined in the configuration, call:
inverse_results = nemf.inverse_model(model)

# To plot the results call 
nemf.plot.output_summary(inverse_results)
```

For details on the yaml configuration file, see 
[yaml documentation](https://nemf.readthedocs.io/en/latest/manual/YAML.html)  
The presented functions offer many (crucial) options, which are discussed in 
the [documentation](https://nemf.readthedocs.io/en/latest/api.html).


## How does it work?

For a conceptual description of the internals of this library, see 
[model concept](https://nemf.readthedocs.io/en/latest/manual/concept.html)


## Example

A simple example might be a simple NPZD model as presented below:
![interaction graph](doc/figures/network_diagram.svg "Exemplary interaction graph")

The user needs to define the compartments and interactions between them.
If the model should be fitted, they also need to provide some constraints and some data to fit it to.  
After that the model automatically generates a graph to visualize the system and to find potential errors in the configuration.

Once the model is configured the framework handles the time integration as well as the fitting without any further required user-interaction.


The results of such a run might then look like the following:
![exemplary results](doc/figures/exemplary_results.svg "exemplary fit results")

Top left shows the cost of the current model configuration. The cost is a quadratic measure of distance of the current model to the desired model.
Top right shows a output of the model after it reached its steady state for every parameter set tested.
Bottom left shows the all tested parameters sets.
Bottom right shows the full model output of the best fitted model found.