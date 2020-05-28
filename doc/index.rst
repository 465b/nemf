.. gemf documentation master file, created by
   sphinx-quickstart on Tue May 26 14:51:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

General Ecosystem Modeling Framework (GEMF)
===========================================

GEMF is a ecosystem modelling framework written in python.
It is designed to offer an easy to use method for modelling ecosystems with low- to intermediate complexity.  
The framework offers the functionality to handle non-equilibrium, non-linear interactions.
For the typical use-cases, user do need to write any code but only provide a model configuration.
The use-cases can easily be extended with simple user written functions without needing to change any of the framework code.  
The framework offers an easy to use method that fits any parameter of the model to mimic the studied system.
For simplicity, the current version of the framework is limited to non-spatially resolved models (box-models).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   README_concept.md
   README_YAML.md
   README_interaction_functions.md
   README_reference_data.md
   releases.md
   modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
