API reference
=============

   
Importing Model
---------------

.. automodule:: nemf
   :members: load_model
   :show-inheritance:


Visualizing Model
-----------------

.. automodule:: nemf
   :members: interaction_graph
   :noindex:


Forcasting Model
----------------

.. automodule:: nemf
   :members: forward_model
   :noindex:


Fitting Model
-------------

.. automodule:: nemf
   :members: inverse_model
   :noindex:


Visualizing Results
-------------------

.. automodule:: nemf
   :members: output_summary
   :noindex:

Interaction Functions
---------------------

Here a list of all currently implemented interaction functions is presented.
These can be used in the YAML model configuration to describe interactions 
between to compartments.

.. note::

   The compartments are referenced via indices in the implementation of the 
   interaction functions.
   This is simply an implementation detail. 
   The user can (and should) use the compartment names when referencing them in 
   the YAML configuration file.
   The framework handles the mapping of the names to the corresponding indeces 
   internally.

.. automodule:: nemf.interaction_functions
   :members:
   :undoc-members:
   :show-inheritance:
