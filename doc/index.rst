.. nemf documentation master file, created by
   sphinx-quickstart on Tue May 26 14:51:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Network-based ecosystem Modeling Framework (NEMF)
=================================================

NEMF is a ecosystem modelling framework written in python.
It is designed to offer an easy to use method for modelling ecosystems with low- to intermediate complexity.  
The framework offers the functionality to handle non-equilibrium, non-linear interactions.
For the typical use-cases, user do need to write any code but only provide a model configuration.
Without needing to change any of the framework code, the range of use-cases can easily be extended with simple user-written functions 
NEMF offers an easy to use method that fits any parameter of the model to mimic the studied system.
For simplicity, the current version of the framework is limited to non-spatially resolved models (box-models).

For a brief introduction to the ideas behind this library, 
you can read the :doc:`introductory notes <introduction>`.
A more detailed description can be found in the paper [placeholder url].
Visit the :doc:`installation page <installation>` to see how you can download the 
package. 

.. You can browse the :doc:`examples <examples/index>` to see what you can do with 
.. NEMF, and then check out the :doc:`tutorial <tutorial>` and 

You can find more detailed description of certain parts of the library in the
:doc:`Manual <manual>`,
and detailed description on the code in :doc:`API reference <api>`.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   introduction.md
   Paper <paper>
   installation.md
   manual.rst
   releases.md
   api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
