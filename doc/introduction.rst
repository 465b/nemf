An introduction to NEMF
=======================

The network-based ecosystem modelling framework (NEMF) is a framework
for general ecosystem modelling problems. It consists of three
conceptual parts:

1. *Network-based model description*
2. *Forward modelling*
   (By this we mean, numerically solving the differential equations
   implicitly defined by the network. Hence, providing forecasts how the
   model behaves.)
3. *Inverse modelling*
   (Meaning the fitting of model forecasts to observational or other
   reference data.)

It aims to keep the configuration complexity minimal for the user such
that it can be quickly learned and applied for i.e. rapid prototyping of
model ideas. To keep configuration and computational complexity low it
can only applied to non-spatially-resolved, also known as box-models.

Network and Forward
~~~~~~~~~~~~~~~~~~~

An example of what we mean by that:

.. code:: python

   import nemf
   model = nemf.load_model('exemplary_npzd_model.yml')
   nemf.interaction_graph(model)
   results = nemf.forward_model(model)
   nemf.output_summary(results)


.. list-table::

  * - .. figure:: figures/network_diagram.png

         Network diagram of the ecosystem model.
    
    - .. figure:: figures/time_evo.png
         
         Time evolution of the modeled system.


Let’s go through the lines one by one to see what happened:

1. First we imported the nemf python library.

   .. code:: python

      import nemf

   This tells python that we want to us this library and because not
   stated otherwise that we will address it as *nemf*

2. We tell the nemf library which model we want to use.

   .. code:: python

      model = nemf.load_model('exemplary_npzd_model.yml')

   Models are typically defined in an extra file. This file contains the
   description of the model in a humon-readable standard called YAML.
   Hence, the file extension *.yml* More on the yml standard and how it
   is used to define models can be found *here* [placeholer]

3. We visualize the network defined in the model, as shown in the left
   figure.

   .. code:: python

      nemf.interaction_graph(model)

   NEMF offers the option to draw up the network defined in the model
   configuration. This is helps to catch errors that might have happened
   during the configuration and gives a nice overview over the model.
   Each knot represents a compartment in the model, i.e. population of
   chemical quantities. The arrows between them show what flows from one
   compartment to another while the label on the arrow describes how it
   does that.

4. We solve the differential equations underlying the model numerically
   with:

   .. code:: python

      results = nemf.forward_model(model)

   The network with its flows between compartments implicitly defines a
   set of differential equation that couples the compartments to each
   other. The framework solves these differential equations to give a
   forecast how the model is expected to evolve over time.

5. The result of the forecasting are visulized by calling:

   .. code:: python

      nemf.output_summary(results)

   This is shown in the right hand side figure above. Each line
   represents one compartment and how its associated quantity (i.e. a
   population size) changes over time.


Model description via YAML configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the example above, we assumed that a model (*exemplary_npzd_model*) has 
already been defined.
If we want to construct a new model, we need to write our own configuration 
file.

There are three major parts of the configuration file:

1. **Compartments**
   contains a list of all model compartments, like species population pools or 
   nutrition pools.

2. **Interactions**
   contains a list of all interactions between compartments, like what eats what
   and what happens when it dies

3. **Configuration**
   contains a list of technical details that decide the framework behavior 
   during the forecast and fitting.

The configuration file is written in the YAML standard. 

It consists of what is called key-value pairs. 
Each key is associated with a value.
These values can also be lists which are indicated by leading "-", like 
bullet points.
You can find details about the YAML standard on http://yaml.org/.
Note that the YAML website is itself perfectly valid yaml.


A simple example for the compartment section looks like this: ::
 
  compartment:      # header of the compartment section
    A:              # name of compartment 
      value: 1.0    # initial value of the compartment
    B:
      value: 2.0
    ..
    ..

Interactions are defined similarly: ::

  A:B:              # flow from B to A (predator:prey)
  - fkt: grazing    # type of interaction
  - parameters:     # parameters used for the interaction
    - 1             # i.e. hunting rate
    - 2             #      food processing time
  B:A
  - fkt: natural mortality
  - parameters:
    - 0.01          # natural mortality rate


A description of how this works in detail can be found in section 
:doc:`YAML manual<README_YAML>`.


Inverse modelling
~~~~~~~~~~~~~~~~~

So far, we covered the first two aspect; the network-based approach and
the forward modelling.
We can also fit unknown, or imprecisely known parameters such that the forecast 
resembles a provided data set as closely as possible.

We can achieve this with the *inverse_model* method.

.. code:: python

   import nemf
   model = nemf.load_model('exemplary_npzd_model.yml')
   results = nemf.inverse_model(model)
   nemf.output_summary(results)

Most of this code is the same as previously shown.
The only new line is:

.. code:: python

   results = nemf.inverse_model(model)

Instead of calculating the forecast once as previously, the *inverse_model* we 
now calculates it for different sets of parameters in such a way that we find 
the best solution quickly.

However, for this to work we implicitly provided some additional information in 
the yaml configuration file.
There are two things we need to provide:

1. Reference Data (i.e observational data)
2. Optimized parameters

The reference data is expected do be in a separate file.
Details about its format and how it can be imported can be found in the 
:doc:`reference data section of the manual<README_reference_data>`.


The parameters that shall be optimized are selected in the YAML configuration 
file by adding the 'optimise' key and providing its upper and lower bounds in
which the method tries to find the best solution.

.. code:: python

   compartment:      # header of the compartment section
     A:              # name of compartment 
       value: 1.0    # initial value of the compartment
       optimise:     
         lower: 0    # lower and
         upper: 2    # upper bound during the fitting process

Detail on the configuration of the YAML file can be found in the YAML section 
the :doc:`yaml manual<README_YAML>`.

The results are then visualized with:

.. code:: python

   nemf.output_summary(results)


Which creates the following figure:

.. figure:: figures/fit_results.png
   :alt: Visualization of model fit

   This is the caption of the figure (a simple paragraph).
   Visualization of the results of the model fit.
   The upper figure shows the tested parameter during the fitting process,
   while the lower figure shows the "optimally" fitted model.



Next steps
----------

You have a few options for where to go next. You might first want to learn how 
to :doc:`install name<installation>`. 
Once that’s done, you can browse the :doc:`examples<examples>` to get a 
broader sense for what kind problems nemf is designed for. 
Or you can read through the manual for a deeper discussion of the 
different parts of the library and how they are designed.
If you want to know specifics of the nemf functions implementations, 
you could check out the :doc:`API reference<api>`, which documents each 
function and its parameters.

