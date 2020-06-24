# Overview of the YAML configuration file

The idea of this configuration file is to provide a simple, well structured, easily human-readable configuration of the full framework.
A run of the framework is fully determined by this configuration file.
It in
This contains the configuration of the interaction network, the resulting ODEs,
their time integration, the fitting methods and its configuration including the 
reference to the data that shall be used to fit.

If you first want to take a look at a rough overview of the library,
see [introduction](../introduction).
If you would like to get a better understanding how the library works internally,
see [manual/concept](concept).

Even a the small NPZD-type model that is presented in the library overview
has numerous interactions and many more parameters and constants.
In addition to the configuration of the model,
we also need to define what parameters we want to fit and how.
Therefore, the configuration of them can becomes become confusing and we
 we which need a clear structure to define them.
To avoid loosing track of all these individual things,
we store all in one well structured file.
YAML provides a ideal format for this,
since it is easily readable by humans as well as machines.

## What is defined in the file?
The file contains three sections: Compartments, Interactions, Configuration.
Compartments and Interaction are used to construct the network that is used to
construct the set of ODEs. 
Configuration defines the methods for time integration and fitting.

### Compartments
Compartments are the observed and modelled quantity of the ODE system.
In the NPZD model, they are:
(N)utritients, (P)hytoplankton, (Z)ooplankton, (D)etritus.
All are counted in a shared currency, i.e. carbon mass.

The relevant information that needs to be defined is:
* name
* initial value (in the time evolution)
* upper and lower bound, if this value shall be fitted

In the YAML format this looks like the following:

``` yaml
compartment:      # header of the compartment section
  N:              # name of compartment 
    value: 1.0    # initial value of the compartment
    optimise:     
      lower: 0    # lower and
      upper: 2    # upper bound during the fitting process
  P:
    value: 1.0
    optimise: null # no fitting required for this compartment
```  
A more detailed description of all the possible options is provided further down.

### Interaction

Interactions represent any flow from one compartment to an other.
A interaction is described either by a function (keyword: 'fkt')
defined in the framework library (see [models.py/#interaction_models](../api)),\
or by a user defined function.
The only condition to the function name provided in the yaml file
is that has to match the function known to the python interpreter exactly.

In addition to the name of the compartments and their interaction function,
we require the set of parameters used in the function to be defined.
These can either be constants, or compartments.
Furthermore, the direction of the flow needs to be defined ('sign').

We can define individual parameters to be fitted.
This is done by providing the index of the parameter (i.e.: second parameter)
and its range.
Naturally, this does not apply if the selected parameter is a compartment
because they are calculated during the time evolution.

A small example of the interaction of between N and P might look like the following:
``` yaml
interactions:
  # the functions are automatically multiplied by the value
  # of the second compartment 
  N:P:                              # name of source:destination of the flow
  - fkt: nutrient_limited_growth   # function defining the type of flow
    parameters:
    - 'N'                           # the first parameter is the current value
    - 0.27                          # of the 'N' compartment
    - 0.7
    sign: '-1'                      # direction of flow (from P to N)
    optimise:
      - parameter_no: 2             # second parameter (0.27) is optimized
        lower: 0.1                  # and values in the range of 
        upper: 0.3                  # [0.1,0.3] are allowed to be used
```
A crucial implementation detail is, that by default, the first named compartment
is implicitly passed as an parameter to the interaction functions.
The second compartment needs to be explicitly named.


### Configuration

This section of the file contains all technical implementation details
of the ODE solver and inverse problem solver.
Find a annotated example below.

``` yaml
# ode solver related details:
time_evo_max: 100
dt_time_evo: 0.1
```

#### ode solver related configuration details:

* **time_exo_max**: Ideally the ode system reached a stable state and stops
  automatically.
  If this is not the case it stops the latest at the ode-model time defined here.
* **dt_time_evo** defines the step size in ode-model time
  needed for a net-flux fit model.