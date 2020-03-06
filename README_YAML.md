# Overview of the YAML configuration file

The key idea of this file is provide a simple, well structured, easily human-readable configuration of the model.
By model, we mean the ODE system model as well as the fitting model.

If you are looking for a rough overview of the library, see [README.md](README.md).
If you would like to get a better understanding of how the library works internally, see [README_concept.md](README_concept.md)

Even a small NPZD-type model has numerous interactions and many more parameters and constants which need to be defined.
In addition to the configuration of the system of ODE's, we also need to define what parameters we want to fit and how.

To avoid loosing track of all these individual things, we store all in one well structured file.
YAML provides a ideal format for this, since it is easily readable by humans as well as machines.

## What is defined in the file?
### Compartments
Compartments are the observed and modelled quantity of the ODE system.
In a NPZD model, they would be exactly them: (N)utritients, (P)hytoplankton, (Z)ooplankton, (D)etritus.

The relevant information we define here is:
* their name
* initial value to be used in the time evolution
* if this value shall be fitted, if so, its upper and lower bound

In the YAML format this looks like the following:

``` yaml
states:
  N:
    optimise:
      lower: 0
      upper: 2
    value: 1.0
  P:
    optimise: null 
    value: 1.0
```  
A more detailed description of all the possible options is provided further down.

### Interaction

Interactions represent any flow from one compartment to an other.
A interaction is described either by a function ('fkt') defined in the model (see [models.py/#interaction_models](modules/models.py)), or by a user defined function.
The only condition is that the name provided in the yaml file has to match the function known to the interpreter exactly.

In addition to the name of the compartments and their interaction function, we require the set of parameters used in the function to be defined. These can either be constants, or compartments.
Furthermore, the direction of the flow needs to be defined ('sign').

We can define individual parameters to be fitted, similarly to the fitting of the compartments.
This is done by providing the index of the parameter (i.e.: second parameter) and its range.
Naturally, this does not apply if the selected parameter is a compartment, as there optimisation is defined in their definition.

A small example of the interaction of between N and P might look like the following:
``` yaml
interactions:
  # the functions are automatically multiplied by the value
  # of the second compartment 
  N:P:
  - fkt: nutrition_limited_growth 
    optimise:
    parameters:
    - 'N'
    - 0.27
    - 0.7
    sign: '-1'
```
A crucial implementation detail is, that by default, an interaction is always linear with respect to the second named compartment.
Meaning that if I want to model the following flow,
<p align=center>
<a href="https://www.codecogs.com/eqnedit.php?latex=\partial_{t}A&space;=&space;-f_{AB}(x_{1},x_{2})B&space;|_{x_{1}=1,&space;x_{2}=2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\partial_{t}A&space;=&space;-f_{AB}(x_{1},x_{2})B&space;|_{x_{1}=1,&space;x_{2}=2}" title="\partial_{t}A = -f_{AB}(x_{1},x_{2})B |_{x_{1}=1, x_{2}=2}" /></a>
</p>
it has to be defined like:

``` yaml
A:B:
  - fkt: f_AB
  - parameters:
    - 1
    - 2
  - sign: '-1'
  - optimise:
```

### Configuration

This section of the file contains all technical details of the ODE solver and fitting details.
Find a annotated example below.

``` yaml
# ode solver related details:
integration_scheme: euler_forward
time_evo_max: 100
dt_time_evo: 0.1
ode_coeff_model: interaction_model_generator

# to check if the ode time evolution is stable and converges:
stability_rel_tolerance: 1e-3
tail_length_stability_check: 10
start_stability_check: 50

# fitting related details
sources: null
sinks: null
fit_model: direct_fit_model
fit_target: [1,1,0,0]
```

#### ode solver related configuration details:

* integration scheme: name of the function that calculates the next time integration step. It can either be chosen from [models.py/#integration_schemes](modules/models.py), or defined by the user.
* time_exo_max: Ideally the ode system reached a stable state and stops automatically. If this is not the case it stops the latest at the ode-model time defined here.
* dt_time_evo defines the step size in ode-model time
* ode_coeff_model: Every step in the time time integration can be represented by a matrix multiplication. ode_coeff_model defines how this matrix is constructed. Usually, you do not want to change this from the default. However, it is implemented in such a way that this function can be defined by the user if needed.

#### convergence check configurations:

* start_stability_check: time step in model time after which the convergence of the ode-model output is checked.
* stability_check_tail_length defines the number last ode-model outputs (hence, tail) to use to check the stability
* stability_rel_tolerance: to test the convergence, we define a range in which the model is allowed to fluctuate. This range is defined relatively to its value. Hence, a value of 1e-3 allows the ode-model to fluctuate by one permille  If this range is exceeded the model is (not yet) stable.

#### fitting related details:

* fit model: The ODE-model output can be of arbitrary shape. Depending on its application, there are many possible derived quantities you might want to fit your model to. One of the simplest examples could be that the model reaches a certain stable state. However, more complex derived complex quantities might also be of interest. I.e. for a ecosystem model it might be desirable that your model does have a net-zero energy flow. All this can be done with a suitable choice of the the fit_model function. A fit model can either be chosen from the [models.py/#fit_models](modules/models.py) functions or be by the user.
* fit_target defines that output value or the set of values that your model is fitted to. In the case that a zero net flux is desired would be set to zero.
* sources/sinks: marks certain compartments as sources or sinks, needed for a net-flux fit model.