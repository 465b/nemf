# Inverse Modeling Library

This is a python implementation of a inverse modeling tool for ODE's (ordinary differential equations).
It was created to assist in development of an carbon flux model in the Baltic sea by Maike Scheffold, but is designed to work for all n-dimensional ODE's.

## Quick Start

See examples.py

## Usage

The function intended for the user are gathered in the caller module.

The main top level function is **'dn_monte_carlo()'**.
It requires the path to the model configuration.
The configuration file is a standardized yaml file, containing the configuration of the ODE system as well as several technical parameters like time integration step size.
If fitting of the model is desired, the 'fit model' and its desired output is also defined in the configuration file.

For more details on the yaml file, see [README_YAML.md](README_YAML.md)

After **'dn_monte_carlo()'** ran, it returns a set of the optimized free parameter which can be randomly chosen (hence,  the name monte carlo), or prescribed.
Additionally, it returns the value of the cost function, the prediction of the fit model, as well as the full ode-model output for the final parameter set as well as all intermediate steps.

**dn_monte_carlo()**  calls the the **'gradient_descent()'** function which does the actual optimization and is therefore responsible for most of the heavy lifting concerning the fitting in this library.
For a given set of initial free parameters (among many other input parameters), an optimized set is returned, based on the gradient descent approach starting from the initial set.

The last top level function is the **'run_time_evo()'** on which the gradient_descent() routine heavily relies on. It does the time integration of the system of ODEs and by doing so tests the influence of the free parameter.


## What is this library for?

Assume we have a set of coupled ordinary differential equations.
Some of parameters of this set of equations are not yet determined and need to be found or optimized i.e. due to measurement uncertainties.
These parameters might be i.e. initial values that reproduce a certain steady state.

This library provides a framework where a set of parameters, both initial conditions as well as ODE coefficients, can be chosen and if possible adjusted in such a way that your model reaches a certain desired stable state.


To do so, we assume the following:
* It is a system of first-order-coupled ordinary differential equation.
* The desired state is a stable state of the ODE system. 
* The modeled quantity of the desired system is conserved
    
## How does it work?

For a conceptual description of the internals of this library, see [README_concept.md](README_concept.md)

