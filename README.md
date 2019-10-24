# Inverse Modeling Library

This is a python implementation of a inverse modeling tool for first-order ODE's (ordinary differential equations).
It was created to assist in development of an carbon flux model in the Baltic sea by Maike Scheffold, but is designed to work for all n-dimensional first-order ODE's.

## Quick Start

See examples.py

## Usage

The function intended for the user are gathered in the caller module.

The main top level function is **'dn_monte_carlo()'**. It requires the path to a set of initial values. These are the initial state of the ODE system, the coupling constants of the ODE system and the desired output of the model regarding a certain 'fit model'. It returns a set of optimized free parameter which have been randomly chosen. Hence, the name 'monte carlo'. In addition it return the value of the cost function field at the point which is defined by the set of optimized free parameters. Idealy one can then find a good enough set of free parameters in this list.

**dn_monte_carlo()**  calls the the **'gradient_descent()'** function which does the actual optimisation and is therefore responsible for most of the heavy lifting in this library.
It returns for a given set of initial free parameters (among many other input parameters) an optimized set based on the gradient descent approach starting from the initial set.

The last top level funtion is the **'run_time_evo()'** on which the gradient_descent() routine heavily relies on. It does the actual time integration of the system of ODEs and by doing so tests the influence of the free parameter.



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

### ODE's and their time evolution
To keep it as general as possible we do not assume anything about the dependency of the model on the optimized input parameters.
Therefore, we are required to study their effect on the model by examining the results of the integrated ODE's. 

We assume a first-order coupled ODE system of the following form:
$$ \partial_{t} u^{i} = \alpha^{i}_{j} u^{j}$$
Where alpha is the set of real valued coefficients.
A simple example would be something of the form:
$$ \partial_{t} u^{1} = u^{2} \\
   \partial_{t} u^{2} = u^{1}
$$
Here, alpha takes the form:
$$ 
\alpha^{i}_{j} = \left\lbrace
    \begin{matrix}
        0 & 1 \\
        1 & 0
    \end{matrix}
    \right\rbrace
$$
We require that the desired state of the ODE system is a steady state of the system. From an optimasation point of view,we usually also desire a certain steady state of the system.
This gives us the following two constrains of stability and suitability:


  $$ \text{Stability} \; \;   \exists \; i : j>i \; || G_{j}(u_{0},\alpha) - G_{i}(u_{0},\alpha) || < \epsilon_{1} $$
$$ \text{Suitability} \; \; \exists  \; x : || G(u_{0},\alpha) - y || < \epsilon_{2} $$

Where u_{0} is the vector containing the quanteties observed in the ODE in their initial state (t=0). alpha is a matrix containg the coupling coefficients of them.
G(u,alpha) is the map of time evolution of the system defined by u, alpha.
The subscripts (i and j) represent the iteration step of the time evolution and both epsilon are small (compared to F(x)) positive real valued objects.
The desired state of the system is represented by the vector y.
The subscripts (i and j) represent the iteration step of the time evolution and both epsilon are two small (compared to G) positive real valued objects.

In general, the parameters we might desire to be optimised are elements of u and alpha. In the following we will refer to them as free parameters (x). As stated above they are
$$ x \subset (u_{0} \cup \alpha) $$

All elements of u_{0} and alpha are constant in the optimization process. Therefore, we will in the following only discuss the behaviour of the time evolution dependant on the free parameters F(x). F(x) is a different representation of the same map. Hence,
$$ F(x) = G(u_{0},\alpha). $$


As implied by the suitability condition, we can define a measure for how close the system represents the desired system. Keep in mind that we only consider steady state outputs of the integrated ODE F(x). 
This measure is usually referred to as 'objective function', but often also called cost- or loss function.
Hence, for every set of free parameter (x) we can calculate a cost function J(x). The standard objective function is of the form
$$ J(x,y) = \frac{1}{N_{dim}} \left\lbrace F(x)-y)^{2} \right\rbrace$$

The effect of the different free parameters can be examined through a perturbation approach. The free parameters are perturbed slightly.
$$ x_{perturbed} = x + \rho \;\; \text{with}\;\;  \rho \ll x $$
Then we recalculated the time evolution of the ODE system and calculate the new cost function.

Theoretically, we could continue to sample the parameter space of the free parameters in this way or by testing a homogeneous distribution of free parameters over the sample space.
However, this is computationally costly because it requires the evaluation of a large set of free parameters. Especially for a high dimensional search space, meaning a large set of observed quantities modelled by the ODE and/or a large set of free parameters, this becomes unfeasible.

Therefore we require a better sampling approach to reduce the computational cost of finding an optimal set of free parameters.

### Gradient Descent
The approach chosen in this library is called 'Gradient Descent'.
Its core idea is that we calculate a local gradient of the unknown cost function field and always move along the direction of the steepest descent. This way we find in principle the closest local minima without sampling the whole region but only one path. This drastically cuts the computational cost.

In its simplest form such a gradient descent is given by
$$ x^{n+1} = x^{n} - \lambda \nabla J(x^{n})  $$
where x^{n+1} is the next tested free parameter set and lambda is the step size, sometimes also called *learning rate*, of the gradient of J. The gradient always points into direction of the steepest ascent. Hence, by going into the opposite direction we are always walking downhill on the steepest path.

This approach is also used in many machine learning problems and therefore well studied.
Its great advantage is that we only require the field of the cost function J(x,y) to be continuous and differentiable.
This enables this approach to be applicable in many different problems.
However, the generality of this approach introduces certain problems as well.
Firstly, it is generally unknown if an ideal solution exists. Secondly, assuming such an ideal solution exists, it is not guaranteed that the algorithm finds it.
Hence, if the best output of the algorithm is still not satisfactory, one does not know if the studied system does not have a better solution or if some parameter of the algorithm was badly chosen.
