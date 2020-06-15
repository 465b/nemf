# Conceptual overview of library

## ODE's and their time evolution
To keep it as general as possible we do not assume anything about the dependency of the model on the optimized input parameters.
Therefore, we are required to study their effect on the model by examining the results of the integrated ODE's. 

We assume a first-order coupled ODE system of the following form:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\partial_{t}&space;u^{i}&space;=&space;\alpha^{i}_{j}&space;u^{j}" title="\large \partial_{t} u^{i} = \alpha^{i}_{j} u^{j}" />
</p>

Where alpha is the set of real valued coefficients.
A simple example would be something of the form:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\partial_{t}&space;u^{1}&space;=&space;u^{2}" title="\large \partial_{t} u^{1} = u^{2}" />
</p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\large&space;\partial_{t}&space;u^{2}&space;=&space;u^{1}" title="\large \partial_{t} u^{2} = u^{1}" />
</p>

Here, alpha takes the form:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\alpha^{i}_{j}&space;=&space;\left\lbrace&space;\begin{matrix}&space;0&space;&&space;1&space;\\&space;1&space;&&space;0&space;\end{matrix}&space;\right\rbrace" title="\large \alpha^{i}_{j} = \left\lbrace \begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix} \right\rbrace" />
</p>

## Fitting

### Stability and Suitability

We require that the desired state of the ODE system is a steady state of the system. From an optimization point of view,we usually also desire a certain steady state of the system.
This gives us the following two constrains of stability and suitability:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\text{Stability}&space;\;&space;\;&space;\exists&space;\;&space;i&space;:&space;j>i&space;\;&space;||&space;G_{j}(u_{0},\alpha)&space;-&space;G_{i}(u_{0},\alpha)&space;||&space;<&space;\epsilon_{1}" title="\large \text{Stability} \; \; \exists \; i : j>i \; || G_{j}(u_{0},\alpha) - G_{i}(u_{0},\alpha) || < \epsilon_{1}" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\text{Suitability}&space;\;&space;\;&space;\exists&space;\;&space;x&space;:&space;||&space;G(u_{0},\alpha)&space;-&space;y&space;||&space;<&space;\epsilon_{2}" title="\large \text{Suitability} \; \; \exists \; x : || G(u_{0},\alpha) - y || < \epsilon_{2}" />
</p>

Where u_{0} is the vector containing the quanteties observed in the ODE in their initial state (t=0). alpha is a matrix containg the coupling coefficients of them.
G(u,alpha) is the map of time evolution of the system defined by u, alpha.
The subscripts (i and j) represent the iteration step of the time evolution and both epsilon are small (compared to F(x)) positive real valued objects.
The desired state of the system is represented by the vector y.
The subscripts (i and j) represent the iteration step of the time evolution and both epsilon are two small (compared to G) positive real valued objects.

In general, the parameters we might desire to be optimised are elements of u and alpha. In the following we will refer to them as free parameters (x). As stated above they are

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x&space;\subset&space;(u_{0}&space;\cup&space;\alpha)" title="\large x \subset (u_{0} \cup \alpha)" />
</p>

All elements of u_{0} and alpha are constant in the optimization process. Therefore, we will in the following only discuss the behaviour of the time evolution dependant on the free parameters F(x). F(x) is a different representation of the same map. Hence,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;F(x)&space;=&space;G(u_{0},\alpha)" title="\large F(x) = G(u_{0},\alpha)" />
</p>

### Cost/Loss function

As implied by the suitability condition, we can define a measure for how close the system represents the desired system. Keep in mind that we only consider steady state outputs of the integrated ODE F(x). 
This measure is usually referred to as 'objective function', but often also called cost- or loss function.
Hence, for every set of free parameter (x) we can calculate a cost function J(x). The standard objective function is of the form

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;J(x,y)&space;=&space;\frac{1}{N_{dim}}&space;\left\lbrace&space;F(x)-y)^{2}&space;\right\rbrace" title="\large J(x,y) = \frac{1}{N_{dim}} \left\lbrace F(x)-y)^{2} \right\rbrace" />
</p>

The effect of the different free parameters can be examined through a perturbation approach. The free parameters are perturbed slightly.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x_{perturbed}&space;=&space;x&space;&plus;&space;\rho&space;\;\;&space;\text{with}\;\;&space;\rho&space;\ll&space;x" title="\large x_{perturbed} = x + \rho \;\; \text{with}\;\; \rho \ll x" />
</p>
Then we recalculated the time evolution of the ODE system and calculate the new cost function.

Theoretically, we could continue to sample the parameter space of the free parameters in this way or by testing a homogeneous distribution of free parameters over the sample space.
However, this is computationally costly because it requires the evaluation of a large set of free parameters. Especially for a high dimensional search space, meaning a large set of observed quantities modelled by the ODE and/or a large set of free parameters, this becomes unfeasible.

Therefore we require a better sampling approach to reduce the computational cost of finding an optimal set of free parameters.

### Gradient Descent
The approach chosen in this library is called 'Gradient Descent'.
Its core idea is that we calculate a local gradient of the unknown cost function field and always move along the direction of the steepest descent. This way we find in principle the closest local minima without sampling the whole region but only one path. This drastically cuts the computational cost.

In its simplest form such a gradient descent is given by
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x^{n&plus;1}&space;=&space;x^{n}&space;-&space;\lambda&space;\nabla&space;J(x^{n})" title="\large x^{n+1} = x^{n} - \lambda \nabla J(x^{n})" />
</p>

where x^{n+1} is the next tested free parameter set and lambda is the step size, sometimes also called *learning rate*, of the gradient of J. The gradient always points into direction of the steepest ascent. Hence, by going into the opposite direction we are always walking downhill on the steepest path.

This approach is also used in many machine learning problems and therefore well studied.
Its great advantage is that we only require the field of the cost function J(x,y) to be continuous and differentiable.
This enables this approach to be applicable in many different problems.
However, the generality of this approach introduces certain problems as well.
Firstly, it is generally unknown if an ideal solution exists. Secondly, assuming such an ideal solution exists, it is not guaranteed that the algorithm finds it.
Hence, if the best output of the algorithm is still not satisfactory, one does not know if the studied system does not have a better solution or if some parameter of the algorithm was badly chosen.
