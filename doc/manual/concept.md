# Conceptual overview of library

In the introduction we presented the concepts the user needs know when using 
the model. We will now take a look at the concept behind the user interface.

## ODE's and their time evolution

While the user provides a network to the framework, the underlying concept is 
the set of differential equations that is created from the network.
This differential equation is solved numerically to calculate the model output.
Therefore, we study model by examining the results of the integrated ODE's. 

We will now take a look a very reduced example.

We assume a first-order coupled ODE system of the following form:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;\partial_{t}&space;u^{i}&space;=&space;\alpha^{i}_{j}&space;u^{j}" title="\large \partial_{t} u^{i} = \alpha^{i}_{j} u^{j}" />
</p>

Where alpha takes the form of a matrix which elements are functions *f* of the 
form:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(\gamma_{1},\gamma_{2},..,\gamma_{m}):\mathbb{R}^{n}%20\otimes%20\mathbb{R}^{n}%20\otimes%20...\otimes%20\mathbb{R}^{n}%20\rightarrow%20%20\mathbb{R}" title="\large \partial_{t} u^{1} = u^{2}" />
</p>
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

In general, the parameters we might desire to be optimized are elements of u and beta (parameters of *f*).
In the following we will refer to them as free parameters *x*. As stated above they are

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x&space;\subset&space;(u_{0}&space;\cup&space;\beta)" title="\large x \subset (u_{0} \cup \beta)" />
</p>

All elements of u_{0} and beta are constant in the optimization process. 
Therefore, we will in the following only discuss the behavior of the time 
evolution  dependant on the free parameters.
Hence, we find a unique solution to the time evolution for every set of free parameters.
We write the time evolution base on a specific set of free parameters as F(x).
Because x is a subset of u and beta, this can also be expressed:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;F(x)&space;=&space;G(u_{0},\beta)" title="\large F(x) = G(u_{0},\beta)" />
</p>

### Cost/Loss function

To compare "how close" the system represents the desired system we need to 
introduce a measure. 
This measure is usually referred to as 'objective function', 
but often also called cost- or loss function.
For every set of free parameter *x* we can calculate a cost function *J*.
*J* is calculated based on *F(x)* and *F(y)*, where *y* represents the set of 
free parameters of the desired system.
In general *y* might not exist, i.e. if the network or the matrix alpha is 
"badly" chosen.
*J* is calculated by:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;J(x,y)&space;=&space;\frac{1}{N_{dim}}&space;\left\lbrace&space;F(x)-F(y)&space;\right\rbrace^{2}" title="\large J(x,y) = \frac{1}{N_{dim}} \left\lbrace F(x)-F(y))^{2} \right\rbrace" />
</p>

*F(y)* is given to the framework as a set of reference or measurement data 
points.

The effect of the different free parameters can be examined through a perturbation approach. The free parameters are perturbed slightly.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x_{perturbed}&space;=&space;x&space;&plus;&space;\rho&space;\;\;&space;\text{with}\;\;&space;\rho&space;\ll&space;x" title="\large x_{perturbed} = x + \rho \;\; \text{with}\;\; \rho \ll x" />
</p>
Then we recalculated the time evolution of the ODE system and calculate the new cost function.

Theoretically, we could continue to sample the parameter space of the free parameters in this way or by testing a homogeneous distribution of free parameters over the sample space.
However, this is computationally costly because it requires the evaluation of a large set of free parameters. Especially for a high dimensional search space, meaning a large set of observed quantities modelled by the ODE and/or a large set of free parameters, this becomes unfeasible.

Therefore we require a better sampling approach to reduce the computational cost of finding an optimal set of free parameters.

### Gradient Descent
The general idea is to "always walk downhill".
Imagine the cost function field as landscape.
The best solution is the one with the lowest value of *J*, 
hence the deepest valley.
Therefore, we have a chance to find it when we always walk downhill.
If we want to reach that point quickly, we might chose to take the steepest path
downhill. 
Be aware thought that going down the a certain downhill path might end up at a
different valley. Furthermore, there might be a deeper valley at the other side
of the mountain.

This concept is represented in its simplest form in the so called 
'gradient descent'.
Its core idea is that we calculate a local gradient of the unknown 
cost function field and always move along the direction of the steepest descent.
This way we find in principle the closest local minima without sampling the 
whole region but only one path. This drastically cuts the computational cost.

In its simplest form such a gradient descent is given by
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\large&space;x^{n&plus;1}&space;=&space;x^{n}&space;-&space;\lambda&space;\nabla&space;J(x^{n})" title="\large x^{n+1} = x^{n} - \lambda \nabla J(x^{n})" />
</p>

where x^{n+1} is the next tested free parameter set and lambda is the step size,
of the gradient of J.
The gradient always points into direction of the steepest ascent.
Hence, by going into the opposite direction we are always walking downhill on 
the steepest path.

This approach is also used in many machine learning problems and therefore well 
studied.
Its great advantage is that we only require the field of the cost function 
J(x,y) to be continuous and differentiable.
This enables this approach to be applicable in many different problems.
However, the generality of this approach introduces certain problems as well.
Firstly, it is generally unknown if an ideal solution exists.
Secondly, assuming such an ideal solution exists, it is not guaranteed that the 
algorithm finds it.
Hence, if the best output of the algorithm is still not satisfactory, one does 
not know if the studied system does not have a better solution or if the 
algorithm just didn't find it. 
We are blind walkers on a foggy mountains.
