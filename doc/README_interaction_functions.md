# Interaction Functions
A short description what interaction functions are, how they are used and how 
they are implemented.

## General 
Compartments in the model are linked through flows.
These flows, are characterized through, what we call, interaction functions.
Each interaction function precisely describe the change per time step between
two compartments.
[ref paper for details]

The user has to deicide what type of interaction suits best to describe an
interaction between to compartments.
A common examples of such types are 
[*holling type*](https://en.wikipedia.org/wiki/Functional_response)
functional responses.
A more simple example might be simple linear mortality, where a percentage of 
compartment dies and is transformed into some sort of detritus.


## Implementation
The framework offers a wide range of typical functional responses to be chosen 
as an interaction function.

### Conceptually 
Conceptually a interaction function *f* is of the form:
f(a,b,c, ... ) -> y
where *a,b,c* are the interaction parameters, such a natural mortality rate or
the amount of predators. *y* is a scalar representing the change per time step
in the "flow-destination" compartment.

The user has to decide on which function *f* they want to use to represent a 
certain interaction, as well as to provide the function dependent parameter 
*a,b,c ...*
This could be for example a linear mortality of the form  
f(alpha,A) = alpha * A
where alpha represents a mortality rate, expressed in units of (unit of A)/time.

### Technically
For technical reasons this is implemented slightly different.
Each interaction function also takes the a list of all compartments, including
the name or index of the origin and destination compartment.
Hence, the previous function signature ends up looking like this:

f(X,[ i_origin,j_destination ],a,b,c, ... ) -> y,

where X represents the array containing all compartments, and i and j represent
the corresponding indices of the origin and destination compartments.

## Units
In Ecology, or in Biology in general there are often many different units used 
to describe similar things.
Therefore, it can sometimes be difficult to gather all the necessary information 
in a consistent form to use them in interaction functions.

Particular attention is therefore required when creating the interaction 
functions, as any wrongly interpreted or transformed unit may render the model 
invalid.

In general, the unit of an interaction function, as for a compartment, are 
arbitrary.
It is up for the user to decide what kind of unit they may like to use to 
represent their model.
However, this design choice has to be consistent in the entire model.

As the interaction functions represent changes in model compartment over time, 
they need to be expressed as a rate of the same unit as the corresponding 
compartment that they are summed up with.

I.e.: Assume we wish to represent the flow between two compartments *A* and *B*.
Compartment B is described in units of gram. If we wish to describe a flow from 
compartment A to compartment B, this flow needs also to be expressed in units of 
gram as it is effectively summed up with the quantity in compartment B.

We strongly recommend, that when designing an interaction function, a sanity 
check is performed to verify that all units used add up to the necessary one.
I.e.:
In a simple example the flow from A to B might be represented by a linear function *f_AB* of the form:  
f_AB = alpha * A

Because f_AB is added to B for every time step dt, f_AB needs to take the form of a rate. If unit_B is the unit of the compartment B and the time steps are expressed in seconds than f_AB needs to add up to the unit unit_B/s. Hence, alpha needs to be in the unit 1/s.
Otherwise the interaction is invalid, and with it the resulting model.


In general compartments might have different units.
However, to keep the models simple and avoid further confusion in the choice of unit we designed the framework ins such a way that all compartments need to share a common unit.
An example of such a shared unit might be carbon mass in kg, in contrast to a wet weight of a certain species. 


## Renaming or providing user-defined interaction functions 

Even a simple model can contain many different interactions. 
However, often many of those interaction share the same underlying behavior.
I.e. a natural mortality might scale linearly with the total population just as
a exudation process might scale linearly as well.
For model simplicity we recommend using the same function to represent these 
interactions. Nevertheless, it might be helpful to distinguish the processes 
with different names. This is especially helpful when drawing a larger model as
 a network, as it helps to identify the actual processes one wants to describe.

For this reason, the model allows to rename existing interaction functions 
easily by the user.
This can be achieved by two different ways:

* The first option is to define a list of the alternative names in the yaml
  configuration file. An example of this might look like:
  
  ``` yaml
  configuration:
    [...]
    alternative_interaction_names:
      'alternative_name_one': 'existing_function'
      'alternative_name_two': 'existing_function'
  ```

* The second option is to define them in the python script that is executed to run
  the model.
  In that case the newly written or renamed functions
  need to be past to the function *import_interaction_functions([func1,func2,...])*
  
  To give an example:
  * In the case of renaming a existing function
    ``` python
    import gemf
    alternative_name_one = existing_function
    alternative_name_two = existing_function
    
    gemf.model.import_interaction_functions([alternative_name_one,alternative_name_two])
    ```

  * When a new interaction function is written, it has to use the same signature
    as the existing ones. See [interaction functions code documentation](https://general-ecosystem-modeling-framework.readthedocs.io/en/latest/gemf.html#module-gemf.interaction_functions)
    
    Currently something like this would be expected:
    ```python
    def new_function(X, ii, jj, args):
        [...] # calculates the changes per time step
        return df
    
    import gemf
    gemf.model.import_interaction_functions([new_function])
