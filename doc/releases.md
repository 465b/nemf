# Releases


## v0.3.1 (June 2020)

This is a minor release introducing a user-friendly option to rename interaction
functions as well as constraints for optimized parameters.
An initial documentation has been drafted and is hosted on [readthedocs.io](https://nemf.readthedocs.io/en/latest/).

### Importing and renaming interaction functions

* *init_alternative_interaction_names( )*  
  This is called when ever a new model i initialized.
  It checks if alternative names for an existing interaction were declared in the 
  yaml configuration file.
  For details check the
  [docs](https://nemf.readthedocs.io/en/latest/README_interaction_functions.html).

* *import_interaction_functions( )*  
  This can be called in the execution file by the user to import user-defined or
  renamed interaction functions.
  For details check the
  [docs](https://nemf.readthedocs.io/en/latest/README_interaction_functions.html).

The previous version of renaming interaction functions was to declare the 
alternative names in the execution file. However, this did not work consistently
in all execution environments. Therefore, we now introduce to dedicated 
functions to rename and/or import interaction functions.


### Importing constraints

* Added *import_constraints( )*  
  Reads constraints from an python file and adds them to the model.
  For details check the
  [docs](https://nemf.readthedocs.io/en/latest/README_interaction_functions.html).

In some circumstances, the parameter fitted in an optimization run are not
independent of each other. I.e. one might want to enforce that sum of two 
parameters is always equal to one. Such a behavior is enforced through 
'constraints'.


### Documentation

* Hosted the current drafts of the documentation on [readthedocs.io](https://nemf.readthedocs.io/en/latest/).
* Documentation created via sphinx
* Added automated API reference creation based on docstrings via autodoc



## v0.3.0 (May 2020)

This is a major release introducing scipy's integration and optimization 
routines into the framework.

### Replaced reference data import

* Removed the old reference data input which only allowed for simple 
  steady-state fitting
* New version requires the data to be stored in a separate file and imported
  either by passing the link in the yaml file or when calling the inverse_method 
  routine. The latter overwrites the path given in the yaml file.

### New forward modelling

* Replaced the self written *time_integration* routine with 
  *scipy.integrate.solve_ivp*.
* Adjusted *forward_model* and plotting routines accordingly and introduced 
  supporting functions.

### New inverse modelling

* Replaced the self written *gradient_descent* method with 
  *scipy.optimize.minimize*.
* Adjusted *inverse_model* and plotting routines accordingly and introduced 
  supporting functions.

### Visualization

* Added *initial_guess* plotting function to visualize the model output before 
  any optimization has been applied.