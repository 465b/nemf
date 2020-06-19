# Releases

## v0.3.2 (June 2020)

This is a major release introducing the new name "nemf" of the framework as 
well as a improved reference data import and some changes in the plotting of 
the framework output.

### New name 

  The framework has been renamed **G**eneral **E**cosystem **M**odelling 
  **F**ramework (GEMF) to **N**etwork-based **E**cosystem 
  **M**odelling **F**ramework (NEMF).

  While it still remains a *general* modelling framework, the name change shall
  emphasis the core distinguishing concept that distinguishes it from other 
  other modelling frameworks.
  Additionally, the network concept is key concept that users need to work with
  and keep in mind while using the framework, while the *general*-aspekt fades
  into the background once the framework has been chosen.

### Better reference data input

  * Previously reference data had to be provided for all modelled compartments.
    It is now possible to provide reference data for only a subset of the 
    compartment.

    * Datasets must now contain a header that contains the names of the columns
      which will be compared to the compartment names.
    
    * Data that is present in the reference data set which does not correspond 
      to modelled compartment will now be simply ignored and does not need be 
      removed manually. However, this also requires the header names to match 
      the compartment names exactly.
    
  * New feature of "Datetime" timestamps for reference data in excel files.
    "Datetimes" (i.e. '18.09.1783 12:34') are automatically transformed into 
    the default POSIX timestamp (seconds before/after 1.1.1970).

  
### Plotting changes

  * *Major*: Output summary plots now automatically use scatter plots instead 
    of line plots if the amount of data points is low. 
    This has two main reasons:

    * Plots with only one point are now also proprly drawn.
      (Lines require two points, and are drawn empty if only one is present)

    * Avoids implying a linear behavior between two distant points, which can 
      confuse the user while interpreting the plot.

  * *Minor*: Implemented a new color selection scheme when plotting. This 
    avoids that colors are reused when many lines are present. Additionally, it 
    makes sure that the reference data and model output use the same colors,
    even if only a subset of the model compartments is present in the refer
    data.


### Other changes

  * The interaction function *nutrition_limited_growth* has been renamed to 
    *nutrient_limited_growth*.
  
  * *load_model* now also allows the import of reference data.
  
  * Many new descriptions and comments have been added to the documentation



## v0.3.1 (June 2020)

This is a minor release introducing a user-friendly option to rename interaction
functions as well as constraints for optimized parameters.
An initial documentation has been drafted and is hosted on 
[readthedocs.io](https://nemf.readthedocs.io/en/latest/).

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