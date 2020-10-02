# Releases

## v0.3.4 ( Oktober 2020)

This is a minor release introducing the option to enforce strictly positive 
solutions besides several minor changes.

### Strictly positive ODE solutions

Depending on the model you are trying to develop the differential equations
might be [stiff](https://en.wikipedia.org/wiki/Stiff_equation). 
As a result of that it can happen that the solution to the differential 
equations diverge from the exact solution. 
Most notably this happens if a compartment value reaches a negative value for 
a compartment where this should not be possible (i.e. negative populations).
Because the differential equations are no longer well defined in such a case
the solutions become nonphysical.
To avoid this behaviour we implemented interactions functions 
(i.e. "non_negative_holling_type_II") that can be used if stiffness becomes a 
problem.

### Others

* Radau is now the default solver for the ODE solver implemented in 
  **forward_model()**
* Fixed bug in **forward_model()** where key-word arguments where not parsed
  correctly
* Minor bug fixed and docstring changes



## v0.3.3 (June 2020)

This is a minor release introducing new options for both non-linear-programming
aka fitting solvers and initial-value-problem aka forecasting solvers. It also
brings some bugfixes and consistency changes as well as additional 
documentation.

### New Non-Linear-Programming (NLP) solvers

  Previously, only the two solvers that worked in all cases (bounded, 
  constraint & bounded) were properly implemented.

  However, both 'SLSQP' and 'trust-constr' showed to be sometimes outperformed 
  by other solvers. 'trust-constr' showed to be much slower, and 'SLSQP' while 
  fast sometimes exited without actually optimizing the result.
  For that reason we added additional solves which are available in the case 
  where no constraints are present.
  
  The full list of solvers is now:  
    * ‘SLSQP’
    * ‘trust-constr’
    * **'L-BFGS-B'**
    * **'TNC'**
    * **'Powell'**
  where the bold ones are newly implemented.
  To distinguish the NLP and IVP solvers, the key used to pass them to the 
  inverse_model has been changed from 'method' to 'nlp_method'.
  For more detail, see the [API references](api)

### New Initial-Value-Problem (IVP) solvers

  Previously, the IVP solver was not free for the user to change and always 
  defaulted to de 'RK45' solver.
  However, the 'RK45' showed to have some issues.
  In some cases it got stuck, mostly because the numerical errors resulted in 
  negative compartment values even if the ODE would not allow it.
  Hence, the error increased over time drastically and the output,
  if calculated in a reasonable time were unusable.

  Other solvers showed to performed better in these circumstances.
  The default solver is now 'Radau'. While it is generally slower then the 
  'RK45' it showed to perform better in the above described circumstances.

  The full list of IVP solvers is now:
    * 'Radau' (default)
		* 'RK45'
    * 'RK23'
    * 'DOP853'
    * 'BDF'
    * 'LSODA'
  To distinguish the NLP and IVP solvers, the key used to pass them is called
  'ivp_method'.
  For more detail, see the [API references](api)

### Bug fixed and consistency changes

#### Changes

* Different verbosity options can now be used for 'trust-constr', providing 
  more information during runtime. See [API references](api) for more 
  details.

* Changed the output of the *load_reference_data()* function. Previously it 
  returned the output directly. Hence, the user had to pass it back to the 
  model. Because there is (currently) no application to use it without passing 
  it back to the model it does this now automatically.

#### Bugs

* *load_model()* did not allow to not pass a reference data path as presented 
  in the examples. This is now possible.

* *load_reference_data()* did not correctly accept keyword arguments but only
  non-keyword ones. This as an error as all optional ones are kwargs.
  
#### Minor internal changes

* *internal:* In some cases the reference data sets have been referred to as 
  'fit data' because it has been at some point during development used 
  exclusively for fitting. Because is does not need to be used to fit anymore 
  but can also be used to plot, we now call it reference data everywhere.

* *internal*: The data type used to describe the bounds of parameters during the 
  optimization process was inconsistent. The compartments used list while 
  the interactions used sets. This was not also inconsistent but also caused 
  issues as some solvers did not parse them correctly.
  Now, all bounds are passed as lists.


## v0.3.2 (June 2020)

This is a minor release introducing the new name "nemf" of the framework as 
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
      confuse the user while interpret
      ing the plot.

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
  [docs](https://nemf.readthedocs.io/en/latest/api.html#interaction-functions).

* *import_interaction_functions( )*  
  This can be called in the execution file by the user to import user-defined or
  renamed interaction functions.
  For details check the
  [docs](https://nemf.readthedocs.io/en/latest/api.html#interaction-functions).

The previous version of renaming interaction functions was to declare the 
alternative names in the execution file. However, this did not work consistently
in all execution environments. Therefore, we now introduce to dedicated 
functions to rename and/or import interaction functions.


### Importing constraints

* Added *import_constraints( )*  
  Reads constraints from an python file and adds them to the model.
  For details check the
  [docs](https://nemf.readthedocs.io/en/latest/api.html#interaction-functionsl).

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