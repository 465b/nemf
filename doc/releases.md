# Releases

## v0.3.0 (May 2020)

This is a major release introducing scipy's integration and optimization routines into the framework.

### Replaced reference data import

* Removed the old reference data input which only allowed for simple 
  steady-state fitting
* New version requires the data to be stored in a seperate file and imported
  either by passing the link in the yaml file or when callin the inverse_method 
  routine. The latter overwrites the path given in the yaml file.

### New forward modelling

* Replaced the self written *time_integration* routine with *scipy.integrate.solve_ivp*.
* Adjusted *forward_model* and plotting routines accordingly and introduced supporting functions.

### New inverse modelling

* Replaced the self written *gradient_descent* method with *scipy.optimize.minimize*.
* Adjusted *inverse_model* and plotting routines accordingly and introduced supporting functions.

### Visualization

* Added *initial_guess* plotting function to visualize the model output before any optimization has been applied.