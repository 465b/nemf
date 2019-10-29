import numpy as np
import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)


# Gradiant Decent

## Cost function related Methods
def cost_function(prediction,y):
    """ normalized squared distance between model prediction and
        desired model behavior 
        
        Parameters
        ----------
        prediction : numpy.array
            1D-array contains the output of the fit model which
            ran the time integration
        y : numpy.array
            1D-array,same length as prediction containing the desired
            model output.
            
        Returns
        -------
        cost : float
            contains the value of the cost function for the given prediction
            and desired state"""


    cost = (1/len(y))*np.sum( (prediction - y)**2 )
    return cost


def barrier_function(free_param,constrains=np.array([None]),barrier_slope=1):
    """ Constructs the additional cost that is added closer to constrains.
        This is helpfull to keep the optimization process from trying to 
        exceed the search space. However, also distorts the optimal output
        slightly.

        Parameters
        ----------
        free_param : numpy.array
            1D-array of length n containing the current set of optimized free
            paratmers
        constrains : numpy.array
            2D-array of shape (n,2) containing the constraints limits
        barrier_slope : positive-float
            Defines the slope of the barrier used for the soft constrain.

        Returns
        -------
        cost_barrier : float
            additional cost created from beeing close to a constraints
            barrier. """

    # checks if contains are defined. else sets them to +/- inf
    if (constrains == None).all():
        constrains = np.zeros((len(free_param),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    # checks if constrains has the correct length
    if len(constrains) != len(free_param):
        raise ValueError('List of constrains must have'+
                         'the same length as free_param')

    # checks if barrier_slope is positive
    if barrier_slope <= 0:
        raise ValueError('barrier_slope must be a postive value.')

    # initializes arrays containing the additional cost
    cost_barrier_left = np.zeros(len(free_param))
    cost_barrier_right = np.zeros(len(free_param))
    cost_barrier = np.zeros(len(free_param))
    
    # calculates the additional cost
    """ note that we always apply the cost of both sides.
        We assume that if one of them is non-negligible,
        the other one is """
    for ii,[left,right] in enumerate(constrains):
        #left constrain
        if (left == -np.inf):
            cost_barrier_left[ii] = 0
        else:
            cost_barrier_left[ii] = -np.log(free_param[ii]-left)

        # right constrain
        if (right == np.inf):
            cost_barrier_right[ii] = 0
        else:
            cost_barrier_right[ii] = -np.log(-free_param[ii]+right)

    cost_barrier = cost_barrier_left + cost_barrier_right
    cost_barrier = np.sum(cost_barrier)*barrier_slope
    
    return cost_barrier


def barrier_hard_enforcement(free_param,jj,constrains=None,
                             pert_scale=1e-2,seed=137):
    """ if outside the search space we enforce a 'in-constrain'
        search space by ignoring the recommended step and
        moving it back into the search space

    Parameters
    ----------
    free_param : numpy.array
        1D-array containing the set of optimized free parameter.
    jj : positive integer
        Index of the iteration step in which the function is called.
        This is logged for debugging purposes.
    constrains : numpy.array
        2D-array containing the upper and lower limit of every free input
        parameter in the shape (len(free_param),2)
    pert_scale : positive float
        Maximal value which the system can be perturbed if necessary
        (i.e. if instability is found). Actual perturbation ranges
        from [0-pert_scal) uniformly distributed.
    seed : positive integer
        Initializes the random number generator. Used to recreate the
        same set of pseudo-random numbers. Helpfull when debugging.
    
    Returns
    -------
    free_param : numpy.array
        1D-array containing the set of optimized free parameter.
        Now, guaranteed to be inside of the search space.
    """

    # initializes constrains from plus to minus inf if not provied
    if (constrains == None).all():
        constrains = np.zeros((len(free_param),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    # tests if constrains is provided for all free_parameters
    if len(constrains) != len(free_param):
        raise ValueError('List of constrains must have'+
                         'the same length as free_param')

    # applies the actual enforcement
    for ii,[left,right] in enumerate(constrains):
        """ we add a small perturbation to avoid 
            that the we remain on the boarder """
        if (free_param[ii] <= left):
            buffer = left + np.random.rand()*pert_scale
            warn_string = ( 'Left  barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(
                                free_param[ii],buffer))
            logging.debug(warn_string)
            free_param[ii] = buffer

        if (free_param[ii] >= right):
            buffer = right - np.random.rand()*pert_scale
            warn_string = ( 'Right barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(
                                free_param[ii],buffer))
            logging.debug(warn_string)
            free_param[ii] = buffer
            
    return free_param


# Time Evolution

## Stability

def verify_stability_time_evolution(prediction, stability_rel_tolerance=1e-6,
                                    tail_length=10):
    """ checks if the current solution is stable by 
        comparing the relative fluctuations in the 
        last tail_length model outputs to a stability_rel_tolerance value
        returns true if all values are below that threshold.
        
    Parameters
    ----------
    prediction : numpy.array
        1D-array containing the output of the time evolution for every time
        integration step.
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative fluctuation range in the tail
        of the time evolution. If below, system is called stable.
    tail_length_stability_check : positive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    
    Returns
    -------
    is_stable : bool
        true if stability conditions are met
    """

    
    prediction_tail = prediction[-tail_length-1:-1]

    average = np.average(prediction_tail,axis=0)
    spread = np.max(prediction_tail,axis=0) -np.min(prediction_tail,axis=0)
    rel_spread = spread/average
    is_stable = (rel_spread <= stability_rel_tolerance).all()
    
    return is_stable


# Helper Functions

## General Helper Function
def filter_free_param(ODE_state=None,ODE_coeff=None,
                      ODE_state_indexes=None,ODE_coeff_indexes=None):
    """ Takes the initial state and coefficient arrays and returns the values
        selected by the index-objects. Returns an array filled with the values
        of the entries selected for optimization.

    Parameters
    ----------
    ODE_state : numpy.array
        1D array containing the initial state of the obersrved quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_state_indexes : numpy.array
        1D-array containing sets of indices used to select which elements
        of the ODE_state array are optimized. 'None' if none are optimized.
    ODE_coeff_indexes : numpy.array
        1D-array containing sets of indices used to select which elements  
        of the ODE_coeff array are optimized. 'None' if none are optimized.
    Returns:
    --------
    free_param : numpy.array
        1D-array containing the set of to-be-optimized free parameters.
    """
    

    if ( (ODE_state is None) or (ODE_state_indexes is None) ):
        free_param_state = np.array([])
    else:
        free_param_state = ODE_state[ODE_state_indexes]
    
    if ( (ODE_coeff is None) or (ODE_coeff_indexes is None) ):
        free_param_coeff = np.array([])
    else:
        free_param_coeff = ODE_coeff[ODE_coeff_indexes]

    free_param = np.append(free_param_state,free_param_coeff)
    
    return free_param


def fill_free_param(free_param,ODE_state,ODE_coeff,
                    ODE_state_indexes,ODE_coeff_indexes):
    """ Takes the free_parameters and filles them back into their origin which
        are determined by the {state,coeff} and their corresponding index
        objects. Overwrites the original values in the provess

    Parameters
    ----------
    free_param : numpy.array
        1D-array containing the set of to-be-optimized free parameters.
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_state_indexes : numpy.array
        1D-array containing sets of indices used to select which elements
        of the ODE_state array are optimized. 'None' if none are optimized.
    ODE_coeff_indexes : numpy.array
        1D-array containing sets of indices used to select which elements  
        of the ODE_coeff array are optimized. 'None' if none are optimized.
    Returns:
    --------
    ODE_state : numpy.array
        1D array containing the state of the observed quantities
        in the ODE. Now filled with potentially optimized values.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE.
        Now filled with potentially optimized values.
    """

    if ODE_state_indexes is None:
        n_ODE_state = 0
    else:
        n_ODE_state = len(ODE_state_indexes)
        ODE_state[ODE_state_indexes] = free_param[:n_ODE_state]
    
    if ODE_coeff_indexes is None:
        pass
    else:
        ODE_coeff[ODE_coeff_indexes] = free_param[n_ODE_state:]

    return ODE_state,ODE_coeff


def division_scalar_vector_w_zeros(a,b):
    """ Divides an an array (a/b) while catching divisions by zero
        and sets them to zero """
    free_param = np.divide(a, b, out=np.zeros_like(b), where=b!=0)
    
    return free_param


def prediction_and_costfunction(free_param, ODE_state, ODE_coeff,
            ODE_coeff_model,y,fit_model, integration_scheme, time_evo_max,
            dt_time_evo, constrains=np.array([None]),barrier_slope=1,
            stability_rel_tolerance=1e-5,tail_length_stability_check=10,
            start_stability_check=100):
    """ calculates the (stable) solution of the time evolution (prediction)
        and the resulting costfunction (cost) while applying a
        barrier function that penalizes points close to the barrier 
        
        Parameters
    ----------

    ODE_state : numpy.array
        1D array containing the initial state of the observed quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    y : numpy array
        1D-array containing the desired output of the model in the form
        defined by the fit-model
    fit_model : function
        {net_flux_fit_model, direct_fit_model}
        defines how the output of the time evolution get accounted for.
        i.e. the sum of the output is returned or all its elements
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    time_evo_max
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    constrains : numpy.array
        2D-array containing the upper and lower limit of every free input
        parameter in the shape (len(free_param),2).
    barrier_slope : positive-float
        Defines the slope of the barrier used for the soft constrain.
        Lower numbers, steeper slope. Typically between (0-1].
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative fluctuation range in the tail
        of the time evolution. If below, system is called stable.
    tail_length_stability_check : positive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned

    Returns
    -------
    prediction : numpy.array
        2D-array containing the output of the time evolution
        after the fit_model has been applied, stacked along the first axis.
    cost
        1D-array containing the corresponding values of the cost
        function calculated based on the free_param at the same
        first-axis index.
    is_stable : bool
        true if stability conditions are met. 
        See verify_stability_time_evolution() for more details.
    """

    prediction,is_stable = fit_model(integration_scheme, time_evo_max,
                                     dt_time_evo, ODE_state, ODE_coeff,
                                     ODE_coeff_model,stability_rel_tolerance,
                                     tail_length_stability_check,
                                     start_stability_check)

    cost = cost_function(prediction,y)
    cost += barrier_function(free_param,constrains,barrier_slope)
    
    return prediction,cost,is_stable


## PDE-weights related Helper Functions
def normalize_columns(A):
    """ Enforces that all columns of A are normalized to 1.
        Otherwise an carbon mass transport amplification would occur.
        Does so by perserving the ratios between the values in a single column. 
    
    Parameters:
    -----------
    A : numpy.array
        can be any square-matrix containing scalars.
    
    Returns:
    --------
    A : numpy.array
        normalized version of the input """
    
    overshoot = np.sum(A,axis=0)    
    for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
        if truth_val : 
            diag_val = A[ii,ii]
            if diag_val == overshoot[ii]:
                # only the diagnoal entry is filled
                # hence, its the dump.
                pass
            elif (diag_val == 0): 
                overshoot[ii] -= 1
                if (abs(overshoot[ii]) > 1e-16):
                    A[:,ii] -= A[:,ii]/(overshoot[ii]+1)*overshoot[ii]
                    A[ii,ii] = diag_val
            else:
                A[:,ii] -= A[:,ii]/(overshoot[ii]-diag_val)*overshoot[ii]
                A[ii,ii] = diag_val

    return A


## Gradient Decent related Helper Functions
def init_variable_space(free_param,y,max_iter):
    """ Initializes the arrays needed for the iteration process 
        Has no other function then to unclutter the code """
    
    # number of model input/output variables
    
    n_x = len(free_param.flatten())
    n_y = len(y.flatten())
    
    # shape of map
    #map_shape = (n_x,n_y)
    
    # init variable space
    x_init = free_param.copy()
    free_param = np.zeros( (max_iter,n_x) )
    free_param[0] = x_init
    prediction = np.zeros( (max_iter,n_y) )
    cost = np.zeros( max_iter )
    #A = np.zeros( (max_iter,)+map_shape)
    
    return free_param,prediction,cost


def perturb(x,pert_scale=1e-4):
    "Adds a small perturbation to input (1D-)array"
    
    delta_x = (np.random.rand(len(x)) - 0.5)*pert_scale
    
    return x+delta_x


def monte_carlo_sample_generator(constrains):
    """ Constructs a set of homogenously distributed random values 
        in the value range provided by 'constrains'.
        Returns an array of the length of 'constrains' 
        Caution: Samples the FULL float search space if an inf value is provided! 
        
    Parameter:
    ----------
    constrains : numpy.array
        2D-array containing the upper and lower limit of every free input
        parameter in the shape (len(free_param),2).
        
    Returns:
    --------
    sample_set : numpy.array 
        1D-array containing a random vector in the range of constrains """

    """ returns min/max possible value if a +/- infinite value is pressent """
    constrains[constrains==np.inf] = np.finfo(float).max
    constrains[constrains==-np.inf] = np.finfo(float).min

    constrains_width = constrains[:,1] - constrains[:,0]
    sample_set = constrains_width*np.random.rand(len(constrains))+constrains[:,0]

    return sample_set


def local_gradient(free_param,y,fit_model,integration_scheme,
                   ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes,
                   ODE_coeff_model,constrains=np.array([None]),
                   barrier_slope=0.01, pert_scale = 1e-6,
                   time_evo_max=100, dt_time_evo=1/5,
                   stability_rel_tolerance=1e-6,
                   tail_length_stability_check=10,
                   start_stability_check=100):

    """ Calculates the gradient in the local area around the last 
        parameter set (free_param). Does so by exploring the local environment
        through random small steps in each dimension and recalculating the
        time evolution with this new slightly perturbed free_sample set.
        Local means that it roughly the same size of the previous step.

    Parameters
    ----------
    fit_model : function
        {net_flux_fit_model, direct_fit_model}
        defines how the output of the time evolution get accounted for.
        i.e. the sum of the output is returned or all its elements
    integration_scheme: function
        {euler_forward, runge_kutta}
        Selects which method is used in the integration of the time evolution.
        Euler is of first order, Runge-Kutta of second
    ODE_state : numpy.array
        1D array containing the initial state of the oberserved quantities
        in the ODE. Often also referred to as initial conditions.
    ODE_coeff : numpy.array
        2d-square-matrix containing the coefficients of the ODE
    ODE_coeff_model : function
        selects the function used for the calculation of the ODE
        coefficients. I.e. if dependencies of the current state are present.
        If no dependency is present use 'standard_weights_model'
    ODE_state_indexes : numpy.array
        1D-array containing sets of indices used to select which elements
        of the ODE_state array are optimized. 'None' if none are optimized.
    ODE_coeff_indexes : numpy.array
        1D-array containing sets of indices used to select which elements  
        of the ODE_coeff array are optimized. 'None' if none are optimized.
    constrains : numpy.array
        2D-array containing the upper and lower limit of every free input
        parameter in the shape (len(free_param),2).
    barrier_slope : positive-float
        Defines the slope of the barrier used for the soft constrain.
        Lower numbers, steeper slope. Typically between (0-1].
    pert_scale : positive float
        Maximal value which the system can be perturbed if necessary
        (i.e. if instability is found). Actual perturbation ranges
        from [0-pert_scal) uniformly distributed.
    time_evo_max
        Maximal amount of iterations allowed in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    dt_time_evo
        Size of time step used in the time evolution.
        Has the same unit as the one used in the initial ODE_state
    stability_rel_tolerance : positive float
        Defines the maximal allowed relative fluctuation range in the tail
        of the time evolution. If below, system is called stable.
    tail_length_stability_check : posi  tive integer
        Defines the length of the tail used for the stability calculation.
        Tail means the amount of elements counted from the back of the
        array.
    start_stability_check : positive integer
        Defines the element from which on we repeatably check if the
        time evolution is stable. If stable, iteration stops and last
        value is returned

    Returns
    -------
    gradient : numpy.array
        Gradient at the center point calculated by the randomly chosen 
        local environment. The gradient always points in the direction
        of steepest ascent.
    is_stable : bool
        true if stability conditions are met. 
        See verify_stability_time_evolution() for more details.
    """
    
    if len(free_param) == 1:
        free_param_diff = free_param[-1] - perturb(free_param[-1],pert_scale)
    else:
        free_param_diff = free_param[-1]-free_param[-2]

    n_x = len(free_param_diff)    
    free_param_center = free_param[-1]

    # the following 2 lines is already computed. optimization possibility
    ODE_state,ODE_coeff = fill_free_param(free_param_center,ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
    cost_center = prediction_and_costfunction(
                        free_param_center,ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,barrier_slope,
                        stability_rel_tolerance,tail_length_stability_check, start_stability_check)[1]

    free_param_local = np.full( (n_x,n_x), free_param_center)
    cost_local = np.zeros(n_x)
    
    for ii in np.arange(n_x):
        free_param_local[ii,ii] += free_param_diff[ii]

        # the following block is a terribly bad implementation performance wise
        free_param_local[ii,ii] = barrier_hard_enforcement(np.array([free_param_local[ii,ii]]),ii,
                                                  np.array([constrains[ii]]))[0]
        ODE_state,ODE_coeff = fill_free_param(free_param_local[ii],ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        ODE_coeff = normalize_columns(ODE_coeff)
        free_param_local[ii] = filter_free_param(ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        
        cost_local[ii],is_stable = prediction_and_costfunction(
                        free_param_local[ii],ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,barrier_slope,
                        stability_rel_tolerance,tail_length_stability_check, start_stability_check)[1:]
        if is_stable is False:
            return None,is_stable

    cost_diff = cost_local - cost_center

    """ The following line prevents a division by zero if 
        by chance a point does not move in between iterations.
        This is more of a work around then a feature """
    gradient = division_scalar_vector_w_zeros(cost_diff,free_param_diff)

    return gradient,is_stable