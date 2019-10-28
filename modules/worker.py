import numpy as np
import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)


# Gradiant Decent



## Cost function related Methods
def cost_function(prediction,y):
    """ normalized squared distance between prediction&y """
    cost = (1/len(y))*np.sum( (prediction - y)**2 )
    return cost


def barrier_function(free_param,constrains=np.array([None]),barrier_slope=1):
    """ constructs the additional cost that is created close
        to constrains.

        free_param:          paramter set of length n
        constrains: set of constrains of shape (n,2)
        barrier_slope: multiplier that controls the softness of the edge
        must be positive, the larger the softer """
    

    if (constrains == None).all():
        constrains = np.zeros((len(free_param),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    if len(constrains) != len(free_param):
        raise ValueError('List of constrains must have the same length as free_param')


    if barrier_slope <= 0:
        raise ValueError('barrier_slope must be a postive value.')

    cost_barrier_left = np.zeros(len(free_param))
    cost_barrier_right = np.zeros(len(free_param))
    cost_barrier = np.zeros(len(free_param))
    
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


def barrier_hard_enforcement(free_param,jj,constrains=None,pert_scale=1e-2,seed=137):
    """ if outisde the search space we enforce a 'in-constrain'
        search space by ignoring the recommendet step and
        moving it back into the search space """
    #np.random.seed(seed)

    if (constrains == None).all():
        constrains = np.zeros((len(free_param),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    if len(constrains) != len(free_param):
        raise ValueError('List of constrains must have the same length as free_param')

    for ii,[left,right] in enumerate(constrains):
        
        """ we add a small pertubation to avoid that the we remain on the boarder """
        if (free_param[ii] <= left):
            buffer = left + np.random.rand()*pert_scale
            warn_string = ( 'Left  barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(free_param[ii],buffer))
            logging.debug(warn_string)
            free_param[ii] = buffer

        if (free_param[ii] >= right):
            buffer = right - np.random.rand()*pert_scale
            warn_string = ( 'Right barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(free_param[ii],buffer))
            logging.debug(warn_string)
            free_param[ii] = buffer
            
    return free_param


# Time Evolution

## Stability

def verify_stability_time_evolution(prediction, stability_rel_tolerance=1e-6, tail_length=10):
    """ checks if the current solution is stable by 
        comparing the relative fluctuations in the 
        last tail_length model outputs to a stability_rel_tolerance value
        returns true if stable """

    prediction_tail = prediction[-tail_length-1:-1]

    average = np.average(prediction_tail,axis=0)
    spread = np.max(prediction_tail,axis=0) -np.min(prediction_tail,axis=0)
    rel_spread = spread/average
    is_stable = (rel_spread <= stability_rel_tolerance).all()
    
    return is_stable


# Helper Functions

## General Helper Function
def filter_free_param(ODE_state=None,ODE_coeff=None,ODE_state_indexes=None,ODE_coeff_indexes=None):
    """ 
        ODE_state: 1D-np.array
        ODE_coeff: 2d-square-np.array """

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


def fill_free_param(free_param,ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes):
    """ updates ODE_state & ODE_coeff with the set of optimized parameters free_param """
    
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
    """ inverts vector (1/free_param) while catching divisions by zero
        and sets them to zero """
    free_param = np.divide(a, b, out=np.zeros_like(b), where=b!=0)
    
    return free_param


def prediction_and_costfunction(free_param, ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
            integration_scheme, time_evo_max, dt_time_evo, 
            constrains=np.array([None]),barrier_slope=1,
            stability_rel_tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
    """ calculates the (stable) solution of the time evolution (prediction)
        and the resulting costfunction (cost) while applying a
        barrier_function() """

    prediction,is_stable = fit_model(integration_scheme, time_evo_max, dt_time_evo, ODE_state, ODE_coeff, ODE_coeff_model,
                            stability_rel_tolerance, tail_length_stability_check, start_stability_check)

    cost = cost_function(prediction,y)
    cost += barrier_function(free_param,constrains,barrier_slope)
    
    return prediction,cost,is_stable


## PDE-weights related Helper Functions
def discontinued_fill_free_param(x_sub,x_orig):
    """ brings the reduces parameter set back into
        its orig square matrix form 
        
        x_sub:  reduced set of original matrix
                (all non-zero/non-one)
        x_orig: the original square matrix which is filled
                with a new set of values """

    n = len(x_orig)
    idx_x = filter_free_param(x_orig.flatten())[:,0]
    free_param = x_orig.flatten()
    for ii,val in zip(idx_x,x_sub):
        free_param[int(ii)] = val
    free_param = np.reshape(free_param,(n,n))
    
    return free_param


def careful_reshape(free_param):
    n_obs = len(free_param)
    size_of_square = int(np.sqrt(n_obs))

    try:
        free_param = np.reshape(free_param, (size_of_square,size_of_square))
        return free_param
    except ValueError:
        print('Parameter input (free_param) does not fit into a square matrix.\n'+
              'Hence, is no valid input for normalisation scheme')
        return -1


def normalize_columns(free_param):
    """ enforces that all columns of free_param are normalized to 1
        otherwise an carbon mass transport amplification would occur """
    
    overshoot = np.sum(free_param,axis=0)    
    for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
        if truth_val : 
            buffer = free_param[ii,ii]
            if buffer == overshoot[ii]:
                # only the diagnoal entry is filled
                # hence, its the dump.
                pass
            elif (buffer == 0): 
                overshoot[ii] -= 1
                if (abs(overshoot[ii]) > 1e-16):
                    free_param[:,ii] -= free_param[:,ii]/(overshoot[ii]+1)*overshoot[ii]
                    free_param[ii,ii] = buffer
            else:
                free_param[:,ii] -= free_param[:,ii]/(overshoot[ii]-buffer)*overshoot[ii]
                free_param[ii,ii] = buffer

    return free_param


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


def perturb(x,pert_scale=1e-4,seed=137):
    "adds a small pertubation to input (1D-)array"
    
    delta_x = (np.random.rand(len(x)) - 0.5)*pert_scale
    
    return x+delta_x


def monte_carlo_sample_generator(constrains):
    """ constructs a set of homogenously distributed random values 
        in the value range provided by 'constrains'
        returns an array of the length of 'constrains' 
        Carefull: samples the FULL float search space if an inf value is provided! """

    """ returns min/max posible value if a +/- infinite value is pressent """
    constrains[constrains==np.inf] = np.finfo(float).max
    constrains[constrains==-np.inf] = np.finfo(float).min

    constrains_width = constrains[:,1] - constrains[:,0]
    sample_set = constrains_width*np.random.rand(len(constrains))+constrains[:,0]

    return sample_set


def local_gradient(free_param,y,fit_model,integration_scheme,
                   ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes,
                   ODE_coeff_model,constrains=np.array([None]),barrier_slope=0.01,
                   pert_scale = 1e-6,
                   time_evo_max=100, dt_time_evo=1/5,stability_rel_tolerance=1e-6, tail_length_stability_check=10,
                   start_stability_check=100):
    """ calculates the gradient in the local area
        around the last parameter set (free_param).
        Local meaning with the same step size
        as in the previous step. """
    
    if len(free_param) == 1:
        X_diff = free_param[-1] - perturb(free_param[-1],pert_scale)
    else:
        X_diff = free_param[-1]-free_param[-2]

    n_x = len(X_diff)    
    X_center = free_param[-1]

    # the following 2 lines is already computed. optimaztion possibility
    ODE_state,ODE_coeff = fill_free_param(X_center,ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
    cost_center = prediction_and_costfunction(
                        X_center,ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,barrier_slope,
                        stability_rel_tolerance,tail_length_stability_check, start_stability_check)[1]

    X_local = np.full( (n_x,n_x), X_center)
    cost_local = np.zeros(n_x)
    
    for ii in np.arange(n_x):
        X_local[ii,ii] += X_diff[ii]

        # the following block is a terribly bad implementation performance wise
        X_local[ii,ii] = barrier_hard_enforcement(np.array([X_local[ii,ii]]),ii,
                                                  np.array([constrains[ii]]))[0]
        ODE_state,ODE_coeff = fill_free_param(X_local[ii],ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        ODE_coeff = normalize_columns(ODE_coeff)
        X_local[ii] = filter_free_param(ODE_state,ODE_coeff,ODE_state_indexes,ODE_coeff_indexes)
        
        cost_local[ii],is_stable = prediction_and_costfunction(
                        X_local[ii],ODE_state, ODE_coeff, ODE_coeff_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,barrier_slope,
                        stability_rel_tolerance,tail_length_stability_check, start_stability_check)[1:]
        if is_stable is False:
            return None,is_stable

    cost_diff = cost_local - cost_center

    """ The following line prevents a division by zero if 
        by chance a point does not move in between iterations.
        This is more of a work around then a feature """
    gradient = division_scalar_vector_w_zeros(cost_diff,X_diff)

    return gradient,is_stable