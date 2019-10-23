import numpy as np
import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)


# Gradiant Decent



## Cost function related Methods
def cost_function(F,y):
    """ normalized squared distance between F&y """
    J = (1/len(y))*np.sum( (F - y)**2 )
    return J


def barrier_function(x,constrains=np.array([None]),mu=1):
    """ constructs the additional cost that is created close
        to constrains.

        x:          paramter set of length n
        constrains: set of constrains of shape (n,2)
        mu: multiplier that controls the softness of the edge
        must be positive, the larger the softer """
    

    if (constrains == None).all():
        constrains = np.zeros((len(x),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    if len(constrains) != len(x):
        raise ValueError('List of constrains must have the same length as x')


    if mu <= 0:
        raise ValueError('mu must be a postive value.')

    J_barrier_left = np.zeros(len(x))
    J_barrier_right = np.zeros(len(x))
    J_barrier = np.zeros(len(x))
    
    for ii,[left,right] in enumerate(constrains):
        #left constrain
        if (left == -np.inf):
            J_barrier_left[ii] = 0
        else:
            J_barrier_left[ii] = -np.log(x[ii]-left)

        # right constrain
        if (right == np.inf):
            J_barrier_right[ii] = 0
        else:
            J_barrier_right[ii] = -np.log(-x[ii]+right)

    J_barrier = J_barrier_left + J_barrier_right
    J_barrier = np.sum(J_barrier)*mu
    
    return J_barrier


def barrier_hard_enforcement(x,jj,constrains=None,pert_scale=1e-2,seed=137):
    """ if outisde the search space we enforce a 'in-constrain'
        search space by ignoring the recommendet step and
        moving it back into the search space """
    #np.random.seed(seed)

    if (constrains == None).all():
        constrains = np.zeros((len(x),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    if len(constrains) != len(x):
        raise ValueError('List of constrains must have the same length as x')

    for ii,[left,right] in enumerate(constrains):
        
        # we add a small pertubation to avoid
        # that the we remain on the boarder
        if (x[ii] <= left):

            new_x = left + np.random.rand()*pert_scale
            warn_string = ( 'Left  barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(x[ii],new_x))
            logging.debug(warn_string)
            x[ii] = left + np.random.rand()*pert_scale
            

        if (x[ii] >= right):
            new_x = right - np.random.rand()*pert_scale
            warn_string = ( 'Right barrier enforcement'+
                            'at step {:4d} {:4d}. '.format(jj,ii)+
                            'Value shifted from {:+8.2E} to {:+8.2E}'.format(x[ii],new_x))
            logging.debug(warn_string)
            x[ii] = left + np.random.rand()*pert_scale
            
    return x


# Time Evolution

## Stability

def verify_stability_time_evolution(F, tolerance=1e-6, N=10):
    """ checks if the current solution is stable by 
        comparing the relative fluctuations in the 
        last N model outputs to a tolerance value
        returns true if stable """

    F_tail = F[-N-1:-1]

    average = np.average(F_tail,axis=0)
    spread = np.max(F_tail,axis=0) -np.min(F_tail,axis=0)
    rel_spread = spread/average
    is_stable = (rel_spread <= tolerance).all()
    
    return is_stable


# Helper Functions

## General Helper Function
def construct_X_from_d0_d1(d0=None,d1=None,d0_indexes=None,d1_indexes=None):
    """ 
        d0: 1D-np.array
        d1: 2d-square-np.array """

    if ( (d0 is None) or (d0_indexes is None) ):
        x_d0 = np.array([])
    else:
        x_d0 = d0[d0_indexes]
    
    if ( (d1 is None) or (d1_indexes is None) ):
        x_d1 = np.array([])
    else:
        x_d1 = d1[d1_indexes]

    x = np.append(x_d0,x_d1)
    
    return x


def fill_X_into_d0_d1(X,d0,d1,d0_indexes,d1_indexes):
    """ updates d0 & d1 with the set of optimized parameters X """
    
    if d0_indexes is None:
        n_d0 = 0
    else:
        n_d0 = len(d0_indexes)
        d0[d0_indexes] = X[:n_d0]
    
    if d1_indexes is None:
        pass
    else:
        d1[d1_indexes] = X[n_d0:]

    return d0,d1


def division_scalar_vector_w_zeros(a,b):
    """ inverts vector (1/x) while catching divisions by zero
        and sets them to zero """
    x = np.divide(a, b, out=np.zeros_like(b), where=b!=0)
    
    return x


def prediction_and_costfunction(X, d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
            integration_scheme, T, dt, 
            constrains=np.array([None]),mu=1,
            tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
    """ calculates the (stable) solution of the time evolution (F)
        and the resulting costfunction (J) while applying a
        barrier_function() """

    F,is_stable = fit_model(integration_scheme, T, dt, d0, d1, d1_weights_model,
                            tolerance, tail_length_stability_check, start_stability_check)

    J = cost_function(F,y)
    J += barrier_function(X,constrains,mu)
    
    return F,J,is_stable


## PDE-weights related Helper Functions
def fill_free_param(x_sub,x_orig):
    """ brings the reduces parameter set back into
        its orig square matrix form 
        
        x_sub:  reduced set of original matrix
                (all non-zero/non-one)
        x_orig: the original square matrix which is filled
                with a new set of values """

    n = len(x_orig)
    idx_x = filter_free_param(x_orig.flatten())[:,0]
    x = x_orig.flatten()
    for ii,val in zip(idx_x,x_sub):
        x[int(ii)] = val
    x = np.reshape(x,(n,n))
    
    return x


def filter_free_param(x):
    x_sub = [(ii,val) for ii, val in enumerate(x) if
             ((val != 0) & ( val != -1) & ( val != 1))]
    return np.array(x_sub)


def careful_reshape(x):
    n_obs = len(x)
    size_of_square = int(np.sqrt(n_obs))

    try:
        x = np.reshape(x, (size_of_square,size_of_square))
        return x
    except ValueError:
        print('Parameter input (x) does not fit into a square matrix.\n'+
              'Hence, is no valid input for normalisation scheme')
        return -1


def normalize_columns(x):
    """ enforces that all columns of x are normalized to 1
        otherwise an carbon mass transport amplification would occur """
    
    overshoot = np.sum(x,axis=0)    
    for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
        if truth_val : 
            buffer = x[ii,ii]
            if buffer == overshoot[ii]:
                # only the diagnoal entry is filled
                # hence, its the dump.
                pass
            elif (buffer == 0): 
                overshoot[ii] -= 1
                if (abs(overshoot[ii]) > 1e-16):
                    x[:,ii] -= x[:,ii]/(overshoot[ii]+1)*overshoot[ii]
                    x[ii,ii] = buffer
            else:
                x[:,ii] -= x[:,ii]/(overshoot[ii]-buffer)*overshoot[ii]
                x[ii,ii] = buffer

    return x


## Gradient Decent related Helper Functions
def init_variable_space_for_adjoint(x,y,max_iter):
    """ Initializes the arrays needed for the iteration process 
        Has no other function then to unclutter the code """
    
    # number of model input/output variables
    
    n_x = len(x.flatten())
    n_y = len(y.flatten())
    
    # shape of map
    #map_shape = (n_x,n_y)
    
    # init variable space
    x_init = x.copy()
    x = np.zeros( (max_iter,n_x) )
    x[0] = x_init
    F = np.zeros( (max_iter,n_y) )
    J = np.zeros( max_iter )
    #A = np.zeros( (max_iter,)+map_shape)
    
    return x,F,J


def add_pertubation(x,pert_scale=1e-4,seed=137):
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


def local_gradient(X,y,fit_model,integration_scheme,
                   d0,d1,d0_indexes,d1_indexes,
                   d1_weights_model,constrains=np.array([None]),mu=0.01,
                   pert_scale = 1e-6,
                   T=100, dt=1/5,tolerance=1e-6, tail_length_stability_check=10,
                   start_stability_check=100):
    """ calculates the gradient in the local area
        around the last parameter set (X).
        Local meaning with the same step size
        as in the previous step. """
    
    if len(X) == 1:
        X_diff = X[-1] - add_pertubation(X[-1],pert_scale)
    else:
        X_diff = X[-1]-X[-2]

    n_x = len(X_diff)    
    X_center = X[-1]

    # the following 2 lines is already computed. optimaztion possibility
    d0,d1 = fill_X_into_d0_d1(X_center,d0,d1,d0_indexes,d1_indexes)
    J_center = prediction_and_costfunction(
                        X_center,d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, T, dt,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)[1]

    X_local = np.full( (n_x,n_x), X_center)
    J_local = np.zeros(n_x)
    
    for ii in np.arange(n_x):
        X_local[ii,ii] += X_diff[ii]

        # the following block is a terribly bad implementation performance wise
        X_local[ii,ii] = barrier_hard_enforcement(np.array([X_local[ii,ii]]),ii,
                                                  np.array([constrains[ii]]))[0]
        d0,d1 = fill_X_into_d0_d1(X_local[ii],d0,d1,d0_indexes,d1_indexes)
        d1 = normalize_columns(d1)
        X_local[ii] = construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)
        
        J_local[ii],is_stable = prediction_and_costfunction(
                        X_local[ii],d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, T, dt,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)[1:]
        if is_stable is False:
            return None,is_stable

    J_diff = J_local - J_center

    """ The following line prevents a division by zero if 
        by chance a point does not move in between iterations.
        This is more of a work around then a feature """
    gradient = division_scalar_vector_w_zeros(J_diff,X_diff)

    return gradient,is_stable