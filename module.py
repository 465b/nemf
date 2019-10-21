import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from NPZD_lib import *
import pickle
import datetime

import logging
logging.basicConfig(filename='carbonflux_inverse_model.log',level=logging.DEBUG)


# decorators 
def logging_deco(func):
    def wrapper(*args,**kwargs):
        
        date = datetime.datetime.now()
        date = "_".join( ("-".join( (str(date.year),
                                   str(date.month).zfill(2),
                                   str(date.day).zfill(2)) ),
                         str(date.hour).zfill(2)+
                         str(date.minute).zfill(2)) )
        
        result = func(*args,**kwargs)
        
        list_args = [str(ii) for ii in args]
        list_kwargs = [(k, v) for k, v in kwargs.items()]
        commented_results = [list_args+list_kwargs, result]
        
        pickling_on = open("_".join((date,func.__name__))+".pickle","wb")
        pickle.dump(commented_results,pickling_on)
        pickling_on.close()
        
        return result
    return wrapper


# Gradiant Decent

## Gradient Decent Methods
def SGD_basic(X,gradient,grad_scale):
        """ construct the gradient of the cost function  
        to minimize it with an 'SGD' approach """
        
        x_next = X[-1] - grad_scale*gradient
        
        return x_next


def SGD_momentum(X,gradient,grad_scale):
        """ construct the gradient of the cost function  
            to minimize it with an 'SGD-momentum' approach """
        
        if len(X) >= 3:
            previous_delta_x = X[-2]-X[-3]
        else:
            previous_delta_x = 0

        alpha = 1e-1 # fix function to accept alpha as an input
        x_next = X[-1] - grad_scale*gradient + alpha*previous_delta_x
        
        return x_next


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

## Integration Schemes
def euler_forward(d0,d1_weights,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    #print(d0,d1_weights)
    d0 = d0 + np.matmul(d1_weights,d0)*time_step_size
    
    return d0

def runge_kutta(d0,d1_weights,time_step,time_step_size):
    """ develops the carbon mass and carbon mass flux 
        based on a euler forward method """
    
    d0_half = d0 + time_step_size/2*np.matmul(d1_weights,d0)
    d0 = d0_half + np.matmul(d1_weights,d0_half)*time_step_size
    time_step += 1
    
    return d0, time_step


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


## Integrating Time Evolutoin

###d1 weights model
""" all weights model have the form: model(d1).
    no d0 dependence (so far neccessary) and all required
    constants should be called in the model through a function 
    (i dont know if thats very elegant, but saves an uneccessary )"""


def standard_weights_model(d0,d1):
    return d1


### Fit models
def direct_fit_model(integration_scheme, T, dt, d0, d1_weights=None, 
                       d1_weights_model=standard_weights_model,
                       tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
    """ NPZD model """
    
    F_ij,is_stable = run_time_evo(integration_scheme, T,dt,d0,d1_weights_model,d1_weights,
                tolerance,tail_length_stability_check,start_stability_check)
    F_i = F_ij[-1]

    return F_i,is_stable


def standard_fit_model(integration_scheme, T, dt, d0, d1_weights=None, 
                       d1_weights_model=standard_weights_model,
                       tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):

    F_ij, is_stable = run_time_evo(integration_scheme, T,dt,d0,d1_weights_model,d1_weights,
                tolerance,tail_length_stability_check,start_stability_check)
    F_i = F_ij[-1]
    F = np.array(np.sum(F_i) - 2*F_i[-1])

    return F, is_stable


def run_time_evo(integration_scheme, T, dt, d0, d1_weights_model, 
                 d1_weights=None,
                 tolerance=1e-5,tail_length_stability_check=10, start_stability_check=100):
                 
    """ integrates first order ODE

        integration_scheme: {euler,runge_kutta}

        T: postive float, time span that is integrated
        dt: postive float, time step that is integrated

        d0: 1D-numpy.array, set of initial values
        d1_weights_model: function, updates d1 (ODE coupling coefficients).
                          necessaray becaus coupling coefficient are allowed 
                          to be 'time' dependent, i.e the coupling is dependent
                          on the last d0 values (i.e. NPZD)
        d1_weights: 2D-square-numpy.array with ODE coefficients 
        
        tolerance: positive float, max allowed fluctuation to decide if 
                   time evolution is stable 
        tail_length_stability_check: postive integer, number of entries used from the tail d0
                                     from which the stablity is checked
        start_stability_check: positive integer, amount of steps before stability is checked """

    # initialize data arrays
    n_obs = len(d0)
    n_steps = int(T/dt)
    
    d0_log = np.zeros( (n_steps,n_obs) )
    d0_log[0] = d0

    # calculate the time evolution
    time_step = 0
    is_stable = False
    for ii in np.arange(1,n_steps):
        d1_weights = d1_weights_model(d0_log[ii],d1_weights)
        d0_log[ii] = integration_scheme(d0_log[ii-1],d1_weights,time_step,dt)

        if ((ii >= start_stability_check) & (ii%tail_length_stability_check == 0) &
            (tolerance != 0)):
            is_stable = verify_stability_time_evolution(d0_log[:ii+1],
                                            tolerance,tail_length_stability_check)
        
            if is_stable:
                return d0_log[:ii+1], is_stable
        
    
    return d0_log, is_stable


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
                   T=100, dt=1/5,tolerance=1e-6, tail_length_stability_check=10,
                   start_stability_check=100):
    """ calculates the gradient in the local area
        around the last parameter set (X).
        Local meaning with the same step size
        as in the previous step. """
    
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


# Top level Routine

def gradient_decent(fit_model,gradient_method,integration_scheme,
                    d0,d1, d1_weights_model, y,
                    d0_indexes = None, d1_indexes = None,
                    constrains=np.array([None]), mu=1e-2,
                    gd_max_iter=100,time_evo_max=100,dt_time_evo=0.2, 
                    pert_scale=1e-5,grad_scale=1e-9,
                    tolerance=1e-9,tail_length_stability_check=10, 
                    start_stability_check = 100,seed = 137):
    """ framework for applying a gradient decent approach to a 
        a model, applying a certain method 
        
        x: initial guess for the parameter set
        y: expected outcome of the model F(x)"""    

    np.random.seed(seed)
    
    x = construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)
    X,F,J = init_variable_space_for_adjoint(x,y,gd_max_iter)
    
    # ii keeps track of the position in the output array
    # jj keeps track to not exceed max iterations
    ii = 0; jj = 0
    
    while jj < gd_max_iter-1:
        if ii == 0:
            X[ii] = barrier_hard_enforcement(X[ii],ii,constrains,pert_scale)
            d0,d1 = fill_X_into_d0_d1(X[ii],d0,d1,d0_indexes,d1_indexes)
            d1 = normalize_columns(d1)
            X[ii] = construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)

            F[ii],J[ii],is_stable = prediction_and_costfunction(
                        X[ii],d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)
            if ((not is_stable) & (jj == 0)):
                warn_string = ('Initial conditions do not result in a stable model output. \n'.format(x))
                logging.debug(warn_string)
                X[ii] = add_pertubation(X[ii],pert_scale)
            else:
                X[ii+1] = add_pertubation(X[ii],pert_scale)
                ii += 1
        
        else:
            """ evaluate the system by iterating until a stable solution (F) is found
                and constructing the cost function (J) """
            X[ii] = barrier_hard_enforcement(X[ii],ii,constrains,pert_scale)
            d0,d1 = fill_X_into_d0_d1(X[ii],d0,d1,d0_indexes,d1_indexes)
            d1 = normalize_columns(d1)
            X[ii] = construct_X_from_d0_d1(d0,d1,d0_indexes,d1_indexes)
            
            F[ii],J[ii],is_stable = prediction_and_costfunction(
                        X[ii],d0, d1, d0_indexes,d1_indexes,d1_weights_model,y,fit_model,
                        integration_scheme, time_evo_max, dt_time_evo,constrains,mu,
                        tolerance,tail_length_stability_check, start_stability_check)
            if not is_stable:
                X[ii] = add_pertubation(X[ii],pert_scale)
            else:
                """ applying a decent model to find a new ( and better)
                    input variable """
                gradient,is_stable = local_gradient(X[:ii+1],y,fit_model,integration_scheme,
                                          d0,d1,d0_indexes,d1_indexes,
                                          d1_weights_model,constrains,mu,
                                          time_evo_max, dt_time_evo,
                                          tolerance=1e-6,tail_length_stability_check=10,
                                          start_stability_check=100)
                if not is_stable:
                    X[ii] = add_pertubation(X[ii],pert_scale)
                else:
                    X[ii+1] = gradient_method(X[:ii+1],gradient,grad_scale)
                    ii += 1
        jj += 1

            
    return X, F, J


## monte carlo methods

@logging_deco
def dn_monte_carlo(path_d0_init,path_d1_init,y,
                    fit_model = standard_fit_model,
                    gradient_method = SGD_basic,
                    integration_method = euler_forward,
                    d1_weights_model = standard_weights_model,
                    mu=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    d0 = np.genfromtxt(path_d0_init)
    d0_indexes = None
    d2_init = np.genfromtxt(path_d1_init,skip_header=1)[:,1:]
    d1_index = np.where( (d2_init != 0) & (d2_init != -1) & (d2_init != 1) )
    d1 = d2_init
    x = construct_X_from_d0_d1(d1=d1,d1_indexes=d1_index)
    
    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    X,F,L = init_variable_space_for_adjoint(x,y,gd_max_iter)
    X_stack = np.zeros((sample_sets,) + np.shape(X))
    F_stack = np.zeros((sample_sets,) + np.shape(F))
    #L_stack = np.zeros((sample_sets,) + np.shape(L))

    if sample_sets == 0 :
        X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                d0,d1,d1_weights_model, y,
                                d0_indexes, d1_index,constrains,mu,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            x = monte_carlo_sample_generator(constrains)
            X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                        d0,d1,d1_weights_model, y,
                                        d0_indexes, d1_index,constrains,mu,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            X_stack[ii] = X
            F_stack[ii] = F
        
    return X_stack,F_stack


@logging_deco
def NPZD_monte_carlo(path_d0_init,path_d1_init,y,
                    fit_model = direct_fit_model,
                    gradient_method = SGD_basic,
                    integration_method = euler_forward,
                    d1_weights_model = LLM_model,
                    mu=1e-6,
                    sample_sets = 5,
                    gd_max_iter=100,
                    time_evo_max=300,
                    dt_time_evo=1/5,
                    pert_scale=1e-4,
                    grad_scale=1e-12,
                    tolerance=1e-5,
                    tail_length_stability_check=10,
                    start_stability_check=100,
                    seed=137):

    d0 = np.genfromtxt(path_d0_init)
    d0_indexes = np.where(d0 == d0)
    d1 = LLM_model(d0,None)
    d1_index = None
    
    x = construct_X_from_d0_d1(d0=d0,d0_indexes=d0_indexes)
    
    constrains = np.zeros((len(x),2))
    constrains[:,0] = 0
    constrains[:,1] = 1    
    
    X,F,L = init_variable_space_for_adjoint(x,y,gd_max_iter)
    X_stack = np.zeros((sample_sets,) + np.shape(X))
    F_stack = np.zeros((sample_sets,) + np.shape(F))
    #L_stack = np.zeros((sample_sets,) + np.shape(L))

    if sample_sets == 0 :
        X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                d0,d1,d1_weights_model, y,
                                d0_indexes, d1_index,constrains,mu,
                                gd_max_iter,time_evo_max,dt_time_evo,
                                pert_scale,grad_scale,
                                tolerance,tail_length_stability_check,
                                start_stability_check,
                                seed)
        
        return X,F,L

    else:
        for ii in np.arange(0,sample_sets):
            np.random.seed()
            x = monte_carlo_sample_generator(constrains)
            X, F, L = gradient_decent(fit_model,gradient_method,integration_method,
                                        d0,d1,d1_weights_model, y,
                                        d0_indexes, d1_index,constrains,mu,
                                        gd_max_iter,time_evo_max,dt_time_evo,
                                        pert_scale,grad_scale,
                                        tolerance,tail_length_stability_check,
                                        start_stability_check,
                                        seed)
            
            X_stack[ii] = X
            F_stack[ii] = F
        
    return X_stack,F_stack


# plotting routines

def plotting_coupling_matrix(d2_weights,d1_weights,names):
    plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title("d2 coupling matrix")
    plt.imshow(d2_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()

    ax = plt.subplot(122)
    ax.set_title("d1 coupling matrix")
    plt.imshow(d1_weights,cmap='PiYG',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(names)),names, rotation=30)
    plt.yticks(np.arange(len(names)),names)
    ax.xaxis.tick_top()
    plt.savefig('coupling_matrices.svg')
    plt.show()


def plotting_time_evolution(d0_log,d1_log,time_step_size,names):
    """ plotting_time_evolution(d0_log=d0_log,d1_log=d1_log,
        time_step_size=time_step_size,names=names) """
    # plotting
    time = np.arange(np.shape(d0_log)[0]) * time_step_size 

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    for kk in np.arange(len(d1_log[0])):
        plt.title( "Absolute Carbon Mass")
        plt.plot(time,d0_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Absolute Carbon mass [mg]")
        plt.xlabel("Time [years]")
    

    plt.subplot(122)
    for kk in np.arange(len(d1_log[0])):
        plt.title( "Carbon Mass Flux")
        plt.plot(time,d1_log[:,kk],label=names[kk])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel("Carbon mass change [mg per yer]")
        plt.xlabel("Time [years]")
    plt.tight_layout()
    plt.savefig('individual_flux_mass.svg')
    plt.show()

    
    plt .figure(figsize=(10,3))
    
    plt.subplot(121)
    plt.plot(time, np.sum(d0_log,axis = 1) - 2*d0_log[:,-1])
    plt.title("Total Carbon mass over time")
    plt.ylabel("Total Carbon mass [mg]")
    plt.xlabel('Time [years]')

    plt.subplot(122)
    plt.plot(time, np.sum(d1_log,axis = 1) - 2*d1_log[:,-1] )
    plt.title("Total Carbon mass flux over time")
    plt.ylabel("Total Carbon mass flux [mg/y]")
    plt.xlabel('Time [years]')
    plt.tight_layout()
    plt.savefig('total_flux_mass.svg')
    plt.show()


def plot_XFL(X=None,F=None,L=None,context='talk'):
    
    sns.set_context(context)
    
    if (X != None).all():
        plt.figure()
        plt.title('Free Input Parameter')
        plt.plot(X[:-1])
        plt.ylabel('Input value (arb. u.)')
        plt.ylabel('Iteration Step')
        plt.show()

    if (F != None).all():
        plt.figure()
        plt.title('Time Evolution Output')
        plt.plot(F[:-1])
        plt.ylabel('Output value (arb. u.)')
        plt.xlabel('Iteration Step')
        plt.show()

    if (L != None).all():
        plt.figure()
        plt.title('Loss function over time')
        plt.plot(L[:-1])
        plt.ylabel('Loss function')
        plt.xlabel('Iteration Step')
        plt.show()