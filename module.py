import numpy as np
import matplotlib.pyplot as plt
import warnings

## integration schemes

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


def verify_stabel_soloution_of_time_evol():
    """ i will require some criteria to check if solution is sufficiently stable
        to break time_evolution early (safe comp. time) or to exclude because its unstable """
    pass


def run_time_evo(method,T,dt,d0,d1_weights_model,
                 d1_weights=None,weights_model_constants=None):
    """ integrates first order ODE

        integration scheme
        method: {euler,runge_kutta}

        T: postive float, time span that is integrated
        dt: postive float, time step that is integrated

        d0: 1D-numpy.array, set of initial values
        d1_weights: 2D-square-numpy.array with ODE coefficients """

    # calculate all time constants
    n_obs = len(d0)
    n_steps = int(T/dt)
    
    # initialize data arrays
    d0_log = np.zeros( (n_steps,n_obs) )
    d0_log[0] = d0

    # calculate the time evolution
    time_step = 0
    for ii in np.arange(1,n_steps):
        d1_weights = d1_weights_model(d0_log[ii],d1_weights,weights_model_constants)
        d0_log[ii] = method(d0_log[ii-1],d1_weights,time_step,dt)
    
    return d0_log


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


def add_pertubation(x,pert_scale=0.00001,seed=137):
    "adds a small pertubation to input (1D-)array"
    
    np.random.seed(seed)
    delta_x = np.random.rand(len(x))*pert_scale
    print('Added random pertubation')
    
    return x+delta_x


def division_scalar_vector_w_zeros(a,b):
    """ inverts vector (1/x) while catching divisions by zero
        and sets them to zero """
    x = np.divide(a, b, out=np.zeros_like(b), where=b!=0)
    
    return x


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
             ((val != 0) & ( val != -1) & ( val != -1))]
    return np.array(x_sub)


## model specific functions

def normalize_columns(x):
    """ enforces that all columns of x are normalized to 1
        otherwise an carbon mass transport amplification would occur """
    
    overshoot = np.sum(x,axis=0)
    
    for ii,truth_val in enumerate(abs(overshoot) > 1e-16):
        if truth_val : 
            if x[ii,ii] == overshoot[ii]:
                # only the diagnoal entry is filled
                # hence, its the dump. 
                pass 
            else: 
                # the following assumed that only diagonal values can be negative
                buffer = x[ii,ii]
                x[:,ii] -= x[:,ii]/(overshoot[ii]-buffer)*overshoot[ii]
                x[ii,ii] = buffer

    return x


def interaction_matrix_modifier():
    """ this function is inteded to modify the interactino matrices (d1_/d2_weights)
        to find appropriate solutions to the system. """
    
    """ we have 53 non-zero, non-negative-one values.
        len(d2_weights[((d2_weights != -1) & (d2_weights != 0)) == True])
        
        assuming we constrain our precision to a resolution of 5% 
        we have a parameter searchspace of 20**53
        which gives us convenient search time of 3e61 years
        wow, such convenience """


## Gradiant Decent

### initialization routines 

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

### Gradiant Decent Methods

def local_gradient(X,y,fit_model,constrains,mu=0.01):
    """ calculates the gradient in the local area
        around the last parameter set (X).
        Local meaning with the same step size
        as in the previous step. """
    
    X_diff = X[-1]-X[-2]
    n_x = len(X_diff)
    
    X_center = X[-1]
    F_center = fit_model(X_center)
    J_center = cost_function(F_center,y)
    
    X_local = np.full( (n_x,n_x), X_center)
    F_local = np.zeros((n_x,n_x))
    J_local = np.zeros(n_x)
    
    for ii in np.arange(n_x):
        X_local[ii,ii] += X_diff[ii]
    
    for ii in np.arange(n_x):
        F_local[ii] = fit_model(X_local[ii])
        J_local[ii] = cost_function(F_local[ii],y)
        J_local[ii] += barrier_function(X[-1],constrains,mu)
    
    J_diff = J_local - J_center
    gradient = J_diff/X_diff
    
    return gradient


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

def cost_function(F,y):
    """ normalized squared distance between F&y """
    J = (1/len(F))*np.sum( (F - y)**2 )
    return J

### applying Gradiant Decent

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
            J_barrier_left[ii] = -np.log(left+x[ii])

        # right constrain
        if (right == np.inf):
            J_barrier_right[ii] = 0
        else:
            J_barrier_right[ii] = -np.log(right-x[ii])

        J_barrier = J_barrier_left + J_barrier_right
        J_barrier = np.sum(J_barrier)*mu
    
    return J_barrier

def barrier_hard_enforcement(x,constrains=None,pert_scale=1e-4,seed=137):
    """ if outisde the search space we enforce a 'in-constrain'
        search space by ignoring the recommendet step and
        moving it back into the search space """
    np.random.seed(seed)

    if (constrains == None).all():
        constrains = np.zeros((len(x),2))
        constrains[:,0] = -np.inf
        constrains[:,1] = +np.inf

    if len(constrains) != len(x):
        raise ValueError('List of constrains must have the same length as x')

    for ii,[left,right] in enumerate(constrains):
        if (x[ii] <= left):
            x[ii] = left + np.random.rand()*pert_scale
            warnings.warn('A hard (left) barrier enforcment was necessary. '+
                          'Consider adjusting your input parameter', Warning)
            # we add a small pertubation to avoid
            # that the we remain on the boarder
        if (x[ii] >= right):
            x[ii] = right - np.random.rand()*pert_scale
            warnings.warn('A hard (right) barrier enforcment was necessary. '+
                          'Consider adjusting your input parameter', Warning)

    return x


def gradient_decent(fit_model,gradient_method,x,y,
                    constrains=np.array([None]),
                    max_iter=5,mu=1,pert_scale=0.01,grad_scale=0.01,
                    seed = 137):
    """ framework for applying a gradient decent approach to a 
        a model, applying a certain method 
        
        x: initial guess for the parameter set
        y: expected outcome of the model F(x)"""    
    
    X,F,J = init_variable_space_for_adjoint(x,y,max_iter)
    
    for ii in np.arange(0,max_iter-1):
        if ii == 0:
            """ At the beginning of the iteration the cost function space 
                is completely unexpored. To get started we do a random move """

            F[ii] = fit_model(X[ii])
            J[ii] = cost_function(F[ii],y)
            J[ii] += barrier_function(X[ii],constrains,mu)
            X[ii+1] = add_pertubation(X[ii],pert_scale,seed)

        else:
            """ constructing the cost function (J)
                we call the cost function at a single point 'forcing' """
            
            
            """ evaluate change in field caused by last step """
            F[ii] = fit_model(X[ii]) 
            J[ii] = cost_function(F[ii],y)
            J[ii] += barrier_function(X[ii],constrains,mu)
            #print('forcing:\t\t{}'.format(delta_forcing))
            #print('x_diff:\t\t\t{}'.format(delta_x))
            #print('gradient:\t\t{}'.format(delta_forcing/delta_x))

            """ applying a decent model to find a new ( and better)
                input variable """
            
            gradient = local_gradient(X[:ii+1],y,fit_model, constrains, mu)
            X[ii+1] = gradient_method(X[:ii+1],gradient,grad_scale)
            X[ii+1] = barrier_hard_enforcement(X[ii+1],constrains)
            
            
    return X, F, J


## plotting

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


def plotting_standard(d0_log,d1_log,time_step_size,names):
    """ plotting_standard(d0_log=d0_log,d1_log=d1_log,time_step_size=time_step_size,names=names) """
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