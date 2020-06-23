import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from nemf import worker
from nemf import models
from nemf import decorators

#import logging
import warnings
#logging.basicConfig(filename='carbonflux_inverse_model.log',
#					level=logging.DEBUG)


def forward_model(model,method='RK45',verbose=False,t_eval=None):

	""" Runs the time integration for a provided model configuration.
		
	Parameters
	----------
	model : model_class object
		class object containing the model configuration
		and its related methods. See load_configuration
	method : string, optional
		Type of solver used for the initial-value problem aka forecasting.
		Should be on of:
			* 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [2]_. Can be applied in
              the complex domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
              Python implementation of the "DOP853" algorithm originally
              written in Fortran [14]_. A 7-th order interpolation polynomial
              accurate to 7-th order is used for the dense output.
              Can be applied in the complex domain.
            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
              order 5 [4]_. The error is controlled with a third-order accurate
              embedded formula. A cubic polynomial which satisfies the
              collocation conditions is used for the dense output.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the
              complex domain.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
              switching [7]_, [8]_. This is a wrapper of the Fortran solver
              from ODEPACK.

		Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
        for non-stiff problems and implicit methods ('Radau', 'BDF') for
        stiff problems [9]_. Among Runge-Kutta methods, 'DOP853' is recommended
        for solving with high precision (low values of `rtol` and `atol`).
        If not sure, first try to run 'RK45'. If it makes unusually many
        iterations, diverges, or fails, your problem is likely to be stiff and
        you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
        choice, but it might be somewhat less convenient to work with as it
        wraps old Fortran code.

	verbose : bool, optional
		Flag for extra verbosity during runtime
	t_eval : 1d-array, optional
		contains time stamps in posix time for which a solution shall be 
		found and returned.

	Returns
	-------
	model : model_class object
		class object containing the model configuration, model run results,
		and its related methods

	References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.

	"""

	[initial_states,args] = model.fetch_param()
	differential_equation = model.de_constructor()
	model.initialize_log(maxiter=1)	

	if t_eval is None:
		t_start = 0
		t_stop = model.configuration['time_evo_max']
		dt = model.configuration['dt_time_evo']
		t = np.arange(t_start,t_stop,dt)
	else:
		t_start = min(t_eval)
		t_stop = max(t_eval)
		t = np.linspace(t_start,t_stop,num=1000)
	
	sol = solve_ivp(differential_equation,[t_start,t_stop],initial_states,
					method=method,args=[args], dense_output=True)
	y_t = sol.sol(t).T

	if verbose:
		print(f'ode solution: {sol}')
		print(f't_events: {sol.t_events}')

	t = np.reshape(t,(len(t),1))
	time_series = np.concatenate( (t,y_t),axis=1)
	model.log['time_series'] = time_series

	return model


def inverse_model(model,nlp_method='SLSQP',
					ivp_method='Radau',
					sample_sets = 3,
					maxiter=1000,
					seed=137,
					verbose=False,
					debug=False):

	""" Fits the model to data.

	Optimizes a set of randomly generated free parameters and returns
	their optimized values and the corresponding fit-model and cost-
	function output 

	Parameters
	----------
	model : model_class object
		class object containing the model configuration
		and its related methods. See load_configuration()
	nlp_method : string, optional
		Type of solver for the non-linear-programming problem aka fitting.
		Should be one of:
			* ‘trust-constr’
			* ‘SLSQP’
			* 'L-BFGS-B'
			* 'TNC
			* 'Powell'

		For problems with constraints use one of:
			* ‘trust-constr’
			* ‘SLSQP’

	ivp_method : string, optional
		Type of solver used for the initial-value problem aka forecasting.
		Should be on of:
            * 'Radau' (default): 
				Implicit Runge-Kutta method of the Radau IIA family
			* 'RK45': Explicit Runge-Kutta method of order 5(4) [1]_.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method
            * 'LSODA': Adams/BDF method with automatic stiffness detection

		Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
        for non-stiff problems and implicit methods ('Radau', 'BDF') for
        stiff problems [9]_. Among Runge-Kutta methods, 'DOP853' is recommended
        for solving with high precision (low values of `rtol` and `atol`).
        If not sure, first try to run 'RK45'. If it makes unusually many
        iterations, diverges, or fails, your problem is likely to be stiff and
        you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
        choice, but it might be somewhat less convenient to work with as it
        wraps old Fortran code.

	
	sample_sets : positive integer, optional
		Amount of randomly generated sample sets used as initial free
		parameters
	maxiter : positive integer, optional
		Maximal amount of iterations allowed in the gradient descent
		algorithm.
	seed : positive integer, optional
		Initializes the random number generator. Used to recreate the
		same set of pseudo-random numbers. Helpfull when debugging.
	verbose : boo, optional
		Flag for extra verbosity during runtime

	Returns
	-------
	model : model_class object
		class object containing the model configuration, 
		model run results (parameters, model, prediction, cost),
		and its related methods

	Notes
	-----

	Non-linear-programming solvers aka minimizers
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	**Bound-Constrained minimization**

    Method `L-BFGS-B` uses the L-BFGS-B
    algorithm [B6]_, [B7]_ for bound constrained minimization.
 
    Method `Powell` is a modification
    of Powell's method [B3]_, [B4]_ which is a conjugate direction
    method. It performs sequential one-dimensional minimizations along
    each vector of the directions set (`direc` field in `options` and
    `info`), which is updated at each iteration of the main
    minimization loop. The function need not be differentiable, and no
    derivatives are taken. If bounds are not provided, then an
    unbounded line search will be used. If bounds are provided and
    the initial guess is within the bounds, then every function
    evaluation throughout the minimization procedure will be within
    the bounds. If bounds are provided, the initial guess is outside
    the bounds, and `direc` is full rank (default has full rank), then
    some function evaluations during the first iteration may be
    outside the bounds, but every function evaluation after the first
    iteration will be within the bounds. If `direc` is not full rank,
    then some parameters may not be optimized and the solution is not
    guaranteed to be within the bounds.
    
	Method `TNC` uses a truncated Newton
    algorithm [B5]_, [B8]_ to minimize a function with variables subject
    to bounds. This algorithm uses gradient information; it is also
    called Newton Conjugate-Gradient. It differs from the *Newton-CG*
    method described above as it wraps a C implementation and allows
    each variable to be given upper and lower bounds.


	**Constrained Minimization**
    
    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
    Least SQuares Programming to minimize a function of several
    variables with any combination of bounds, equality and inequality
    constraints. The method wraps the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft [B12]_. Note that the
    wrapper handles infinite values in bounds by converting them into
    large floating values.
    Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a
    trust-region algorithm for constrained optimization. It swiches
    between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in [B17]_ and in [B5]_, p. 549. When
    inequality constraints  are imposed as well, it swiches to the trust-region
    interior point  method described in [B16]_. This interior point algorithm,
    in turn, solves inequality constraints by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

	Initial-Value-Problem solvers aka forecasting
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	The available options are:
		
		* 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
			The error is controlled assuming accuracy of the fourth-order
			method, but steps are taken using the fifth-order accurate
			formula (local extrapolation is done). A quartic interpolation
			polynomial is used for the dense output [2]_. Can be applied in
			the complex domain.
		* 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
			is controlled assuming accuracy of the second-order method, but
			steps are taken using the third-order accurate formula (local
			extrapolation is done). A cubic Hermite polynomial is used for the
			dense output. Can be applied in the complex domain.
		* 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
			Python implementation of the "DOP853" algorithm originally
			written in Fortran [14]_. A 7-th order interpolation polynomial
			accurate to 7-th order is used for the dense output.
			Can be applied in the complex domain.
		* 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
			order 5 [4]_. The error is controlled with a third-order accurate
			embedded formula. A cubic polynomial which satisfies the
			collocation conditions is used for the dense output.
		* 'BDF': Implicit multi-step variable-order (1 to 5) method based
			on a backward differentiation formula for the derivative
			approximation [5]_. The implementation follows the one described
			in [6]_. A quasi-constant step scheme is used and accuracy is
			enhanced using the NDF modification. Can be applied in the
			complex domain.
		* 'LSODA': Adams/BDF method with automatic stiffness detection and
			switching [7]_, [8]_. This is a wrapper of the Fortran solver
			from ODEPACK.

	Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
	for non-stiff problems and implicit methods ('Radau', 'BDF') for
	stiff problems [9]_. Among Runge-Kutta methods, 'DOP853' is recommended
	for solving with high precision (low values of `rtol` and `atol`).
	If not sure, first try to run 'RK45'. If it makes unusually many
	iterations, diverges, or fails, your problem is likely to be stiff and
	you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
	choice, but it might be somewhat less convenient to work with as it
	wraps old Fortran code.

	
	References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    .. [B3] Powell, M J D. 1964. An efficient method for finding the minimum of
       a function of several variables without calculating derivatives. The
       Computer Journal 7: 155-162.
    .. [B4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
       Numerical Recipes (any edition), Cambridge University Press.
    .. [B5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
       Springer New York.
    .. [B6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
       Algorithm for Bound Constrained Optimization. SIAM Journal on
       Scientific and Statistical Computing 16 (5): 1190-1208.
    .. [B7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm
       778: L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization. ACM Transactions on Mathematical Software 23 (4):
       550-560.
    .. [B8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.
       1984. SIAM Journal of Numerical Analysis 21: 770-778.
    .. [B12] Kraft, D. A software package for sequential quadratic
       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace
       Center -- Institute for Flight Mechanics, Koln, Germany.
    .. [B16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.
        An interior point algorithm for large-scale nonlinear  programming.
        SIAM Journal on Optimization 9.4: 877-900.
    .. [B17] Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the
        implementation of an algorithm for large-scale equality constrained
        optimization. SIAM Journal on Optimization 8.3: 682-706.

	"""

	# seeds random generator to create reproducible runs
	np.random.seed(seed)

	if model.reference_data is None:
		warnings.warn('Monte Carlo optimization method called with '
						+'no parameters to optimise. '
						+'Falling back to running model without '
						+'optimization.')
		return forward_model(model)
	
	else:
		[fit_param, bnd_param] = model.fetch_to_optimize_args()[0][1:3]
		objective_function = worker.construct_objective(
										model,method=ivp_method,debug=debug)
		logger = model.construct_callback(method=nlp_method,debug=debug)
		model.initialize_log(maxiter=maxiter)

		if ((verbose != True) & (nlp_method != 'trust-constr')):
			print(f"Verbose = {verbose} is no valid option for this method. "
					+"The only valid option is 'True'.")
			verbose = True

		cons = model.fetch_constraints()
		if cons ==  None:
			if nlp_method == 'trust-constr':
				out = minimize(objective_function,fit_param,method=nlp_method,
							bounds=bnd_param,callback=logger,
							options={'verbose': verbose, 'maxiter': maxiter})
			else:
				out = minimize(objective_function,fit_param,method=nlp_method,
							bounds=bnd_param,callback=logger,
							options={'disp': verbose, 'maxiter': maxiter})
							
		else:
			if nlp_method not in ('SLSQP','trust-constr'):
				print("Invalid solver selected for a inverse problem with "
						+"constraints. The only valid choices are:\n"
						+"\t'SLSQP' or 'trust-constr'.\nDefaulting to 'SLSQP'.")
				nlp_method = 'SLSQP'

			if nlp_method == 'trust-constr':
				out = minimize(objective_function,fit_param,method=nlp_method,
							bounds=bnd_param,constraints=cons,callback=logger,tol=1e-6,
							options={'verbose': verbose,'maxiter': maxiter})
			else:
				out = minimize(objective_function,fit_param,method=nlp_method,
							bounds=bnd_param,constraints=cons,callback=logger,tol=1e-6,
							options={'disp': verbose,'maxiter': maxiter})

		model.update_system_with_parameters(out.x)
		if verbose:
			print(out)
		
	return model