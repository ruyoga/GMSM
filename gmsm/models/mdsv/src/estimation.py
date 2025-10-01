import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from typing import Dict, Optional, Tuple, List, Callable
import warnings
from dataclasses import dataclass


@dataclass
class EstimationOptions:
    """Options for MDSV estimation"""
    method: str = 'L-BFGS-B'  # Optimization method
    maxiter: int = 1000  # Maximum iterations
    tol: float = 1e-8  # Tolerance
    verbose: bool = False  # Print progress
    n_starts: int = 1  # Number of random starts
    use_bounds: bool = True  # Use parameter bounds
    global_search: bool = False  # Use global optimization
    parallel: bool = False  # Use parallel computation


class MDSVEstimator:
    """
    Parameter estimation for MDSV models

    Provides various estimation methods including:
    - Maximum likelihood estimation
    - Method of moments
    - Two-step estimation
    - Robust estimation
    """

    def __init__(self, model):
        """
        Initialize estimator

        Parameters
        ----------
        model : MDSV
            MDSV model instance
        """
        self.model = model
        self.N = model.N
        self.D = model.D
        self.model_type = model.model_type
        self.leverage = model.leverage

        # Set parameter bounds
        self._set_bounds()

    def _set_bounds(self):
        """Set parameter bounds for optimization"""
        bounds = [
            (0.01, 0.99),  # omega
            (0.5, 0.9999),  # a
            (1.01, 10),  # b
            (0.01, 10),  # sigma
            (0.01, 0.99)  # v0
        ]

        if self.model_type == 1:
            bounds.append((0.01, 10))  # shape
        elif self.model_type == 2:
            bounds.extend([
                (-5, 5),  # xi
                (0, 2),  # varphi
                (-2, 2),  # delta1
                (-0.5, 0.5),  # delta2
                (0.01, 2)  # shape
            ])

        if self.leverage:
            bounds.extend([
                (0.01, 5),  # l
                (0.01, 0.99)  # theta
            ])

        self.bounds = bounds

    def estimate(self, data: np.ndarray,
                 initial_params: Optional[np.ndarray] = None,
                 options: Optional[EstimationOptions] = None) -> Dict:
        """
        Main estimation function

        Parameters
        ----------
        data : np.ndarray
            Observation data
        initial_params : np.ndarray, optional
            Initial parameter values
        options : EstimationOptions, optional
            Estimation options

        Returns
        -------
        dict
            Estimation results
        """
        if options is None:
            options = EstimationOptions()

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T

        # Get initial parameters
        if initial_params is None:
            if options.n_starts > 1:
                initial_params = self._generate_multiple_starts(data, options.n_starts)
            else:
                initial_params = [self._get_initial_params(data)]
        else:
            initial_params = [self.model._nat_to_work(initial_params)]

        # Run optimization
        if options.global_search:
            result = self._global_optimization(data, options)
        else:
            result = self._local_optimization(data, initial_params, options)

        return result

    def _local_optimization(self, data: np.ndarray,
                            initial_params: List[np.ndarray],
                            options: EstimationOptions) -> Dict:
        """Run local optimization"""
        best_result = None
        best_log_lik = np.inf

        for i, x0 in enumerate(initial_params):
            if options.verbose and len(initial_params) > 1:
                print(f"Starting optimization {i + 1}/{len(initial_params)}...")

            # Set up optimization
            opt_options = {
                'maxiter': options.maxiter,
                'disp': options.verbose
            }

            if options.use_bounds:
                # Transform bounds to working parameter space
                working_bounds = self._transform_bounds_to_working()
            else:
                working_bounds = None

            # Optimize
            try:
                result = minimize(
                    fun=self.model._log_likelihood,
                    x0=x0,
                    args=(data,),
                    method=options.method,
                    bounds=working_bounds,
                    options=opt_options,
                    tol=options.tol
                )

                if result.fun < best_log_lik:
                    best_log_lik = result.fun
                    best_result = result

            except Exception as e:
                if options.verbose:
                    print(f"Optimization {i + 1} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All optimization attempts failed")

        return self._process_result(best_result, data)

    def _global_optimization(self, data: np.ndarray,
                             options: EstimationOptions) -> Dict:
        """Run global optimization using differential evolution"""
        if options.verbose:
            print("Running global optimization...")

        # Transform bounds
        working_bounds = self._transform_bounds_to_working()

        # Run differential evolution
        result = differential_evolution(
            func=self.model._log_likelihood,
            bounds=working_bounds,
            args=(data,),
            maxiter=options.maxiter,
            tol=options.tol,
            disp=options.verbose,
            workers=1 if not options.parallel else -1
        )

        # Refine with local optimization
        if options.verbose:
            print("Refining with local optimization...")

        refined = minimize(
            fun=self.model._log_likelihood,
            x0=result.x,
            args=(data,),
            method='L-BFGS-B',
            bounds=working_bounds
        )

        return self._process_result(refined, data)

    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Generate initial parameter values"""
        params = np.zeros(self.model._get_n_params())

        # Basic parameters
        params[0] = 0.5  # omega
        params[1] = 0.95  # a
        params[2] = 2.5  # b
        params[3] = np.std(data[:, 0])  # sigma
        params[4] = 0.7  # v0

        idx = 5
        if self.model_type == 1:
            # RV model
            params[idx] = 2.0  # shape
            idx += 1
        elif self.model_type == 2:
            # Joint model
            params[idx] = -0.5  # xi
            params[idx + 1] = 0.9  # varphi
            params[idx + 2] = -0.1  # delta1
            params[idx + 3] = 0.05  # delta2
            params[idx + 4] = 0.2  # shape
            idx += 5

        if self.leverage:
            params[idx] = 0.8  # l
            params[idx + 1] = 0.9  # theta

        # Transform to working parameters
        return self.model._nat_to_work(params)

    def _generate_multiple_starts(self, data: np.ndarray, n_starts: int) -> List[np.ndarray]:
        """Generate multiple starting points"""
        starts = []

        # First start: default initialization
        starts.append(self._get_initial_params(data))

        # Additional random starts
        for _ in range(n_starts - 1):
            params = np.zeros(self.model._get_n_params())

            # Random values within bounds
            for i, (low, high) in enumerate(self.bounds):
                if i < len(params):
                    params[i] = np.random.uniform(low, high)

            starts.append(self.model._nat_to_work(params))

        return starts

    def _transform_bounds_to_working(self) -> List[Tuple[float, float]]:
        """Transform natural parameter bounds to working parameter space"""
        working_bounds = []

        for i, (low, high) in enumerate(self.bounds):
            # Create dummy parameters at bounds
            params_low = np.zeros(len(self.bounds))
            params_high = np.zeros(len(self.bounds))

            for j in range(len(self.bounds)):
                if j == i:
                    params_low[j] = low
                    params_high[j] = high
                else:
                    params_low[j] = (self.bounds[j][0] + self.bounds[j][1]) / 2
                    params_high[j] = params_low[j]

            # Transform to working space
            work_low = self.model._nat_to_work(params_low)[i]
            work_high = self.model._nat_to_work(params_high)[i]

            # Handle parameter transformations
            if i in [0, 1, 4]:  # omega, a, v0 (logit transform)
                working_bounds.append((work_high, work_low))  # Reversed due to logit
            else:
                working_bounds.append((min(work_low, work_high), max(work_low, work_high)))

        return working_bounds

    def _process_result(self, opt_result, data: np.ndarray) -> Dict:
        """Process optimization result"""
        # Extract parameters
        para_tilde = opt_result.x
        para = self.model._work_to_nat(para_tilde)

        # Create parameter dictionary
        params = self.model._create_param_dict(para)

        # Compute information criteria
        n_obs = len(data)
        n_params = len(para)
        log_lik = -opt_result.fun
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n_obs)

        # Compute standard errors (if Hessian available)
        std_errors = None
        if hasattr(opt_result, 'hess_inv'):
            try:
                cov_matrix = opt_result.hess_inv
                if hasattr(cov_matrix, 'todense'):
                    cov_matrix = cov_matrix.todense()
                std_errors = np.sqrt(np.diag(cov_matrix))
            except:
                pass

        return {
            'parameters': params,
            'parameter_array': para,
            'log_likelihood': log_lik,
            'aic': aic,
            'bic': bic,
            'convergence': opt_result.success,
            'n_iterations': opt_result.nit if hasattr(opt_result, 'nit') else 0,
            'message': opt_result.message if hasattr(opt_result, 'message') else '',
            'std_errors': std_errors,
            'optimization_result': opt_result
        }

    def two_step_estimation(self, data: np.ndarray,
                            options: Optional[EstimationOptions] = None) -> Dict:
        """
        Two-step estimation procedure

        Step 1: Estimate volatility dynamics from returns
        Step 2: Estimate RV parameters conditional on step 1

        Parameters
        ----------
        data : np.ndarray
            Joint data (returns and RV)
        options : EstimationOptions, optional
            Estimation options

        Returns
        -------
        dict
            Estimation results
        """
        if self.model_type != 2:
            raise ValueError("Two-step estimation only for joint models")

        if options is None:
            options = EstimationOptions()

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T

        print("Step 1: Estimating volatility from returns...")

        # Step 1: Fit returns model
        returns_model = type(self.model)(
            N=self.N,
            D=self.D,
            model_type='returns',
            leverage=self.leverage
        )

        step1_result = returns_model.fit(data[:, 0], verbose=options.verbose)

        # Extract volatility parameters
        vol_params = [
            step1_result.parameters['omega'],
            step1_result.parameters['a'],
            step1_result.parameters['b'],
            step1_result.parameters['sigma'],
            step1_result.parameters['v0']
        ]

        if self.leverage:
            vol_params.extend([
                step1_result.parameters['l'],
                step1_result.parameters['theta']
            ])

        print("Step 2: Estimating RV parameters...")

        # Step 2: Estimate RV parameters conditional on volatility
        # Fix volatility parameters and estimate only RV parameters
        def objective_step2(rv_params):
            # Combine fixed volatility params with RV params
            full_params = vol_params[:5] + list(rv_params)
            if self.leverage:
                full_params.extend(vol_params[5:])

            # Transform and evaluate
            para_tilde = self.model._nat_to_work(np.array(full_params))
            return self.model._log_likelihood(para_tilde, data)

        # Initial RV parameters
        rv_params_init = [-0.5, 0.9, -0.1, 0.05, 0.2]  # xi, varphi, delta1, delta2, shape

        # Optimize RV parameters
        rv_bounds = [(-5, 5), (0, 2), (-2, 2), (-0.5, 0.5), (0.01, 2)]

        rv_result = minimize(
            fun=objective_step2,
            x0=rv_params_init,
            bounds=rv_bounds,
            method='L-BFGS-B',
            options={'disp': options.verbose}
        )

        # Combine results
        final_params = vol_params[:5] + list(rv_result.x)
        if self.leverage:
            final_params.extend(vol_params[5:])

        final_params = np.array(final_params)

        # Process final result
        return {
            'parameters': self.model._create_param_dict(final_params),
            'parameter_array': final_params,
            'log_likelihood': -rv_result.fun,
            'step1_result': step1_result,
            'convergence': rv_result.success,
            'two_step': True
        }

    def compute_standard_errors(self, params: np.ndarray, data: np.ndarray,
                                method: str = 'hessian') -> np.ndarray:
        """
        Compute standard errors of parameter estimates

        Parameters
        ----------
        params : np.ndarray
            Parameter estimates
        data : np.ndarray
            Data used for estimation
        method : str
            'hessian' or 'bootstrap'

        Returns
        -------
        np.ndarray
            Standard errors
        """
        if method == 'hessian':
            return self._hessian_std_errors(params, data)
        elif method == 'bootstrap':
            return self._bootstrap_std_errors(params, data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _hessian_std_errors(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute standard errors using Hessian matrix"""
        from scipy.optimize import approx_fprime

        # Transform to working parameters
        para_tilde = self.model._nat_to_work(params)

        # Compute Hessian numerically
        eps = 1e-5
        n_params = len(para_tilde)
        hessian = np.zeros((n_params, n_params))

        def obj(p):
            return self.model._log_likelihood(p, data)

        for i in range(n_params):
            def grad_i(p):
                return approx_fprime(p, obj, eps)[i]

            hessian[i] = approx_fprime(para_tilde, grad_i, eps)

        # Compute covariance matrix
        try:
            cov_matrix = np.linalg.inv(hessian)
            std_errors = np.sqrt(np.diag(cov_matrix))
        except:
            warnings.warn("Hessian inversion failed")
            std_errors = np.full(n_params, np.nan)

        return std_errors