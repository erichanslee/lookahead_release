# Super simple acquisition function interface to be used
import numpy as np
import qmcpy as qp
from scipy.optimize import minimize

class AcquisitionFunctionInterface(object):
    def __init__(self, gaussian_process, opt_domain, **kwargs):
        self.gaussian_process = gaussian_process
        self.opt_domain = opt_domain
        self.opt_domain_constraints = np.array(
            [(interval.min, interval.max) for interval in self.opt_domain.get_bounding_box()]
        )
        self.horizon = 1

    # There are some seeds that could be problems for this, I think (if the point [0, 0] is in)
    def low_discrepancy_points(self, num, seed=1234):
        measure = qp.Gaussian(qp.Sobol(dimension=self.horizon, seed=seed))
        points = measure.gen_mimic_samples(n=2 ** np.ceil(np.log2(num)))[:num]
        points = np.nan_to_num(points)
        assert not np.any(np.isnan(points))
        return points

    def evaluate_at_point_list(self, points_to_evaluate):
        pass

    def joint_function_gradient_eval(self, points_to_evaluate):
        pass

    def next_point_grid(self, num_grid_points=400):
        grid_points = self.opt_domain.generate_quasi_random_points_in_domain(num_grid_points)
        vals = self.evaluate_at_point_list(grid_points)
        idx = np.argmax(vals)
        return grid_points[[idx], :]

    def next_point_grad(self, num_restarts=5):
        ymin = np.inf
        num_guess = 10 * self.opt_domain_constraints.shape[0]
        init_guesses = self.opt_domain.generate_quasi_random_points_in_domain(num_guess)
        guesses = self.evaluate_at_point_list(init_guesses)
        idx_guesses = np.argsort(guesses)
        for i in range(num_restarts):
            x0 = init_guesses[[idx_guesses[-i]], :]
            opt_result = minimize(
                fun=self.joint_function_gradient_eval,
                method='L-BFGS-B',
                x0=x0,
                jac=True,
                bounds=self.opt_domain_constraints,
                options={'maxiter': 40},
            )
            if opt_result.fun < ymin:
                x = opt_result.x
                ymin = opt_result.fun
        return x[np.newaxis, :]

    def next_point(self):
        if self.opt_domain_constraints.shape[0] > 2:
            return self.next_point_grad()
        else:
            return self.next_point_grid()
