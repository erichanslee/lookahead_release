import numpy as np
import multiprocessing as mp
import copy
from multiprocessing import Pool
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface
from lookahead.acquisitions.expected_improvement import ExpectedImprovement
from lookahead.acquisitions.probability_improvement import ProbabilityImprovement
from lookahead.acquisitions.acquisition_optimizer_bo import BayesOptAcquisitionOptimizer


class RolloutEI_VR(AcquisitionFunctionInterface):
    """
    A class for calculating the sequential rewards of MDP BayesOpt
    via parallel MC. The code is threaded, and uses the max number of processors available
    Variance reduction techniques used:
    QMC, CRN, EI, PI Control Variates

    :gaussian_process: A GaussianProcess object
    :domain: The domain to optimize the inner acquisition function over
    :horizon: The time horizon
    :mc_iters: Number of Monte Carlo iterations
    :opt_mode: either 'grad' (default) or 'grid' for rollout acquisition optimization. 'grid' is faster for low-d problems.
    :grid_size: size of rollout grid, not used if opt_mode is 'grad'
    """

    def __init__(
            self, gaussian_process, opt_domain, horizon, mc_iters=int(1e3),
            opt_mode='grad', grid_size=400, seed=0
    ):

        super().__init__(gaussian_process, opt_domain)
        self.gaussian_process = gaussian_process
        self.opt_domain = opt_domain
        self.horizon = horizon
        self.mc_iters = mc_iters
        self.numthreads = int(mp.cpu_count()/2)
        self.opt_mode = opt_mode
        self.grid_size = grid_size
        self.seed = seed

        # Control Variates
        self.ei = ExpectedImprovement(gaussian_process, opt_domain)
        self.pi = ProbabilityImprovement(gaussian_process, opt_domain)

    def evaluate_at_point_list(self, points_to_evaluate):
        num_points = points_to_evaluate.shape[0]
        rollout_values = np.zeros(num_points)
        for i in range(0, num_points):
            rollout_values[i] = self._evaluate_at_point_list(points_to_evaluate[[i], :])
        return rollout_values

    # Iterate through point list, parallelizing MC
    def _evaluate_at_point_list(self, point_to_evaluate):

        self.point_current = point_to_evaluate
        random_number_stream = self.low_discrepancy_points(self.mc_iters, seed=self.seed)
        pool = Pool(processes=self.numthreads)
        rewards = pool.map(self.draw_from_policy, random_number_stream)
        pool.close()
        pool.join()

        # Sum up samples to get estimate
        value_variates = np.sum(rewards) / self.mc_iters

        # Remove effects of control variates
        ei_value = self.ei.evaluate_at_point_list(point_to_evaluate)
        pi_value = self.pi.evaluate_at_point_list(point_to_evaluate)
        value_corrected = value_variates + ei_value + pi_value
        return value_corrected

    # Execute policy once using sequential draws from GPs
    def draw_from_policy(self, random_number_list):
        """
        Performs a draw from the policy, given a fixed random series of numbers
        :param random_number_list: fixed random series of numbers, of size h, assumed unit normal distributed
        :return:
        """
        assert self.horizon > 1, 'Horizon must be greater than 1 to calculate MC'

        gp_temp = copy.deepcopy(self.gaussian_process)
        xi = self.point_current
        h = self.horizon
        rewards = np.zeros(h)

        # Optimize acquisition function and set new sample point
        acquisition = ExpectedImprovement(gp_temp, self.opt_domain)

        for i in range(self.horizon):
            fi = gp_temp.sample_single(xi, random_number_list[i])

            if i < self.horizon - 1:
                rewards[i] = self.reward(gp_temp, fi)
            else:
                rewards[i] = acquisition.evaluate_at_point_list(xi)

            gp_temp.chol_update(xi, fi)
            if self.opt_mode == 'grad':
                xi = acquisition.next_point_grad()
            else:
                xi = acquisition.next_point_grid(self.grid_size)


        # Add in control variates (EI, PI)... in this case, an EI control variate
        # removes the first term, and a PI control variate adds 1 if there is
        # improvement in the first sample
        r = np.sum(rewards)
        r -= rewards[0] # EI Control Variate
        if rewards[0] > 0:
            r -= 1
        return r

    def reward(self, gp, fi):
        _, ytrain = gp.get_historical_data()
        ymin = np.min(ytrain)
        r = max(ymin - float(fi), 0)
        return r

    def next_point(self):
        """
        If self horizon is 1, just return the max of regular EI
        """
        if self.horizon == 1:
            ei = ExpectedImprovement(self.gaussian_process, self.opt_domain)
            return ei.next_point_grid()
        else:
            optimizer = BayesOptAcquisitionOptimizer(self.gaussian_process, self.evaluate_at_point_list, self.opt_domain)
            return optimizer.get_sample_point()

