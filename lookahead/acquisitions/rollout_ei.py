import numpy as np
import multiprocessing as mp
import copy
from multiprocessing import Pool
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface
from lookahead.acquisitions.expected_improvement import ExpectedImprovement
from lookahead.acquisitions.acquisition_optimizer_bo import BayesOptAcquisitionOptimizer
import time


class RolloutEI(AcquisitionFunctionInterface):
    """
    A class for calculating the sequential rewards of MDP BayesOpt
    via parallel MC. The code is threaded, and uses the max number of processors available

    :gaussian_process: A GaussianProcess object
    :domain: The domain to optimize the inner acquisition function over
    :horizon: The time horizon
    :mc_iters: Number of Monte Carlo iterations
    :opt_mode: either 'grad' (default) or 'grid' for rollout acquisition optimization. 'grid' is faster for low-d problems.
    """

    def __init__(self, gaussian_process, opt_domain, horizon, mc_iters=int(1e2), opt_mode='grad'):
        super().__init__(gaussian_process, opt_domain)
        self.gaussian_process = gaussian_process
        self.opt_domain = opt_domain
        self.horizon = horizon
        self.mc_iters = mc_iters
        self.numthreads = int(mp.cpu_count()/2)
        self.opt_mode = opt_mode

    def evaluate_at_point_list(self, points_to_evaluate):
        num_points = points_to_evaluate.shape[0]
        rollout_values = np.zeros(num_points)
        for i in range(0, num_points):  
            rollout_values[i] = \
            self._evaluate_at_point_list(points_to_evaluate[[i], :])
        return rollout_values

    # Iterate through point list, parallelizing MC
    def _evaluate_at_point_list(self, point_to_evaluate):
        self.point_current = point_to_evaluate

        if self.numthreads > 1:
            serial_mc_iters = [int(self.mc_iters/self.numthreads)] * self.numthreads
            pool = Pool(processes=self.numthreads)
            rewards = pool.map(self._evaluate_point_at_list_serial, serial_mc_iters)
            pool.close()
            pool.join()
        else:
            rewards = self._evaluate_point_at_list_serial(self.mc_iters)

        return np.sum(rewards)/self.numthreads

    def _evaluate_point_at_list_serial(self, mc_iters):
        reward = 0
        for iters in range(0, mc_iters):
            r = self.draw_from_policy(self.point_current)
            reward += r
        return reward/mc_iters

    # Execute policy once using sequential draws from GPs
    def draw_from_policy(self, point_to_evaluate):
        
        reward = 0
        gp_temp = copy.deepcopy(self.gaussian_process)
        xi = point_to_evaluate
        h = self.horizon
        i = 0

        # Optimize acquisition function and set new sample point
        acquisition = ExpectedImprovement(gp_temp, self.opt_domain)

        while(True):
            np.random.seed(int(time.time()))
            fi = gp_temp.sample_single(xi)
            ri = self.reward(gp_temp, fi)
            reward += ri
            i = i + 1
            h = h - 1
            if h <= 0:
                break

            gp_temp.chol_update(xi, fi)
            if self.opt_mode == 'grad':
                xi = acquisition.next_point_grad()
            else:
                xi = acquisition.next_point_grid()
        return reward

    def reward(self, gp, fi):
        _, ytrain = gp.get_historical_data()
        ymin = np.min(ytrain)
        r = max(ymin - float(fi), 0)
        return r

    def next_point(self):
        optimizer = BayesOptAcquisitionOptimizer(self.gaussian_process, self.evaluate_at_point_list, self.opt_domain)
        return optimizer.get_sample_point()
