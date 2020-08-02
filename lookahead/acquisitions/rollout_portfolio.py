import numpy as np
import multiprocessing as mp
import copy
from multiprocessing import Pool
from lookahead.acquisitions.acquisition_function_interface import AcquisitionFunctionInterface
from lookahead.acquisitions.expected_improvement import ExpectedImprovement
from lookahead.acquisitions.knowledge_gradient import KnowledgeGradient
from lookahead.acquisitions.ucb import UpperConfidenceBound
MC_ITERS = 200

# The portfolio of acquisiton functions is a dict mapping acquisitions to a list of kwargs to consider. We consider
# EI and the UCB class of acquisition functions.
PORTFOLIO = {
    ExpectedImprovement: [{}],
    UpperConfidenceBound: [{'base_kappa': 0}, {'base_kappa': 1},
                           {'base_kappa': 2}, {'base_kappa': 4}, {'base_kappa': 8}],
    KnowledgeGradient: [{}]
}

class RolloutPortfolio(AcquisitionFunctionInterface):
    """
    Choosing an acquisition function with Rollout. Consider: UCB for constants 1 through 5, EI, KG
    """

    def __init__(self, gaussian_process, opt_domain, horizon):
        assert horizon > 0, 'Horizon must be greater than 0'
        super().__init__(gaussian_process, opt_domain)
        self.horizon = horizon
        self.numthreads = int(mp.cpu_count() / 2)

    def next_point(self):
        x_next = None
        acquisition_chosen = None
        reward_max = -np.inf
        for portfolio_element, kwargs_list in PORTFOLIO.items():
            for kwargs in kwargs_list:
                acquisition = portfolio_element(self.gaussian_process, self.opt_domain, **kwargs)
                x_opt = acquisition.next_point()

                # Roll out EI on KG argmax first to see what happens!
                if portfolio_element is KnowledgeGradient:
                    acquisition_rollout = RolloutPortfolioElement(self.gaussian_process, self.opt_domain,
                                                              self.horizon, ExpectedImprovement, **kwargs)
                else:
                    acquisition_rollout = RolloutPortfolioElement(self.gaussian_process, self.opt_domain,
                                                              self.horizon, portfolio_element, **kwargs)

                reward = acquisition_rollout.evaluate_at_point_list(x_opt)
                if reward > reward_max:
                    acquisition_chosen = str(portfolio_element.__name__)
                    if kwargs:
                        acquisition_chosen += str(kwargs['base_kappa'])
                    x_next = x_opt
                    reward_max = reward
        return x_next, acquisition_chosen


class RolloutPortfolioEI(AcquisitionFunctionInterface):
    """
    Chooses an acquisition function with Rollout. Consider: UCB for constants 1 through 5, EI, KG
    """

    def __init__(self, gaussian_process, opt_domain, horizon):
        assert horizon > 0, 'Horizon must be greater than 0'
        super().__init__(gaussian_process, opt_domain)
        self.horizon = horizon
        self.numthreads = int(mp.cpu_count() / 2)

    def next_point(self):
        x_next = None
        acquisition_chosen = None
        reward_max = -np.inf
        for portfolio_element, kwargs_list in PORTFOLIO.items():
            for kwargs in kwargs_list:
                acquisition = portfolio_element(self.gaussian_process, self.opt_domain, **kwargs)
                x_opt = acquisition.next_point()

                # Roll out EI on KG argmax first to see what happens!
                acquisition_rollout = RolloutPortfolioElement(self.gaussian_process, self.opt_domain,
                                                              self.horizon, ExpectedImprovement, **kwargs)
                reward = acquisition_rollout.evaluate_at_point_list(x_opt)
                if reward > reward_max:
                    acquisition_chosen = str(portfolio_element.__name__)
                    if kwargs:
                        acquisition_chosen += str(kwargs['base_kappa'])
                    x_next = x_opt
                    reward_max = reward
        return x_next, acquisition_chosen

class RolloutPortfolioElement(AcquisitionFunctionInterface):
    """
    Rolls out a specific acquisition function
    """

    def __init__(self, gaussian_process, opt_domain, horizon, acquisition, **kwargs):
        super().__init__(gaussian_process, opt_domain)
        assert horizon > 0, 'Horizon must be greater than 0'
        self.horizon = horizon
        self.mc_iters = MC_ITERS
        self.numthreads = 4
        self.acquisition = acquisition

    def evaluate_at_point_list(self, points_to_evaluate):
        num_points = points_to_evaluate.shape[0]
        rollout_values = np.zeros(num_points)
        for i in range(0, num_points):
            rollout_values[i] = self._evaluate_at_point_list(points_to_evaluate[[i], :])
        return rollout_values

    # Iterate through point list, parallelizing MC
    def _evaluate_at_point_list(self, point_to_evaluate):
        self.point_current = point_to_evaluate
        random_number_stream = self.low_discrepancy_points(self.mc_iters)
        pool = Pool(processes=self.numthreads)
        rewards = pool.map(self.draw_from_policy, random_number_stream)
        pool.close()
        pool.join()

        # Sum up samples to get estimate
        value_variates = np.sum(rewards) / self.mc_iters
        return value_variates

    def random_points(self, num):
        """
        Random points distributed w.r.t. unit gaussian, fixed according to a seed.
        """
        np.random.seed(1234)
        return list(np.random.randn(num, self.horizon))

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
        acquisition_ei = ExpectedImprovement(gp_temp, self.opt_domain)
        acquisition = self.acquisition(gp_temp, self.opt_domain)

        for i in range(self.horizon):
            fi = gp_temp.sample_single(xi, random_number_list[i])

            if i < self.horizon - 1:
                rewards[i] = self.reward(gp_temp, fi)
            else:
                rewards[i] = acquisition_ei.evaluate_at_point_list(xi)
            gp_temp.chol_update(xi, fi)
            xi = acquisition.next_point()

        r = np.sum(rewards)
        return r

    def reward(self, gp, fi):
        _, ytrain = gp.get_historical_data()
        ymin = np.min(ytrain)
        r = max(ymin - float(fi), 0)
        return r