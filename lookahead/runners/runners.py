import numpy as np
from lookahead.runners.bayesian_optimization import BayesianOptimization
from lookahead.acquisitions.expected_improvement import ExpectedImprovement
from lookahead.acquisitions.knowledge_gradient import KnowledgeGradient
from lookahead.acquisitions.ucb import UpperConfidenceBound
from lookahead.acquisitions.rollout_ei_vr import RolloutEI_VR


class RandomRunner(BayesianOptimization):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.opt_name = 'random'

    def get_next_point(self):
        return self.search_space.generate_quasi_random_points_in_domain(1)


class ExpectedImprovementRunner(BayesianOptimization):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.opt_name = 'ei'

    def get_next_point(self):
        ei = ExpectedImprovement(self.gaussian_process, self.search_space)
        return ei.next_point_grad()


class UpperConfidenceBoundRunner(BayesianOptimization):
    def __init__(self, search_space, base_kappa=1):
        super().__init__(search_space)
        self.opt_name = 'ucb' + str(base_kappa)
        self.base_kappa = base_kappa

    def get_next_point(self):
        kwargs = {'base_kappa': self.base_kappa}
        ucb = UpperConfidenceBound(self.gaussian_process, self.search_space, **kwargs)
        return ucb.next_point_grad()


class KnowledgeGradientRunner(BayesianOptimization):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.opt_name = 'kg'

    def get_next_point(self):
        kg = KnowledgeGradient(self.gaussian_process, self.search_space)
        return kg.next_point_grid()


class RolloutRunner(BayesianOptimization):
    def __init__(self, search_space, horizon):
        super().__init__(search_space)
        self.horizon = horizon
        self.opt_name = 'ei' + str(horizon)

    def get_next_point(self):
        eih = RolloutEI_VR(self.gaussian_process, self.search_space, self.horizon)
        return eih.next_point()
