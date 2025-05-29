"""
    random search optimizer performs random search.
    the code is only for example demonstration.
    ``` 
        python random-search-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    you can specify more options to test your optimizer. please use
    ```
        python random-search-optimizer.py -h
    ```
    to check.
"""

import numpy as np
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems.static import StaticProblem


class MOOWrappedProblem(Problem):
    def __init__(self, num_of_objectives, num_of_variables, x_upper_bound_list):
        super().__init__(
            n_var=num_of_variables,
            n_obj=num_of_objectives,
            xl=np.ones(num_of_variables),
            xu=x_upper_bound_list,
            vtype=int,
        )


class NSGA2Optimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
        build a wrapper class for an optimizer.

        parameters
        ----------
        design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)

        self.num_of_objectives = 3

        self.construct_variable_choose_info()
        print("Variables upper bound list:", self.variable_upper_bound_list)

        termination = NoTermination()

        self.problem = MOOWrappedProblem(
            num_of_objectives=self.num_of_objectives,
            num_of_variables=self.num_of_variables,
            x_upper_bound_list=self.variable_upper_bound_list,
        )

        self.n_suggestions = 20

        self.algorithm = NSGA2(
            pop_size=self.n_suggestions,
            sampling=IntegerRandomSampling(),
            crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
            mutation=PM(eta=20, repair=RoundingRepair()),
        )

        self.algorithm.setup(
            self.problem, termination=termination, seed=1, verbose=False
        )

    def construct_variable_choose_info(self):
        components = self.design_space.components
        component_dims = self.design_space.component_dims
        component_offset = self.design_space.component_offset
        print("components:", components)
        print("component_dims:", component_dims)
        print("component_offset:", component_offset)

        # Use tree data structure path to represent a kind of valid microarchitecture.
        # So `root_out_degree` is the number of subdesigns, not components.
        self.root_out_degree = len(component_dims)

        # Need to make a choice for every component in a subdesign
        self.variable_choose_mod_list = np.transpose(np.array(component_dims)).tolist()
        self.variable_choose_lcm_list = [
            int(np.lcm.reduce(x)) for x in self.variable_choose_mod_list
        ]

        print("Variable choose mod list:", self.variable_choose_mod_list)
        print("Variable choose lcm list:", self.variable_choose_lcm_list)

        self.variable_upper_bound_list = self.variable_choose_lcm_list
        self.variable_upper_bound_list.insert(0, self.root_out_degree)
        self.num_of_variables = len(self.variable_upper_bound_list)

    def variables_to_vector(self, variables):
        vec = [
            self.design_space.component_offset[variables[0] - 1][i]
            + int((element % self.variable_choose_mod_list[i][variables[0] - 1]))
            for i, element in enumerate(variables[1:])
        ]

        return vec

    def suggest(self):
        """
        get a suggestion from the optimizer.

        returns
        -------
        next_guess: <list> of <list>
            list of `self.n_suggestions` suggestion(s).
            each suggestion is a microarchitecture embedding.
        """
        self.pop = self.algorithm.ask()
        potential_suggest = [
            self.design_space.vec_to_microarchitecture_embedding(
                self.variables_to_vector(x)
            )
            for x in self.pop.get("X")
        ]

        return potential_suggest

    def observe(self, x, y):
        """
        send an observation of a suggestion back to the optimizer.

        parameters
        ----------
        x: <list> of <list>
            the output of `suggest`.
        y: <list> of <list>
            corresponding values where each `x` is mapped to.
        """
        problem = self.problem

        y_np_array = np.array(y)

        f1 = np.negative(y_np_array[:, 0])  # Performance
        f2 = y_np_array[:, 1]  # Power
        f3 = y_np_array[:, 2]  # Area

        F = np.column_stack((f1, f2, f3))

        static = StaticProblem(problem, F=F)
        Evaluator().eval(static, self.pop)

        self.algorithm.tell(infills=self.pop)


if __name__ == "__main__":
    experiment(NSGA2Optimizer)
