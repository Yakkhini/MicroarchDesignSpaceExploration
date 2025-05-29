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


class RandomSearchOptimizer(AbstractOptimizer):
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
        self.num_of_variables = len(self.design_space.idx_to_vec(1))

        self.variables_upper_bound_list = []

        for unit_info_values in self.design_space.components_mappings.values():
            self.variables_upper_bound_list.append(int(unit_info_values.keys()[-1]))

        self.problem = MOOWrappedProblem(
            num_of_objectives=self.num_of_objectives,
            num_of_variables=self.num_of_variables,
            x_upper_bound_list=self.variables_upper_bound_list,
        )

        self.veriable_type = int

        self.n_suggestions = 20

        self.algorithm = NSGA2(pop_size=self.n_suggestions)

        self.algorithm.setup(
            self.problem, termination=("n_gen", 10), seed=1, verbose=False
        )

    def suggest(self):
        """
        get a suggestion from the optimizer.

        returns
        -------
        next_guess: <list> of <list>
            list of `self.n_suggestions` suggestion(s).
            each suggestion is a microarchitecture embedding.
        """
        try:
            raise NotImplementedError("optimizer is not implemented yet.")

        except AssertionError:
            x_guess = np.random.choice(
                range(1, self.design_space.size + 1), size=self.n_suggestions
            )
            potential_suggest = [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                )
                for _x_guess in x_guess
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
        pass


if __name__ == "__main__":
    experiment(RandomSearchOptimizer)
