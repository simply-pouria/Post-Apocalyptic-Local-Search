import math
import random

from search.local_search_base import LocalSearchBase


class SimulatedAnnealing(LocalSearchBase):

    DEFAULT_T0 = 500.0         
    DEFAULT_ALPHA = 0.995       
    DEFAULT_T_MIN = 0.01       
    DEFAULT_MAX_ITER = 10_000  
    DEFAULT_MAX_NO_IMPROVE = 500  
    def run(self, initial_state, **kwargs):
        
        """
        TODO: Implement the Simulated Annealing algorithm.
        
        Parameters
        ----------
        initial_state : list of tuples
            The initial configuration of sensors.
        **kwargs : 
            Define and add all necessary parameters required for Simulated Annealing 

        Returns
        -------
        best_state : list of tuples
        best_cost : int or float
        evaluations : list
        states_history : list of lists
        """
        # ---- Hyper-parameters ---- #
        T0 = float(kwargs.get("T0", self.DEFAULT_T0))
        alpha = float(kwargs.get("alpha", self.DEFAULT_ALPHA))
        T_min = float(kwargs.get("T_min", self.DEFAULT_T_MIN))
        max_iterations = int(kwargs.get("max_iterations", self.DEFAULT_MAX_ITER))
        max_no_improve = int(kwargs.get("max_no_improve", self.DEFAULT_MAX_NO_IMPROVE))

        # ---- Initialise current state --------------------------------- #
        current_state = list(initial_state) if initial_state is not None else self.initialize_state()
        current_cost = self.evaluate(current_state)

        best_state = list(current_state)
        best_cost = current_cost

        evaluations: list = [current_cost]
        states_history: list = [list(current_state)]

        T = T0
        no_improve_count = 0

        
        raise NotImplementedError("Students must implement this method.")
