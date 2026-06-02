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

        # ---- Initialise current state ---- #
        current_state = list(initial_state) if initial_state is not None else self.initialize_state()
        current_cost = self.evaluate(current_state)

        best_state = list(current_state)
        best_cost = current_cost

        evaluations: list = [current_cost]
        states_history: list = [list(current_state)]

        T = T0
        no_improve_count = 0
        
        for t in range(1, max_iterations + 1):
            T= T0 * (alpha ** t)
            if T < T_min:
                break

            neighbor =self.get_neighbor(current_state)
            neighbor_cost =self.evaluate(neighbor)

            dE = current_cost - neighbor_cost

            if dE > 0:
                current_state =list(neighbor)
                current_cost =neighbor_cost
                no_improve_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = list(current_state)
            
            else:
                acceptance_probability = math.exp(dE / T)   
                if random.random() < acceptance_probability:
                    current_state =list(neighbor)
                    current_cost = neighbor_cost
                   
                no_improve_count += 1

            evaluations.append(current_cost)
            states_history.append(list(current_state))

            
            if no_improve_count >= max_no_improve:
                break

        
        return best_state, best_cost, evaluations, states_history
