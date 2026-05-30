from search.local_search_base import LocalSearchBase

class HillClimbing(LocalSearchBase):
    def run(self, initial_state, **kwargs):
        """
        ~~TODO: Implement the Hill Climbing algorithm.~~ done:)

        Parameters
        ----------
        initial_state : list of tuples
            The initial configuration of sensors.
        **kwargs :
            Define and add any other parameters you might need for the algorithm

        Returns
        -------
        best_state : list of tuples
            The best configuration found.
        best_cost : int or float
            The cost of the best configuration.
        evaluations : list
            List of costs at each iteration (used for plotting).
        states_history : list of lists
            List of states at each iteration (used for animation).
        """

        # get settings or fall back to defaults
        max_iterations = kwargs.get("max_iterations", 1000)
        max_no_improve = kwargs.get("max_no_improve", 100)

        current_state = list(initial_state) if initial_state is not None else []
        current_cost = self.evaluate(current_state)
        best_state = list(current_state)
        best_cost = current_cost

        # history
        evaluations = [current_cost]
        states_history = [list(current_state)]
        no_improve_count = 0

        for _ in range(max_iterations):
            neighbour = self.get_neighbor(current_state)
            neighbour_cost = self.evaluate(neighbour)

            evaluations.append(neighbour_cost)
            states_history.append(list(neighbour))

            # accepting better neighbors
            if neighbour_cost < current_cost:
                current_state = list(neighbour)
                current_cost = neighbour_cost
                no_improve_count = 0
                # update best state
                if neighbour_cost < best_cost:
                    best_cost = neighbour_cost
                    best_state = list(neighbour)
            else:
                no_improve_count += 1

            if no_improve_count >= max_no_improve:
                break

        return best_state, best_cost, evaluations, states_history

