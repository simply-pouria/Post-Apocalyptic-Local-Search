from search.local_search_base import LocalSearchBase


class HillClimbing(LocalSearchBase):
    def run(self, initial_state, **kwargs):
        """
        Random Restart Hill Climbing.

        This variant runs normal hill climbing multiple times from different
        starting states. which according to the slides is better than stochastic hill climbing and also it was easier
        to implement:)
        """

        max_iterations = kwargs.get("max_iterations", 1000)
        restarts = kwargs.get("restarts", 10)
        max_no_improve = kwargs.get("max_no_improve", 100)

        if restarts <= 0:
            restarts = 1

        iterations_per_restart = max(1, max_iterations // restarts)

        best_state = None
        best_cost = float("inf")

        evaluations = []
        states_history = []

        for restart_index in range(restarts):
            # First run starts from the shared initial state
            # Later runs start from new random states
            if restart_index == 0 and initial_state is not None:
                current_state = list(initial_state)
            else:
                current_state = self.initialize_state()

            current_cost = self.evaluate(current_state)

            # Record restart strting point
            evaluations.append(current_cost)
            states_history.append(list(current_state))

            if current_cost < best_cost:
                best_cost = current_cost
                best_state = list(current_state)

            no_improve_count = 0

            for _ in range(iterations_per_restart):
                neighbor = self.get_neighbor(current_state)
                neighbor_cost = self.evaluate(neighbor)

                # hill climbing inside each restart:
                if neighbor_cost < current_cost:
                    current_state = list(neighbor)
                    current_cost = neighbor_cost
                    no_improve_count = 0

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_state = list(current_state)
                else:
                    no_improve_count += 1

                # Record the accepted current state, not every rejected neighbour.
                evaluations.append(current_cost)
                states_history.append(list(current_state))

                # This restart is stuck, so move to the next restart.
                if no_improve_count >= max_no_improve:
                    break

        return best_state, best_cost, evaluations, states_history

