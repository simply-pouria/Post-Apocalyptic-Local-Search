from search.local_search_base import LocalSearchBase


class BeamSearch(LocalSearchBase):
    def run(self, initial_state, **kwargs):
        """
        Local Beam Search for the sensor placement problem.

        """

        beam_width = kwargs.get("beam_width", 5)
        max_iterations = kwargs.get("max_iterations", 1000)
        neighbors_per_state = kwargs.get("neighbors_per_state", 5)
        max_no_improve = kwargs.get("max_no_improve", 100)

        if beam_width <= 0:
            beam_width = 1

        if neighbors_per_state <= 0:
            neighbors_per_state = 1

        #for removing duplicate states
        def state_key(state):
            return tuple(sorted(tuple(position) for position in state))

        beam = []

        if initial_state is not None:
            clean_initial = self._clean_state(initial_state)
            beam.append(clean_initial)

        while len(beam) < beam_width:
            new_state = self.initialize_state()
            new_key = state_key(new_state)

            duplicate = False
            for existing_state in beam:
                if state_key(existing_state) == new_key:
                    duplicate = True
                    break

            if not duplicate:
                beam.append(new_state)

        # evaluate initial beam
        beam_costs = [self.evaluate(state) for state in beam]

        best_index = min(range(len(beam)), key=lambda i: beam_costs[i])
        best_state = list(beam[best_index])
        best_cost = beam_costs[best_index]

        evaluations = [best_cost]
        states_history = [list(best_state)]

        no_improve_count = 0

        for _ in range(max_iterations):
            candidates = []

            # Keep current beam states as candidates too
            for state in beam:
                candidates.append(state)

            for state in beam:
                for _ in range(neighbors_per_state):
                    neighbor = self.get_neighbor(state)
                    candidates.append(neighbor)

            # remove duplicate states
            unique_candidates = []
            seen = set()

            for state in candidates:
                clean_state = self._clean_state(state)
                key = state_key(clean_state)

                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(clean_state)

            scored_candidates = [
                (self.evaluate(state), state)
                for state in unique_candidates
            ]

            scored_candidates.sort(key=lambda item: item[0])

            selected = scored_candidates[:beam_width]

            beam_costs = [cost for cost, state in selected]
            beam = [state for cost, state in selected]

            current_best_cost = beam_costs[0]
            current_best_state = beam[0]

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_state = list(current_best_state)
                no_improve_count = 0
            else:
                no_improve_count += 1

            evaluations.append(best_cost)
            states_history.append(list(best_state))

            if no_improve_count >= max_no_improve:
                break

        return best_state, best_cost, evaluations, states_history
