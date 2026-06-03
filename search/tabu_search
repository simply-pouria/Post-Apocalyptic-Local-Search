import random

from search.local_search_base import LocalSearchBase


class TabuSearch(LocalSearchBase):

    DEFAULT_MAX_ITER =1000
    DEFAULT_TABU_TENURE =15         
    DEFAULT_NEIGHBORS_PER_ITER =20   
    DEFAULT_MAX_NO_IMPROVE =200     
    DEFAULT_MAX_NO_IMPROVE_RESTART =50 

    @staticmethod
    def _move_key(old_state, new_state):
        old_set =set(tuple(p) for p in old_state)
        new_set =set(tuple(p) for p in new_state)
        added   =new_set - old_set
        removed =old_set - new_set

        if added and removed:
            old_pos = min(removed) 
            new_pos = min(added)
            return ("move", (old_pos, new_pos))
        elif added:
            return ("add", min(added))
        elif removed:
            return ("remove", min(removed))
        else:
            return ("noop", ())

    def _generate_neighbours(self, state, n):
        neighbours =[]
        state_key =tuple(sorted(tuple(p) for p in state))
        seen_keys ={state_key}

        for _ in range(n * 5):   
            if len(neighbours) >= n:
                break
            nb =self.get_neighbor(state)
            nb_key = tuple(sorted(tuple(p) for p in nb))
            if nb_key not in seen_keys:
                seen_keys.add(nb_key)
                neighbours.append(nb)

        return neighbours

    def run(self, initial_state, **kwargs):
        max_iterations =int(kwargs.get("max_iterations",         self.DEFAULT_MAX_ITER))
        tabu_tenure =int(kwargs.get("tabu_tenure",            self.DEFAULT_TABU_TENURE))
        neighbors_per_iter =int(kwargs.get("neighbors_per_iter",     self.DEFAULT_NEIGHBORS_PER_ITER))
        max_no_improve =int(kwargs.get("max_no_improve",         self.DEFAULT_MAX_NO_IMPROVE))
        max_no_improve_restart =int(kwargs.get("max_no_improve_restart", self.DEFAULT_MAX_NO_IMPROVE_RESTART))

        current_state =list(initial_state) if initial_state is not None else self.initialize_state()
        current_cost  =self.evaluate(current_state)

        best_state =list(current_state)
        best_cost  =current_cost

        evaluations: list =[current_cost]
        states_history: list =[list(current_state)]

        tabu_list: dict ={}
        no_improve_count =0
        no_improve_restart_count =0

        for iteration in range(1, max_iterations + 1):

            expired =[key for key, expiry in tabu_list.items() if expiry <= iteration]
            for key in expired:
                del tabu_list[key]


            neighbours =self._generate_neighbours(current_state, neighbors_per_iter)

            best_candidate = None
            best_candidate_cost = float("inf")
            best_candidate_move = None

            for nb in neighbours:
                nb_cost =self.evaluate(nb)
                move_key =self._move_key(current_state, nb)

                is_tabu =move_key in tabu_list

                aspiration =nb_cost < best_cost

                if (not is_tabu) or aspiration:
                    if nb_cost < best_candidate_cost:
                        best_candidate = nb
                        best_candidate_cost = nb_cost
                        best_candidate_move = move_key

            if best_candidate is None:
                evaluations.append(current_cost)
                states_history.append(list(current_state))
                no_improve_count += 1
                no_improve_restart_count += 1
            else:
                current_state = list(best_candidate)
                current_cost  = best_candidate_cost

                # Add move to tabu list
                if best_candidate_move is not None:
                    tabu_list[best_candidate_move] = iteration + tabu_tenure

                if current_cost < best_cost:
                    best_cost  = current_cost
                    best_state = list(current_state)
                    no_improve_count         = 0
                    no_improve_restart_count = 0
                else:
                    no_improve_count         += 1
                    no_improve_restart_count += 1

                evaluations.append(current_cost)
                states_history.append(list(current_state))

            if no_improve_restart_count >= max_no_improve_restart:
                current_state = self.initialize_state()
                current_cost  = self.evaluate(current_state)
                tabu_list.clear()
                no_improve_restart_count = 0

            if no_improve_count >= max_no_improve:
                break

        return best_state, best_cost, evaluations, states_history
