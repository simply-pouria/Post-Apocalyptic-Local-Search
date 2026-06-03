import random
from search.local_search_base import LocalSearchBase

class GeneticAlgorithm(LocalSearchBase):
    DEFAULT_POP_SIZE = 30
    DEFAULT_MAX_GEN = 500
    DEFAULT_CROSSOVER_RATE = 0.85
    DEFAULT_MUTATION_RATE = 0.20
    DEFAULT_TOURNAMENT_SIZE = 3
    DEFAULT_ELITISM_COUNT = 2
    DEFAULT_MAX_NO_IMPROVE = 100

    def _encode(self, state):
        max_sensors =self._get_max_sensors()
        chrom =list(state[:max_sensors])
        while len(chrom) < max_sensors:
            chrom.append(None)
        return chrom

    def _decode(self, chrom):
        return [pos for pos in chrom if pos is not None]

    def _repair(self, chrom):
        max_sensors =self._get_max_sensors()
        seen =set()
        valid =[]
        for pos in chrom:
            if pos is None:
                continue
            pos =tuple(pos)
            if pos in seen:
                continue
            if not self._is_position_valid(pos):
                continue
            valid.append(pos)
            seen.add(pos)
            if len(valid) >= max_sensors:
                break
        while len(valid) < max_sensors:
            valid.append(None)
        return valid

    def _create_individual(self):
        state = self.initialize_state()
        return self._encode(state)

    def _init_population(self, initial_state, pop_size):
        population =[]
        if initial_state is not None:
            population.append(self._encode(initial_state))
        while len(population) < pop_size:
            population.append(self._create_individual())
        return population

    def _tournament_select(self, population, fitness, tournament_size):
        contestants =random.sample(range(len(population)), min(tournament_size, len(population)))
        winner = min(contestants, key=lambda i: fitness[i])
        return population[winner]

    def _uniform_crossover(self, parent_a, parent_b):
        child_a, child_b = [], []
        for gene_a, gene_b in zip(parent_a, parent_b):
            if random.random() < 0.5:
                child_a.append(gene_a)
                child_b.append(gene_b)
            else:
                child_a.append(gene_b)
                child_b.append(gene_a)
        return self._repair(child_a), self._repair(child_b)

    def _mutate(self, chrom, mutation_rate):
        if random.random() > mutation_rate:
            return chrom

        filled_indices = [i for i, pos in enumerate(chrom) if pos is not None]
        empty_indices  = [i for i, pos in enumerate(chrom) if pos is None]

        ops =[]
        if filled_indices:
            ops += ["move", "remove"]
        if empty_indices:
            ops.append("add")
        if not ops:
            return chrom

        op =random.choice(ops)
        new_chrom =list(chrom)

        if op == "move":
            idx = random.choice(filled_indices)
            occupied = set(pos for pos in new_chrom if pos is not None)
            for _ in range(100):
                new_pos =tuple(self._random_valid_position())
                if new_pos not in occupied:
                    new_chrom[idx] =new_pos
                    break

        elif op == "add":
            idx =random.choice(empty_indices)
            occupied = set(pos for pos in new_chrom if pos is not None)
            for _ in range(100):
                new_pos =tuple(self._random_valid_position())
                if new_pos not in occupied:
                    new_chrom[idx] = new_pos
                    break

        elif op == "remove":
            idx = random.choice(filled_indices)
            new_chrom[idx] =None

        return self._repair(new_chrom)

    def run(self, initial_state, **kwargs):

        pop_size = int(kwargs.get("pop_size",        self.DEFAULT_POP_SIZE))
        max_generations =int(kwargs.get("max_generations", self.DEFAULT_MAX_GEN))
        crossover_rate =float(kwargs.get("crossover_rate",  self.DEFAULT_CROSSOVER_RATE))
        mutation_rate =float(kwargs.get("mutation_rate",   self.DEFAULT_MUTATION_RATE))
        tournament_size =int(kwargs.get("tournament_size", self.DEFAULT_TOURNAMENT_SIZE))
        elitism_count =int(kwargs.get("elitism_count",   self.DEFAULT_ELITISM_COUNT))
        max_no_improve =int(kwargs.get("max_no_improve",  self.DEFAULT_MAX_NO_IMPROVE))

        population =self._init_population(initial_state, pop_size)
        fitness =[self.evaluate(self._decode(chrom)) for chrom in population]

        best_idx   =min(range(pop_size), key=lambda i: fitness[i])
        best_state =list(self._decode(population[best_idx]))
        best_cost  =fitness[best_idx]

        evaluations:    list =[best_cost]
        states_history: list =[list(best_state)]
        no_improve_count =0
        
        for _gen in range(max_generations):
            
            sorted_indices =sorted(range(pop_size), key=lambda i: fitness[i])
            new_population =[list(population[i]) for i in sorted_indices[:elitism_count]]
            new_fitness =[fitness[i] for i in sorted_indices[:elitism_count]]

            while len(new_population) < pop_size:
                parent_a =self._tournament_select(population, fitness, tournament_size)
                parent_b =self._tournament_select(population, fitness, tournament_size)

                if random.random() < crossover_rate:
                    child_a, child_b =self._uniform_crossover(parent_a, parent_b)
                else:
                    child_a, child_b =list(parent_a), list(parent_b)

                child_a =self._mutate(child_a, mutation_rate)
                child_b =self._mutate(child_b, mutation_rate)

                for child in (child_a, child_b):
                    if len(new_population) < pop_size:
                        decoded = self._decode(child)
                        new_population.append(child)
                        new_fitness.append(self.evaluate(decoded))

            population =new_population
            fitness =new_fitness

            gen_best_idx =min(range(pop_size), key=lambda i: fitness[i])
            gen_best_cost =fitness[gen_best_idx]
            gen_best_state =list(self._decode(population[gen_best_idx]))

            if gen_best_cost < best_cost:
                best_cost =gen_best_cost
                best_state =gen_best_state
                no_improve_count =0
            else:
                no_improve_count += 1

            evaluations.append(best_cost)
            states_history.append(list(best_state))

            if no_improve_count >= max_no_improve:
                break

        return best_state, best_cost, evaluations, states_history

