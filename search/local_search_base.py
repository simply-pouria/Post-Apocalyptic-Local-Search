import random


class LocalSearchBase:
    def __init__(self, world):
        self.world = world

    def _get_rows(self):
        return getattr(self.world, "rows", getattr(self.world, "row", None))

    def _get_cols(self):
        return getattr(self.world, "cols", getattr(self.world, "col", None))

    def _get_max_sensors(self):
        return getattr(
            self.world,
            "max_sensors",
            getattr(self.world, "sensors_max", 0)
        )

    def _get_sensor_range(self):
        return getattr(
            self.world,
            "sensor_range",
            getattr(self.world, "range_sensor", 1)
        )

    def _get_targets(self):
        if hasattr(self.world, "get_targets"):
            return list(self.world.get_targets())

        if hasattr(self.world, "targets"):
            return list(self.world.targets)

        return []

    def _is_position_valid(self, position):
        try:
            x, y = position
        except (TypeError, ValueError):
            return False

        if hasattr(self.world, "is_position_valid"):
            try:
                return self.world.is_position_valid(position)
            except TypeError:
                return self.world.is_position_valid(x, y)

        if hasattr(self.world, "position_valid_is"):
            try:
                return self.world.position_valid_is(position)
            except TypeError:
                return self.world.position_valid_is(x, y)

        if hasattr(self.world, "is_valid_position"):
            try:
                return self.world.is_valid_position(position)
            except TypeError:
                return self.world.is_valid_position(x, y)

        rows = self._get_rows()
        cols = self._get_cols()

        if rows is None or cols is None:
            return False

        if x < 0 or x >= rows or y < 0 or y >= cols:
            return False

        if hasattr(self.world, "obstacles"):
            if position in self.world.obstacles:
                return False

        return True
    def _random_valid_position(self):
        if hasattr(self.world, "random_position"):
            return self.world.random_position()

        if hasattr(self.world, "position_random"):
            return self.world.position_random()

        rows = self._get_rows()
        cols = self._get_cols()

        if rows is None or cols is None:
            raise ValueError("World must have rows and cols attributes.")

        for _ in range(1000):
            position = (
                random.randint(0, rows - 1),
                random.randint(0, cols - 1)
            )

            if self._is_position_valid(position):
                return position

        valid_positions = []

        for x in range(rows):
            for y in range(cols):
                position = (x, y)
                if self._is_position_valid(position):
                    valid_positions.append(position)

        if not valid_positions:
            raise ValueError("No valid position exists in this world.")

        return random.choice(valid_positions)

    def _distance_squared(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def _is_target_covered_by_sensor(self, target, sensor):
        sensor_range = self._get_sensor_range()
        return self._distance_squared(target, sensor) <= sensor_range ** 2

    def _clean_state(self, state):
        clean = []
        seen = set()

        for position in state:
            position = tuple(position)

            if position in seen:
                continue

            if not self._is_position_valid(position):
                continue

            clean.append(position)
            seen.add(position)

        max_sensors = self._get_max_sensors()
        return clean[:max_sensors]

    def _state_is_valid(self, state):
        max_sensors = self._get_max_sensors()

        if len(state) > max_sensors:
            return False

        if len(state) != len(set(state)):
            return False

        for position in state:
            if not self._is_position_valid(position):
                return False

        return True

    def evaluate(self, state):
        if state is None:
            return 10 ** 9

        state = [tuple(position) for position in state]
        targets = self._get_targets()
        max_sensors = self._get_max_sensors()

        invalid_penalty = 0

        if len(state) > max_sensors:
            invalid_penalty += (len(state) - max_sensors) * 10000

        duplicate_count = len(state) - len(set(state))
        invalid_penalty += duplicate_count * 10000

        for sensor in state:
            if not self._is_position_valid(sensor):
                invalid_penalty += 10000

        valid_sensors = self._clean_state(state)

        covered_targets = set()
        redundant_coverage = 0

        for target in targets:
            coverage_count = 0

            for sensor in valid_sensors:
                if self._is_target_covered_by_sensor(target, sensor):
                    coverage_count += 1

            if coverage_count > 0:
                covered_targets.add(target)

            if coverage_count > 1:
                redundant_coverage += coverage_count - 1

        uncovered_targets = len(targets) - len(covered_targets)

        uncovered_cost = uncovered_targets * 1000
        sensor_usage_cost = len(valid_sensors) * 5
        redundancy_cost = redundant_coverage * 2

        total_cost = (
            uncovered_cost
            + sensor_usage_cost
            + redundancy_cost
            + invalid_penalty
        )

        return total_cost

    def initialize_state(self):
        max_sensors = self._get_max_sensors()

        if max_sensors <= 0:
            return []

        initial_state = []
        seen = set()

        attempts = 0
        max_attempts = max_sensors * 100

        while len(initial_state) < max_sensors and attempts < max_attempts:
            position = tuple(self._random_valid_position())

            if position not in seen and self._is_position_valid(position):
                initial_state.append(position)
                seen.add(position)

            attempts += 1

        return initial_state

    def get_neighbor(self, state):
        current_state = self._clean_state(state)
        max_sensors = self._get_max_sensors()

        possible_operations = []

        if len(current_state) > 0:
            possible_operations.append("move")
            possible_operations.append("remove")

        if len(current_state) < max_sensors:
            possible_operations.append("add")

        if not possible_operations:
            return current_state[:]

        operation = random.choice(possible_operations)

        if operation == "move":
            neighbor = current_state[:]
            sensor_index = random.randint(0, len(neighbor) - 1)

            for _ in range(100):
                new_position = tuple(self._random_valid_position())

                if new_position not in neighbor:
                    neighbor[sensor_index] = new_position

                    if self._state_is_valid(neighbor):
                        return neighbor

            return current_state[:]

        if operation == "add":
            neighbor = current_state[:]

            for _ in range(100):
                new_position = tuple(self._random_valid_position())

                if new_position not in neighbor:
                    neighbor.append(new_position)

                    if self._state_is_valid(neighbor):
                        return neighbor

                    neighbor.pop()

            return current_state[:]

        if operation == "remove":
            neighbor = current_state[:]
            sensor_index = random.randint(0, len(neighbor) - 1)
            neighbor.pop(sensor_index)

            if self._state_is_valid(neighbor):
                return neighbor

            return current_state[:]

        return current_state[:]