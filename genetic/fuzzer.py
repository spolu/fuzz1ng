import argparse
import copy
import random
import os
import time
import typing

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.log import Log
from utils.runner import Runner
from utils.runs_db import RunsDB

TRANSITION_ENERGY = 6.0
INPUT_ENERGY = 1.0

REPRODUCTION_ENERGY_FACTOR = 3.0

INSERTION_PROBABILITY = 0.01
BYTE_FLIP_PROBABILITY = 0.1
DELETION_PROBABILITY = 0.005

INPUT_MAX_AGE = 15


class Input:
    def __init__(
            self,
            input: typing.List[int],
            length: int,
            energy: float,
    ) -> None:
        self._input = input
        self._length = length
        self._energy = energy
        self._age = 0

    def input(
            self,
    ) -> typing.List[int]:
        return self._input

    def length(
            self,
    ) -> int:
        return self._length

    def energy(
            self,
    ) -> float:
        return self._energy

    def receive_energy(
            self,
            delta: float,
    ) -> float:
        self._energy += delta
        return self._energy

    def age(
            self,
    ) -> int:
        self._age += 1
        return self._age

    def alive(
            self,
    ) -> bool:
        return self._energy > 0 and self._age < INPUT_MAX_AGE


class GeneticFuzzer:
    def __init__(
            self,
            config: Config,
            runner: Runner,
            runs_db: RunsDB,
    ) -> None:
        if config.get('tensorboard_log_dir') is not None:
            self.tb_writer = SummaryWriter(config.get('tensorboard_log_dir'))

        self._runner = runner
        self._runs_db = runs_db

        self._population = [Input(
            [self._runner.eof()]*self._runner.input_length(), 0, 10,
        )]

        self._cycle_count = 0

    def cycle(
            self,
    ) -> None:
        start_time = time.time()

        coverages, _, generation = self._runner.run(
            [i.input() for i in self._population],
        )

        for i in range(len(self._population)):
            self._runs_db.store(self._population[i].input(), coverages[i])

        add = []
        remove = []

        for i in range(len(self._population)):
            delta = - (self._population[i].length() + 1) * INPUT_ENERGY
            for t in coverages[i].transitions:
                delta += TRANSITION_ENERGY * \
                    coverages[i].transitions[t] / generation.transitions[t]
            self._population[i].receive_energy(delta)

            if not self._population[i].alive():
                remove.append(self._population[i])
            else:
                self._population[i].age()

            if self._population[i].energy() >= \
                    REPRODUCTION_ENERGY_FACTOR * \
                    self._population[i].length() * INPUT_ENERGY:
                add.append(self.reproduce(self._population[i]))

        self._population += add
        for i in remove:
            self._population.remove(i)

        run_time = time.time() - start_time
        self._cycle_count += 1

        Log.out("Cycle done", {
            "cycle_count": self._cycle_count,
            "run_count": self._runs_db.run_count(),
            "population_count": len(self._population),
            "remove_count": len(remove),
            "add_count": len(add),
            "run_time": '%.2f' % (run_time),
            "unique_skip_pathes": self._runs_db.unique_skip_path_count(),
            "unique_count_pathes": self._runs_db.unique_count_path_count(),
        })

        if self._cycle_count % 100 == 0:
            self._runs_db.dump()

    def reproduce(
            self,
            parent: Input,
    ) -> Input:
        child_input = copy.copy(parent.input())

        for i in reversed(range(len(child_input))):
            if random.uniform(0.0, 1.0) <= BYTE_FLIP_PROBABILITY:
                child_input[i] = random.randint(0, self._runner.eof())

            if random.uniform(0.0, 1.0) <= INSERTION_PROBABILITY:
                child_input.insert(i, random.randint(0, self._runner.eof()))

            if random.uniform(0.0, 1.0) <= DELETION_PROBABILITY:
                child_input.pop(i)

        child_input = child_input[:self._runner.input_length()]

        if len(child_input) < self._runner.input_length():
            child_input += [self._runner.eof()] * \
                (self._runner.input_length() - len(child_input))

        assert len(child_input) == self._runner.input_length()

        length = 0
        for i in child_input:
            if i == self._runner.eof():
                break
            length += 1

        energy = parent.length() * INPUT_ENERGY * \
            REPRODUCTION_ENERGY_FACTOR / 2.0

        parent.receive_energy(-energy)

        return Input(child_input, length, energy)


def run():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'runs_db_path',
        type=str, help="path to the runs db",
    )
    args = parser.parse_args()

    assert args.runs_db_path is None or not os.path.isfile(args.runs_db_path)

    config = Config.from_file(args.config_path)
    runner = Runner(config)
    runs_db = RunsDB.from_file(args.runs_db_path, config, runner)

    fuzzer = GeneticFuzzer(config, runner, runs_db)

    while True:
        fuzzer.cycle()
