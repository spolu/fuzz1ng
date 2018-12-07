import argparse
import copy
import os
import random
import time

from utils.config import Config
from utils.log import Log
from utils.runner import Runner
from utils.runs_db import RunsDB

INSERTION_PROBABILITY = 0.005
BYTE_FLIP_PROBABILITY = 0.12
DELETION_PROBABILITY = 0.001


class SimpleFuzzer:
    def __init__(
            self,
            config: Config,
            runner: Runner,
            runs_db: RunsDB,
    ) -> None:
        self.sample_count = config.get('genetic_simple_sample_count')

        self._runner = runner
        self._runs_db = runs_db
        self._cycle_count = 0

    def cycle(
            self,
    ) -> None:
        population = self._runs_db.sample(self.sample_count)

        if len(population) == 0:
            population = [
                [self._runner.eof()]*self._runner.input_length(),
            ]

        start_time = time.time()

        for j in range(len(population)):
            ii = copy.deepcopy(population[j])

            for i in reversed(range(len(ii))):
                if random.uniform(0.0, 1.0) <= BYTE_FLIP_PROBABILITY:
                    ii[i] = random.randint(0, self._runner.eof())

                if random.uniform(0.0, 1.0) <= INSERTION_PROBABILITY:
                    ii.insert(i, random.randint(0, self._runner.eof()))

                if random.uniform(0.0, 1.0) <= DELETION_PROBABILITY:
                    ii.pop(i)

            ii = ii[:self._runner.input_length()]

            if len(ii) < self._runner.input_length():
                ii += [self._runner.eof()] * \
                    (self._runner.input_length() - len(ii))

            assert len(ii) == self._runner.input_length()

            population[j] = ii

        coverages, _, _ = self._runner.run(population)

        for j in range(len(population)):
            self._runs_db.store(population[j], coverages[j])

        run_time = time.time() - start_time
        self._cycle_count += 1

        Log.out("Cycle done", {
            "cycle_count": self._cycle_count,
            "run_count": self._runs_db.run_count(),
            "run_time": '%.2f' % (run_time),
            "path_count": self._runs_db.path_count(),
        })

        if self._cycle_count % 100 == 0:
            self._runs_db.dump()


def fuzz():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'runs_db_dir',
        type=str, help="directory to the runs_db",
    )
    parser.add_argument(
        '--gym_fuzz1ng_env',
        type=str, help="config override",
    )
    parser.add_argument(
        '--genetic_simple_sample_count',
        type=int, help="config override",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.gym_fuzz1ng_env is not None:
        config.override(
            'gym_fuzz1ng_env',
            args.gym_fuzz1ng_env,
        )
    if args.genetic_simple_sample_count is not None:
        config.override(
            'genetic_simple_sample_count',
            args.genetic_simple_sample_count,
        )

    runner = Runner(config)
    runs_db = RunsDB.from_dump_dir(
        os.path.expanduser(args.runs_db_dir),
        config, runner,
    )

    fuzzer = SimpleFuzzer(config, runner, runs_db)

    i = 0
    while True:
        if i % 20 == 0:
            runs_db.dump()

        fuzzer.cycle()

        i += 1
