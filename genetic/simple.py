import argparse
import copy
import os
import random
import time

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.log import Log
from utils.runner import Runner
from utils.runs_db import RunsDB

INSERTION_PROBABILITY = 0.05
BYTE_FLIP_PROBABILITY = 0.1
DELETION_PROBABILITY = 0.005


class SimpleFuzzer:
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
        self._cycle_count = 0

    def cycle(
            self,
    ) -> None:
        population = self._runs_db.sample(8)

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
        '--runs_db_dir',
        type=str, help="directory to dump the runs_db to",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)
    runner = Runner(config)
    runs_db = RunsDB(config, os.path.expanduser(args.runs_db_dir))

    fuzzer = SimpleFuzzer(config, runner, runs_db)

    while True:
        fuzzer.cycle()
