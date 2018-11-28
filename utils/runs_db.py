import base64
import json
import os
import random
import time
import typing

from gym_fuzz1ng.coverage import Coverage

from utils.config import Config
from utils.log import Log
from utils.runner import Runner


class Pool:
    def __init__(
            self,
            config: Config
    ) -> None:
        self._pool_size = config.get('database_pool_size')
        self._runs = []

    def store(
            self,
            input: typing.List[int],
    ) -> None:
        self._runs.append({
            'input': input,
        })
        self._runs = self._runs[:self._pool_size]

    def sample(
            self,
    ) -> typing.List[int]:
        assert len(self._runs) > 0

        return self._runs[random.randint(0, len(self._runs)-1)]['input']

    def __iter__(
            self,
    ):
        yield 'runs', [r['input'] for r in self._runs]

    @staticmethod
    def from_dict(
            spec,
            config: Config,
            runner: Runner,
    ):
        pool = Pool(config)

        inputs = [r for r in spec['runs']]

        for i in range(len(inputs)):
            pool.store(inputs[i])

        return pool


class RunsDB:
    """ `Database` stores historical inputs and coverages.

    It organizes the database around unique count/skip pathes and maintain a
    LIFO sample pool of inputs for each unique path.
    """
    def __init__(
            self,
            config,
            dump_path,
    ) -> None:
        self._config = config
        self._dump_path = dump_path

        self._pool = {}
        self._skip_pool = {}
        self._run_count = 0

    def store(
            self,
            input: typing.List[int],
            coverage: Coverage,
    ) -> None:
        assert len(coverage.skip_path_list()) == 1
        assert len(coverage.path_list()) == 1

        skip_path = coverage.skip_path_list()[0]
        if skip_path not in self._skip_pool:
            self._skip_pool[skip_path] = Pool(self._config)
        self._skip_pool[skip_path].store(input)

        path = coverage.path_list()[0]
        if path not in self._pool:
            self._pool[path] = Pool(self._config)
        self._pool[path].store(input)

        self._run_count += 1

    def run_count(
            self,
    ) -> int:
        return self._run_count

    def unique_skip_path_count(
            self,
    ) -> int:
        return len(self._skip_pool)

    def unique_path_count(
            self,
    ) -> int:
        return len(self._pool)

    def dump(
            self,
    ) -> None:
        start_time = time.time()

        with open(self._dump_path, 'w') as out:
            json.dump(dict(self), out)

        run_time = time.time() - start_time

        Log.out("Dumping RunsDB", {
            "run_time": '%.2f' % (run_time),
            "dump_path": self._dump_path,
        })

    def sample(
            self,
    ) -> typing.List[typing.List[int]]:
        inputs = []
        # All runs in `_skip_pool` are necessarily in `_pool`.
        for p in self._pool:
            inputs.append(self._pool[p].sample())

        return inputs

    def __iter__(
            self,
    ):
        skip_pool = {}
        for p in self._skip_pool:
            key = base64.b64encode(p).decode('utf8')
            skip_pool[key] = dict(self._skip_pool[p])
        pool = {}
        for p in self._pool:
            key = base64.b64encode(p).decode('utf8')
            pool[key] = dict(self._pool[p])

        yield 'pool', pool
        yield 'skip_pool', skip_pool
        yield 'run_count', self._run_count

    @staticmethod
    def from_dict(
            spec,
            config: Config,
            dump_path: str,
            runner: Runner,
    ):
        runs_db = RunsDB(config, dump_path)

        for p in spec['pool']:
            key = base64.b64decode(p.encode('utf8'))
            runs_db._pool[key] = Pool.from_dict(
                spec['pool'][p], config, runner,
            )
        for p in spec['skip_pool']:
            key = base64.b64decode(p.encode('utf8'))
            runs_db._skip_pool[key] = Pool.from_dict(
                spec['skip_pool'][p], config, runner,
            )
        runs_db._run_count = spec['run_count']

        return runs_db

    @staticmethod
    def from_file(
            path: str,
            config: Config,
            runner: Runner,
    ):
        if path is not None and os.path.isfile(path):
            with open(path) as f:
                return RunsDB.from_dict(json.load(f), config, path, runner)
        else:
            return RunsDB(config, path)
