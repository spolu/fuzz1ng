import base64
import json
import os
import time

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
            input: bytes,
    ) -> None:
        self._runs.append({
            'input': input,
        })
        self._runs = self._runs[:self._pool_size]

    def __iter__(
            self,
    ):
        yield 'runs', [
            base64.b64encode(r['input']).decode('utf8')
            for r in self._runs
        ]

    @staticmethod
    def from_dict(
            spec,
            config: Config,
            runner: Runner,
    ):
        pool = Pool(config)

        inputs = [
            base64.b64decode(r.encode('utf8'))
            for r in spec['runs']
        ]

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

        self._skip_pool = {}
        self._count_pool = {}
        self._run_count = 0

    def store(
            self,
            input: bytes,
            coverage: Coverage,
    ) -> None:
        assert len(coverage.skip_path_list()) == 1
        assert len(coverage.count_path_list()) == 1

        skip_path = coverage.skip_path_list()[0]
        if skip_path not in self._skip_pool:
            self._skip_pool[skip_path] = Pool(self._config)
        self._skip_pool[skip_path].store(input)

        count_path = coverage.count_path_list()[0]
        if count_path not in self._count_pool:
            self._count_pool[count_path] = Pool(self._config)
        self._count_pool[count_path].store(input)

        self._run_count += 1

    def run_count(
            self,
    ) -> int:
        return self._run_count

    def unique_skip_path_count(
            self,
    ) -> int:
        return len(self._skip_pool)

    def unique_count_path_count(
            self,
    ) -> int:
        return len(self._count_pool)

    def dump(
            self,
    ) -> None:
        start_time = time.time()

        with open(self._dump_path, 'w') as out:
            json.dump(dict(self), out, indent=2)

        run_time = time.time() - start_time

        Log.out("Dumping RunsDB", {
            "run_time": '%.2f' % (run_time),
            "dump_path": self._dump_path,
        })

    def __iter__(
            self,
    ):
        skip_pool = {}
        for p in self._skip_pool:
            key = base64.b64encode(p).decode('utf8')
            skip_pool[key] = dict(self._skip_pool[p])
        count_pool = {}
        for p in self._count_pool:
            key = base64.b64encode(p).decode('utf8')
            count_pool[key] = dict(self._count_pool[p])

        yield 'skip_pool', skip_pool
        yield 'count_pool', count_pool
        yield 'run_count', self._run_count

    @staticmethod
    def from_dict(
            spec,
            config: Config,
            dump_path: str,
            runner: Runner,
    ):
        runs_db = RunsDB(config, dump_path)

        for p in spec['skip_pool']:
            key = base64.b64decode(p.encode('utf8'))
            runs_db._skip_pool[key] = Pool.from_dict(
                spec['skip_pool'][p], config, runner,
            )
        for p in spec['count_pool']:
            key = base64.b64decode(p.encode('utf8'))
            runs_db._count_pool[key] = Pool.from_dict(
                spec['count_pool'][p], config, runner,
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
