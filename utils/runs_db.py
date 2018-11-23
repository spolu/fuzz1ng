import base64

from gym_fuzz1ng.coverage import Coverage

from utils.config import Config
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
            coverage: Coverage,
    ) -> None:
        self._runs.append({
            'input': input,
            'coverage': coverage,
        })
        self._runs = self._runs[:self._pool_size]

    def __iter__(
            self,
    ):
        yield 'runs', [
            {'input': base64.b64encode(r['input'])} for r in self._runs
        ]

    @staticmethod
    def from_dict(
            spec,
            config: Config,
            runner: Runner,
    ):
        pool = Pool(config)

        inputs = [base64.b64decode(r['input']) for r in spec['runs']]
        coverages = runner.run(inputs)

        for i in range(len(inputs)):
            pool.store(inputs[i], coverages[i])


class RunsDB:
    """ `Database` stores historical inputs and coverages.

    It organizes the database around unique count/skip pathes and maintain a
    LIFO sample pool of inputs for each unique path.
    """
    def __init__(
            self,
            config,
    ) -> None:
        self._config = config

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
        self._skip_pool[skip_path].store(input, coverage)

        count_path = coverage.count_path_list()[0]
        if count_path not in self._count_pool:
            self._count_pool[count_path] = Pool(self._config)
        self._count_pool[count_path].store(input, coverage)

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
