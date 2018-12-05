import argparse
import base64
import json
import os
import random
import shutil
import time
import torch
import typing

from gym_fuzz1ng.coverage import Coverage

from torch.utils.data import Dataset

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
            coverage: Coverage,
    ) -> None:
        self._runs.append({
            'input': input,
            'coverage': coverage,
        })
        self._runs = self._runs[:self._pool_size]

    def sample(
            self,
            count=1,
    ) -> typing.List[int]:
        assert len(self._runs) > 0
        assert count < self._pool_size

        return [
            self._runs[random.randint(0, len(self._runs)-1)]['input']
            for _ in range(count)
        ]


class RunsDB:
    """ `Database` stores historical inputs and coverages.

    It organizes the database around unique count/skip pathes and maintain a
    LIFO sample pool of inputs for each unique path.
    """
    def __init__(
            self,
            config,
            dump_dir,
    ) -> None:
        self._config = config
        self._dump_dir = dump_dir

        self._pools = {}
        self._run_count = 0

    def store(
            self,
            input: typing.List[int],
            coverage: Coverage,
    ) -> bool:
        assert len(coverage.skip_path_list()) == 1
        assert len(coverage.path_list()) == 1

        new_path = False
        path = coverage.path_list()[0]
        if path not in self._pools:
            new_path = True
            self._pools[path] = Pool(self._config)

        self._pools[path].store(input, coverage)

        self._run_count += 1
        return new_path

    def run_count(
            self,
    ) -> int:
        return self._run_count

    def path_count(
            self,
    ) -> int:
        return len(self._pools)

    def sample(
            self,
            count=1,
    ) -> typing.List[typing.List[int]]:
        inputs = []

        for p in self._pools:
            inputs += self._pools[p].sample(count)

        return inputs

    def dump(
            self,
    ) -> None:
        start_time = time.time()

        if not os.path.isdir(self._dump_dir):
            os.mkdir(self._dump_dir)

        if os.path.isdir(os.path.join(self._dump_dir, '_pools')):
            shutil.rmtree(os.path.join(self._dump_dir, '_pools'))

        os.mkdir(os.path.join(self._dump_dir, '_pools'))

        if os.path.isfile(os.path.join(self._dump_dir, 'runs_db.json')):
            os.unlink(os.path.join(self._dump_dir, 'runs_db.json'))

        dump = {
            'run_count': self._run_count,
            'pools': [],
        }

        dump_count = 0

        for p in self._pools:
            key = base64.urlsafe_b64encode(p).decode('utf8')
            dump['pools'].append(key)

            pool_dir = os.path.join(self._dump_dir, '_pools', key)
            os.mkdir(pool_dir)

            for i in range(len(self._pools[p]._runs)):
                with open(os.path.join(pool_dir, str(i) + '.json'), 'w') as f:
                    json.dump(self._pools[p]._runs[i]['input'], f)
                dump_count += 1

        with open(os.path.join(self._dump_dir, 'runs_db.json'), 'w') as f:
            json.dump(dump, f)

        run_time = time.time() - start_time

        Log.out("Dumping RunsDB", {
            "run_time": '%.2f' % (run_time),
            "dump_count": dump_count,
            "dump_dir": self._dump_dir,
        })

    @staticmethod
    def from_dump_dir(
            dump_dir: str,
            config: Config,
            runner: Runner,
    ):
        assert os.path.isdir(dump_dir)

        if not os.path.isdir(os.path.join(dump_dir, '_pools')):
            os.mkdir(os.path.join(dump_dir, '_pools'))

        dump = {
            'pools': {},
            'run_count': 0,
        }
        if os.path.isfile(os.path.join(dump_dir, 'runs_db.json')):
            with open(os.path.join(dump_dir, 'runs_db.json'), 'r') as f:
                dump = json.load(f)

        db = RunsDB(config, dump_dir)

        db._run_count = dump['run_count']

        inputs = []
        for key in dump['pools']:
            pool_dir = os.path.join(dump_dir, '_pools', key)
            assert os.path.isdir(pool_dir)

            runs = [
                os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
                if os.path.isfile(os.path.join(pool_dir, f))
            ]
            for path in runs:
                with open(path, 'r') as f:
                    inputs.append(json.load(f))

        coverages, _, _ = runner.run(inputs)

        for i in range(len(inputs)):
            db.store(inputs[i], coverages[i])

        return db


class RunsDBDataset(Dataset):
    def __init__(
            self,
            config: Config,
            runs_db: RunsDB,
            dict_size: int,
            input_size: int,
            test: bool = False,
    ) -> None:
        self._config = config
        self._runs_db = runs_db
        self._test = test

        self._input_size = input_size
        self._dict_size = dict_size

        self._device = torch.device(config.get('device'))

    def __len__(
            self,
    ) -> int:
        test_size = self._config.get('database_pool_test_size')

        test_count = 0
        all_count = 0

        for key in self._runs_db._pools:
            pool_size = len(self._runs_db._pools[key]._runs)

            pool_test = 0
            pool_all = 0
            if pool_size > 2 * test_size:
                pool_test += test_size
                pool_all += pool_size - test_size
            else:
                pool_all = pool_size

            test_count += pool_test
            all_count += pool_all

        if self._test:
            return test_count
        else:
            return all_count

    def __getitem__(
            self,
            idx: int,
    ):
        test_size = self._config.get('database_pool_test_size')

        run = None
        for key in self._runs_db._pools:
            pool_size = len(self._runs_db._pools[key]._runs)

            pool_test = 0
            pool_all = 0
            if pool_size > 2 * test_size:
                pool_test += test_size
                pool_all += pool_size - test_size
            else:
                pool_all = pool_size

            if self._test:
                if pool_test > idx:
                    run = self._runs_db._pools[key]._runs[idx]
                    break
                else:
                    idx -= pool_test
            else:
                if pool_all > idx:
                    run = self._runs_db._pools[key]._runs[idx]
                    break
                else:
                    idx -= pool_all

        assert run is not None

        return (
            torch.LongTensor(run['input']).to(self._device),
            torch.FloatTensor(run['coverage'].observation()).to(self._device),
        )


def eval():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'runs_db_dir',
        type=str, help="directory of the runs_db",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)
    runner = Runner(config)
    RunsDB.from_dump_dir(
        os.path.expanduser(args.runs_db_dir),
        config, runner,
    )
