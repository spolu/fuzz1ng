import argparse
import base64
import json
import os
import random
import shutil
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
    ) -> None:
        assert len(coverage.skip_path_list()) == 1
        assert len(coverage.path_list()) == 1

        path = coverage.path_list()[0]
        if path not in self._pools:
            self._pools[path] = Pool(self._config)
        self._pools[path].store(input)

        self._run_count += 1

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
    ) -> typing.List[typing.List[int]]:
        inputs = []

        for p in self._pools:
            inputs.append(self._pools[p].sample())

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
        assert os.path.isdir(os.path.join(dump_dir, '_pools'))
        assert os.path.isfile(os.path.join(dump_dir, 'runs_db.json'))

        dump = None
        with open(os.path.join(dump_dir, 'runs_db.json'), 'r') as f:
            dump = json.load(f)

        db = RunsDB(config, dump_dir)

        db._run_count = dump['run_count']

        for key in dump['pools']:
            p = base64.urlsafe_b64decode(key.encode('utf8'))
            db._pools[p] = Pool(config)

            pool_dir = os.path.join(dump_dir, '_pools', key)
            assert os.path.isdir(pool_dir)

            runs = [
                os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
                if os.path.isfile(os.path.join(pool_dir, f))
            ]
            for path in runs:
                with open(path, 'r') as f:
                    db._pools[p].store(json.load(f))

        return db


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
    runs_db = RunsDB.from_dump_dir(
        os.path.expanduser(args.runs_db_dir),
        config, runner,
    )

    _, inputs_data, aggregate = runner.run(runs_db.sample())
