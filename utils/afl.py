import argparse
import os

from utils.config import Config
from utils.runner import Runner
from utils.runs_db import RunsDB


def dump():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'runs_db_dir',
        type=str, help="directory of the runs_db",
    )
    parser.add_argument(
        'afl_input_dir',
        type=str, help="path to the afl input dir",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)
    runner = Runner(config)
    runs_db = RunsDB.from_dump_dir(
        os.path.expanduser(args.runs_db_dir),
        config, runner,
    )

    _, inputs_data, aggregate = runner.run(runs_db.sample())

    for i in range(len(inputs_data)):
        with open(
                os.path.join(args.afl_input_dir, 'runs_db.{}'.format(i)),
                'wb'
        ) as out:
            out.write(inputs_data[i])
