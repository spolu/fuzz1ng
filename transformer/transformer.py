import argparse
import torch

from utils.config import Config


class Coverage:
    def __init__(
            self,
            config,
    ):
        self.config = config
        self.device = torch.device(config.get('device'))


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--runs_db_dir',
        type=str, help="directory to dump the runs_db to",
    )
    parser.add_argument(
        '--device',
        type=str, help="config override",
    )
    parser.add_argument(
        '--tensorboard_log_dir',
        type=str, help="config override",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
    if args.tensorboard_log_dir is not None:
        config.override('tensorboard_log_dir', args.tensorboard_log_dir)
