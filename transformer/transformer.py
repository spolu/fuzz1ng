import argparse
import torch

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.runner import Runner
from utils.runs_db import RunsDB, RunsDBDataset


class Coverage:
    def __init__(
            self,
            config: Config,
            runner: Runner,
            runs_db: RunsDB,
    ):
        self._config = config
        self._runs_db = runs_db
        self._runner = runner

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('transformer_save_dir')
        self._load_dir = config.get('transformer_load_dir')

        self._tb_writer = None
        if self.tensorboard_log_dir:
            self.tb_writer = SummaryWriter(config.get('tensorboard_log_dir'))

        self._train_loader = torch.utils.data.DataLoader(
            RunsDBDataset(self._config, self._runs_db, test=False),
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=True,
            num_workers=0,
        )
        self._test_loader = torch.utils.data.DataLoader(
            RunsDBDataset(self._config, self._runs_db, test=True),
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=False,
            num_workers=0,
        )

        if self.load_dir:
            self.coverage.load_state_dict(
                torch.load(
                    self.load_dir + "/coverage.pt",
                    map_location=self.device,
                ),
            )
            self.coverage_optimizer.load_state_dict(
                torch.load(
                    self.load_dir + "/coverage_optimizer.pt",
                    map_location=self.device,
                ),
            )

    def batch_train(self):
        pass


def train():
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
        '--device',
        type=str, help="config override",
    )
    parser.add_argument(
        '--transformer_save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--transformer_load_dir',
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
    if args.transformer_load_dir is not None:
        config.override('transformer_load_dir', args.transformer_load_dir)
    if args.transformer_save_dir is not None:
        config.override('transformer_save_dir', args.transformer_save_dir)
