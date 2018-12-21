import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from genetic.simple import SimpleFuzzer

from tensorboardX import SummaryWriter

from torch.distributions.categorical import Categorical

from transformer.model import Coverage, Generator

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.runner import Runner
from utils.runs_db import RunsDB, RunsDBDataset


class Transformer:
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

        self._target_size = self._config.get('transformer_target_size')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._coverage_policy = Coverage(
            self._config,
            self._runner.dict_size(),
            self._runner.input_size(),
        ).to(self._device)

        self._coverage_optimizer = optim.Adam(
            self._coverage_policy.parameters(),
            lr=self._config.get('transformer_learning_rate'),
            betas=(
                self._config.get('transformer_adam_beta_1'),
                self._config.get('transformer_adam_beta_2'),
            ),
        )

        self._generator_policy = Generator(
            self._config,
            self._runner.dict_size(),
            self._runner.input_size(),
        ).to(self._device)

        self._generator_optimizer = optim.Adam(
            self._generator_policy.parameters(),
            lr=self._config.get('transformer_learning_rate'),
            betas=(
                self._config.get('transformer_adam_beta_1'),
                self._config.get('transformer_adam_beta_2'),
            ),
        )

        if self._load_dir:
            self._coverage_policy.load_state_dict(
                torch.load(
                    self._load_dir + "/coverage_policy.pt",
                    map_location=self._device,
                ),
            )
            self._coverage_optimizer.load_state_dict(
                torch.load(
                    self._load_dir + "/coverage_optimizer.pt",
                    map_location=self._device,
                ),
            )

            self._generator_policy.load_state_dict(
                torch.load(
                    self._load_dir + "/generator_policy.pt",
                    map_location=self._device,
                ),
            )
            self._generator_optimizer.load_state_dict(
                torch.load(
                    self._load_dir + "/generator_optimizer.pt",
                    map_location=self._device,
                ),
            )

        self._coverage_batch_count = 0
        self._generator_batch_count = 0

        self.reload_datasets()

    def reload_datasets(
            self,
    ):
        self._train_loader = torch.utils.data.DataLoader(
            RunsDBDataset(
                self._config,
                self._runs_db,
                self._runner.dict_size(),
                self._runner.input_size(),
                test=False
            ),
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=True,
            num_workers=0,
        )
        self._test_loader = torch.utils.data.DataLoader(
            RunsDBDataset(
                self._config,
                self._runs_db,
                self._runner.dict_size(),
                self._runner.input_size(),
                test=True),
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=False,
            num_workers=0,
        )

    def save_coverage(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving coverage models", {
                    'save_dir': self._save_dir,
                })
            torch.save(
                self._coverage_policy.state_dict(),
                self._save_dir + "/coverage_policy.pt",
            )
            torch.save(
                self._coverage_optimizer.state_dict(),
                self._save_dir + "/coverage_optimizer.pt",
            )

    def save_generator(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving generator models", {
                    'save_dir': self._save_dir,
                })
            torch.save(
                self._generator_policy.state_dict(),
                self._save_dir + "/generator_policy.pt",
            )
            torch.save(
                self._generator_optimizer.state_dict(),
                self._save_dir + "/generator_optimizer.pt",
            )

    def postprocess_coverages(
            self,
            coverages,
    ):
        coverages = coverages.sum(2)
        coverages = torch.where(
            coverages > 0,
            torch.ones(coverages.size()).to(self._device),
            torch.zeros(coverages.size()).to(self._device),
        )
        return coverages

    def augment_coverages(
            self,
            coverages,
    ):
        interpolation = torch.zeros(coverages.size()).to(self._device)
        for i in range(coverages.size(0)):
            alpha = random.random()
            interpolation[i] = \
                alpha * coverages[i] + (1-alpha) * coverages[i-1]
        interpolation = torch.where(
            interpolation > 0,
            torch.ones(interpolation.size()).to(self._device),
            torch.zeros(interpolation.size()).to(self._device),
        )

        return torch.cat((coverages, interpolation), 0)

    def batch_train_coverage(self):
        self._coverage_policy.train()
        loss_meter = Meter()

        for it, (inputs, coverages) in enumerate(self._train_loader):
            coverages = self.postprocess_coverages(coverages)
            generated = self._coverage_policy(inputs).squeeze(2)

            loss = F.mse_loss(generated, coverages)

            self._coverage_optimizer.zero_grad()
            loss.backward()
            self._coverage_optimizer.step()

            loss_meter.update(loss.item())

            if it == 0 and self._tb_writer is not None:
                c_images = []
                g_images = []

                for i in range(coverages.size(0)):
                    c = coverages[i].cpu()
                    c = c / c.max() * 255
                    c_images.append(c.to(torch.uint8).unsqueeze(0))

                    g = generated[i].cpu()
                    g = g / g.max() * 255
                    g_images.append(g.to(torch.uint8).unsqueeze(0))

                c_grid = torchvision.utils.make_grid(c_images)
                self._tb_writer.add_image(
                    'train/visual/coverage',
                    np.flip(c_grid.numpy(), axis=0),
                    self._coverage_batch_count,
                )
                g_grid = torchvision.utils.make_grid(g_images)
                self._tb_writer.add_image(
                    'train/visual/generated',
                    np.flip(g_grid.numpy(), axis=0),
                    self._coverage_batch_count,
                )

        Log.out("COVERAGE TRAIN", {
            'batch_count': self._coverage_batch_count,
            'loss_avg': loss_meter.avg,
            # 'loss_min': loss_meter.min,
            # 'loss_max': loss_meter.max,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "train/coverage/loss",
                loss_meter.avg, self._coverage_batch_count,
            )

        self._coverage_batch_count += 1

    def batch_test_coverage(
            self,
    ):
        self._coverage_policy.eval()
        loss_meter = Meter()

        with torch.no_grad():
            for it, (inputs, coverages) in enumerate(self._test_loader):
                coverages = self.postprocess_coverages(coverages)
                generated = self._coverage_policy(inputs).squeeze(2)

                loss = F.mse_loss(generated, coverages)

                loss_meter.update(loss.item())

        Log.out("COVERAGE TEST", {
            'batch_count': self._coverage_batch_count,
            'loss_avg': loss_meter.avg,
            # 'loss_min': loss_meter.min,
            # 'loss_max': loss_meter.max,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/coverage/loss",
                loss_meter.avg, self._coverage_batch_count,
            )

        return loss_meter.avg

    def batch_train_generator(self):
        self._generator_policy.train()
        loss_meter = Meter()
        reward_meter = Meter()

        discovered = []

        for it, (inputs, coverages) in enumerate(self._train_loader):
            target_coverages = self.postprocess_coverages(coverages)

            probs = self._generator_policy(target_coverages.unsqueeze(2))

            m = Categorical(probs)
            generated = m.sample()

            population = generated.tolist()
            coverages, _, _ = self._runner.run(population)

            for j in range(len(population)):
                if self._runs_db.is_new(coverages[j]):
                    discovered += [(population[j], coverages[j])]

            coverages = torch.cat([
                torch.FloatTensor(
                    c.observation(),
                ).unsqueeze(0).to(self._device)
                for c in coverages
            ], 0)
            coverages = self.postprocess_coverages(coverages)

            reward = 1 / (1 + torch.norm((coverages - target_coverages), 2, 1))
            loss = -m.log_prob(generated).mean(1) * reward
            loss = loss.mean()

            self._generator_optimizer.zero_grad()
            loss.backward()
            self._generator_optimizer.step()

            loss_meter.update(loss.item())
            reward_meter.update(reward.mean().item())

        for (input, coverage) in discovered:
            self._runs_db.store(input, coverage)

        Log.out("GENERATE", {
            'batch_count': self._generator_batch_count,
            'loss_avg': loss_meter.avg,
            'reward_avg': reward_meter.avg,
            'discovered': len(discovered),
            # 'loss_min': loss_meter.min,
            # 'loss_max': loss_meter.max,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "train/generator/loss",
                loss_meter.avg, self._generator_batch_count,
            )
            self._tb_writer.add_scalar(
                "train/generator/reward",
                reward_meter.avg, self._generator_batch_count,
            )
            self._tb_writer.add_scalar(
                "train/generator/discovered",
                len(discovered), self._generator_batch_count,
            )

        self._generator_batch_count += 1


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
        'phase',
        type=str, help="fuzzer, autoencoder, coverage",
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

    if args.device is not None:
        config.override('device', args.device)
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.transformer_load_dir is not None:
        config.override(
            'transformer_load_dir',
            os.path.expanduser(args.transformer_load_dir),
        )
    if args.transformer_save_dir is not None:
        config.override(
            'transformer_save_dir',
            os.path.expanduser(args.transformer_save_dir),
        )
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

    transformer = Transformer(config, runner, runs_db)
    fuzzer = SimpleFuzzer(config, runner, runs_db)

    i = 0
    while True:
        if args.phase == 'fuzzer':
            fuzzer.cycle()
            if i % 10 == 0:
                runs_db.dump()

        if args.phase == 'coverage':
            if i % 10 == 0:
                transformer.batch_test_coverage()
                transformer.save_coverage()
            transformer.batch_train_coverage()

        if args.phase == 'generate':
            if i % 10 == 0:
                transformer.save_generator()
            transformer.batch_train_generator()

        i += 1
