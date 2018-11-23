import gym
import gym_fuzz1ng  # noqa: F401
import time
import numpy as np
import typing

from gym_fuzz1ng.coverage import Coverage

from utils.config import Config
from utils.log import Log

class Runner:
    def __init__(
            self,
            config: Config,
    ) -> None:
        self._env = gym.make(config.get('gym_fuzz1ng_env'))

    def eof(
            self,
    ) -> int:
        return self._env.action_space.high[0]

    def input_length(
            self,
    ) -> int:
        return self._env.action_space.shape[0]

    def run(
            self,
            inputs: typing.List[bytes],
    ) -> typing.Tuple[typing.List[Coverage], Coverage]:
        start = time.time()
        coverages = []
        aggregate = Coverage()

        for i in inputs:
            _, _, _, info = self._env.step(np.array(i))
            coverages.append(info['step_coverage'])
            aggregate.add(info['step_coverage'])

        delta = time.time() - start

        Log.out("Run done", {
            "run_count": len(inputs),
            "time": '%.2f'%(delta),
            "exec_speed": '%.2f'%(len(inputs) / delta),
        })

        return coverages, aggregate
