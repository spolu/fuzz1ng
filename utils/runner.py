import gym
import gym_fuzz1ng  # noqa: F401
import time
import numpy as np
import signal
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
        return int(self._env.eof())

    def input_length(
            self,
    ) -> int:
        return int(self._env.action_space.shape[0])

    def run(
            self,
            inputs: typing.List[typing.List[int]],
    ) -> typing.Tuple[
        typing.List[Coverage],
        typing.List[bytes],
        Coverage
    ]:
        start_time = time.time()

        coverages = []
        inputs_data = []
        aggregate = Coverage()

        for i in inputs:
            info = self._env.step_raw(np.array(i))

            coverages.append(info['step_coverage'])
            inputs_data.append(info['input_data'])

            aggregate.add(info['step_coverage'])

        run_time = time.time() - start_time

        Log.out("Run done", {
            "run_count": len(inputs),
            "run_time": '%.2f' % (run_time),
            "exec_speed": '%.2f' % (len(inputs) / run_time),
            "transition_count": aggregate.transition_count(),
            "crash_count": aggregate.crash_count(),
            "path_count": aggregate.path_count(),
            "skip_path_count": aggregate.skip_path_count(),
        })

        return coverages, inputs_data, aggregate


def signal_handler(signal, frame):
    pass


signal.signal(signal.SIGINT, signal_handler)
