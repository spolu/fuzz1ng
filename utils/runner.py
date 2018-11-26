import gym
import gym_fuzz1ng  # noqa: F401
import time
import numpy as np
import signal
import threading
import typing

from gym_fuzz1ng.coverage import Coverage

from utils.config import Config
from utils.log import Log


_send_condition = threading.Condition()
_recv_condition = threading.Condition()
_recv_count = 0
_runners = []


class Worker(threading.Thread):
    def __init__(
            self,
            config,
            index,
    ) -> None:
        self._env = gym.make(config.get('gym_fuzz1ng_env'))
        self._index = index
        self._stop = False

        self.inputs = []

        self.coverages = []
        self.inputs_data = []
        self.aggregate = Coverage()

        threading.Thread.__init__(self)

    def stop(
            self,
    ) -> None:
        # TODO(stan): fix this total hack
        assert False

    def run(
            self,
    ) -> None:
        global _send_condition
        global _recv_condition
        global _recv_count

        while True:
            # Wait for the controls to be set.
            _send_condition.acquire()
            _send_condition.wait()
            _send_condition.release()

            self.coverages = []
            self.inputs_data = []
            self.aggregate = Coverage()

            for i in self.inputs:
                _, _, _, info = self._env.step(np.array(i))

                self.coverages.append(info['step_coverage'])
                self.inputs_data.append(info['input_data'])
                self.aggregate.add(info['step_coverage'])

            # Notify that we are done.
            _recv_condition.acquire()
            _recv_count = _recv_count + 1
            _recv_condition.notify_all()
            _recv_condition.release()


class Runner:
    def __init__(
            self,
            config: Config,
    ) -> None:
        global _runners

        self._workers = [
            Worker(config, i)
            for i in range(config.get('runner_cpu_count'))
        ]
        for w in self._workers:
            w.start()

        _runners.append(self)

    def stop(
            self,
    ) -> None:
        for w in self._workers:
            w.stop()

    def eof(
            self,
    ) -> int:
        return int(self._workers[0]._env.eof())

    def input_length(
            self,
    ) -> int:
        return int(self._workers[0]._env.action_space.shape[0])

    def run(
            self,
            inputs: typing.List[typing.List[int]],
    ) -> typing.Tuple[
        typing.List[Coverage],
        typing.List[bytes],
        Coverage
    ]:
        global _recv_count
        global _send_condition
        global _recv_condition

        start_time = time.time()

        _recv_condition.acquire()
        _recv_count = 0

        for i in range(len(self._workers)):
            w = self._workers[i]
            b = int(i * len(inputs) / len(self._workers))
            e = int((i+1) * len(inputs) / len(self._workers))
            w.inputs = inputs[b:e]

        for i in range(len(self._workers)):
            # Release the workers.
            _send_condition.acquire()
            _send_condition.notify()
            _send_condition.release()

        # Wait for the workers to finish.
        first = True
        while _recv_count < len(self._workers):
            if first:
                first = False
            else:
                _recv_condition.acquire()
            _recv_condition.wait()
            _recv_condition.release()

        coverages = []
        inputs_data = []
        aggregate = Coverage()

        for i in range(len(self._workers)):
            w = self._workers[i]
            coverages += w.coverages
            inputs_data += w.inputs_data
            aggregate.add(w.aggregate)

        run_time = time.time() - start_time

        Log.out("Run done", {
            "run_count": len(inputs),
            "run_time": '%.2f' % (run_time),
            "exec_speed": '%.2f' % (len(inputs) / run_time),
        })

        return coverages, inputs_data, aggregate


def signal_handler(signal, frame):
    global _runners

    for r in _runners:
        r.stop()


signal.signal(signal.SIGINT, signal_handler)
