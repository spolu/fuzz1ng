import json


class Config:
    def __init__(
            self,
            config,
    ) -> None:
        self._config = config

    def get(
            self,
            key: str,
    ):
        if key not in self._config:
            raise Exception("Unknown Config key {}".format(key))
        return self._config[key]

    def override(
            self,
            key: str,
            value,
    ) -> None:
        self._config[key] = value

    def __eq__(
            self,
            other,
    ) -> bool:
        a = json.dumps(self._config, sort_keys=True)
        b = json.dumps(other._config, sort_keys=True)
        return a == b

    @staticmethod
    def from_dict(
            config,
    ):
        return Config(config)

    @staticmethod
    def from_file(
            path: str,
    ):
        with open(path) as f:
            return Config.from_dict(json.load(f))
