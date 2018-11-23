import datetime
import typing


class Log:
    @staticmethod
    def out(
            message: str,
            data: typing.Dict[str, typing.Any],
    ) -> None:
        message = "[{}] {}:".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
            message,
        )
        for k in data:
            message += " {}={}".format(k, data[k])

        print(message)
