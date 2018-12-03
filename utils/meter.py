class Meter():
    def __init__(
            self,
    ):
        self.reset()

    def reset(
            self,
    ):
        self.max = None
        self.min = None
        self.avg = None
        self.sum = 0
        self.cnt = 0

    def update(
            self,
            val,
    ):
        self.sum += val
        self.cnt += 1
        if self.max is None or self.max < val:
            self.max = val
        if self.min is None or self.min > val:
            self.min = val
        self.avg = self.sum / self.cnt
