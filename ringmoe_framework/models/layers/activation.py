from mindspore import nn
from mindspore.ops import operations as P


class LogSoftmax(nn.Cell):
    def __init__(self, axis=-1, parallel_config=None):
        super(LogSoftmax, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.max = P.ArgMaxWithValue(axis=axis, keep_dims=True).shard(((dp, 1, 1),))
        self.sub = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.exp = P.Exp().shard(((dp, 1, 1),))
        self.sum = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1), (dp, 1, 1)))
        self.log = P.Log().shard(((dp, 1, 1),))
        self.axis = axis

    def construct(self, x):
        _, maximum = self.max(x)
        logits = self.sub(x, maximum)
        norm_logits = self.sum(self.exp(logits), self.axis)
        log_logits = self.log(norm_logits)
        return self.sub(logits, log_logits)
