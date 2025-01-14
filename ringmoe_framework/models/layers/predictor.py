import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from ringmoe_framework.models.layers.layers import LayerNorm, Linear


class PredictorLG(nn.Cell):
    """ Image to Patch Embedding from DydamicVit"""

    def __init__(self,
                 embed_dim=384,
                 weight_init='normal',
                 layernorm_compute_type=mstype.float32,
                 activation='gelu',
                 parallel_config=None):
        super(PredictorLG, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
        else:
            dp = mp = 1
        self.norm = LayerNorm((embed_dim,), eps=1e-6).to_float(layernorm_compute_type)
        self.norm.shard(((dp, 1, 1),))
        self.dense1 = Linear(
            embed_dim, embed_dim, activation=activation,
            weight_init=weight_init,
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.dense1.shard(strategy_matmul=((dp, 1), (mp, 1)),
                          strategy_bias=((dp, mp), (mp,)),
                          strategy_activation=((dp, mp),))

        self.dense2 = Linear(
            embed_dim, embed_dim // 2, activation=activation,
            weight_init=weight_init,
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.dense2.shard(strategy_matmul=((dp, 1), (mp, 1)),
                          strategy_bias=((dp, mp), (mp,)),
                          strategy_activation=((dp, mp),))

        self.dense3 = Linear(
            embed_dim // 2, embed_dim // 4, activation=activation,
            weight_init=weight_init,
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.dense3.shard(strategy_matmul=((dp, 1), (mp, 1)),
                          strategy_bias=((dp, mp), (mp,)),
                          strategy_activation=((dp, mp),))

        self.dense4 = Linear(
            embed_dim // 4, 2,
            weight_init=weight_init,
            compute_dtype=mstype.float16).to_float(mstype.float16)
        end_mp = mp if mp <= 2 else 2
        self.dense4.shard(strategy_matmul=((dp, 1), (end_mp, 1)),
                          strategy_bias=((dp, end_mp), (end_mp,)),
                          strategy_activation=((dp, end_mp),))

        self.log_softmax = nn.LogSoftmax()
        self.log_softmax.log_softmax.shard(((dp, 1, 1),))
        self.cast = P.Cast()

    def construct(self, x):
        x = self.norm(x)
        x = self.cast(x, mstype.float16)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.log_softmax(x)
        return x
