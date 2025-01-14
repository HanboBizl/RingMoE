import os

from mindspore import nn
from mindspore.ops import operations as P


class TokenExchange(nn.Cell):
    """Two Img Modals Exchange Tokens Each Other."""

    def __init__(self, batch_size, seq_length, embed_dim, modal_mask_threshold=0.002, parallel_config=None):
        super(TokenExchange, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.zeros_like = P.ZerosLike().shard(((dp, 1, 1),))
        self.broadcast = P.BroadcastTo((batch_size // 2, seq_length, embed_dim)).shard(((dp, 1, 1),))
        self.greater = P.GreaterEqual().shard(((dp, 1, 1), ()))
        self.less = P.Less().shard(((dp, 1, 1), ()))
        self.select = P.Select().shard(((dp, 1, 1), (dp, 1, 1), (dp, 1, 1)))
        self.split = P.Split(0, 2).shard(((1, 1, 1),))
        self.cat_3d = P.Concat(axis=0).shard(((1, 1, 1), (1, 1, 1)))
        self.mask_threshold = modal_mask_threshold
        self.split_mm = MMSplit()
        self.cat_mm = MMConcat()


    def split_modal(self, x, modal):
        x_split = self.split_mm(x)
        modal_split = self.split_mm(modal)
        return x_split, modal_split

    def construct(self, x, modal_mask):
        # x 128*197*768  128*197*1 0.02
        x, modal_mask = self.split_modal(x, modal_mask)
        x0_new = self.zeros_like(x[0])
        x1_new = self.zeros_like(x[1])
        modal_mask1 = self.broadcast(modal_mask[0])
        modal_mask2 = self.broadcast(modal_mask[1])
        cond1_greater = self.greater(modal_mask1, self.mask_threshold)
        cond2_greater = self.greater(modal_mask2, self.mask_threshold)
        cond1_less = self.less(modal_mask1, self.mask_threshold)
        cond2_less = self.less(modal_mask2, self.mask_threshold)
        x0 = self.select(cond1_greater, x[0], x0_new)
        x0 = self.select(cond1_less, x[1], x0)
        x1 = self.select(cond2_greater, x[1], x1_new)
        x1 = self.select(cond2_less, x[0], x1)
        out = self.cat_mm(x0, x1)
        return out


class MMSplit(nn.Cell):
    def __init__(self):
        super(MMSplit, self).__init__()
        self.split = P.Split(0, 2).shard(((1, 1, 1),))
        # self.split.add_prim_attr("primitive_target", "CPU")

    def construct(self, x):
        return self.split(x)


class MMConcat(nn.Cell):
    def __init__(self):
        super(MMConcat, self).__init__()
        self.cat_3d = P.Concat(axis=0).shard(((1, 1, 1), (1, 1, 1)))
        # self.cat_3d.add_prim_attr("primitive_target", "CPU")

    def construct(self, x, y):
        return self.cat_3d((x, y))


if __name__ == "__main__":
    import mindspore as ms

    ms.context.set_context(device_target="CPU")
    sdn1 = P.StandardNormal(seed=21)
    sdn2 = P.StandardNormal(seed=2111)
    x = [sdn1((2, 4, 8)), sdn2((2, 4, 8))]
    m = [sdn2((2, 4, 1)), sdn1((2, 4, 1))]
    print(x, m)

    tx = TokenExchange(2, 4, 8)
    print("aaaaaaaaaaaaaa")
    print(tx(x, m))
