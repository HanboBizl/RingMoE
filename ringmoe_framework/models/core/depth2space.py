from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr


@constexpr
def _depth_to_space(c, h, w, b):
    c_new = c // (b * b)
    h_new = h * b
    w_new = w * b
    return c_new, h_new, w_new


class DepthToSapce(nn.Cell):
    def __init__(self, block_size, parallel_config=None):
        super(DepthToSapce, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.block_size = block_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))
        # self.reshape.add_prim_attr("primitive_target", "CPU")
        # self.transpose.add_prim_attr("primitive_target", "CPU")

    def construct(self, x):
        N, C, H, W = x.shape

        C_new, H_new, W_new = _depth_to_space(C, H, W, self.block_size)

        x_reshape = self.reshape(x, (N, self.block_size, self.block_size, C_new, H, W))
        x_transpose = self.transpose(x_reshape, (0, 3, 4, 1, 5, 2))
        x = self.reshape(x_transpose, (N, C_new, H_new, W_new))
        return x


if __name__ == "__main__":
    import mindspore as ms
    from mindspore import Tensor, context
    import numpy as np
    import time

    context.set_context(device_target="Ascend", mode=0, device_id=6)
    input_x = Tensor(np.random.rand(1, 12, 1, 1), ms.float32)
    t1 = time.time()
    print(P.DepthToSpace(2)(input_x))
    print(time.time() - t1)
    t2 = time.time()
    print(DepthToSapce(block_size=2)(input_x))
    print(time.time() - t2)
