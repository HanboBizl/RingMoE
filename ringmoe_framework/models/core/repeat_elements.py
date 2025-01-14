from mindspore import nn
# from mindspore._checkparam import Rel
# from mindspore._checkparam import Validator as validator
##taoht##
from mindspore import _checkparam as validator
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr

##taoht##
'''
from enum import Enum
class Rel(Enum):

    """Numerical relationship between variables, logical relationship enumeration definition of range."""
    # scalar compare
    EQ = 1  # ==
    NE = 2  # !=
    LT = 3  # <
    LE = 4  # <=
    GT = 5  # >
    GE = 6  # >=
    # scalar range check
    INC_NEITHER = 7  # (), include neither
    INC_LEFT = 8  # [), include left
    INC_RIGHT = 9  # (], include right
    INC_BOTH = 10  # [], include both
    # collection in, not in
    IN = 11
    NOT_IN = 12

    @staticmethod
    def get_strs(rel):
        """Get value from rel_strs."""
        return rel_strs.get(rel, "")

    @staticmethod
    def get_fns(rel):
        """Get value from rel_fns."""
        return rel_fns.get(rel, lambda *args: False)


rel_fns = {
    # scalar compare
    Rel.EQ: lambda x, y: x == y,
    Rel.NE: lambda x, y: x != y,
    Rel.LT: lambda x, y: x < y,
    Rel.LE: lambda x, y: x <= y,
    Rel.GT: lambda x, y: x > y,
    Rel.GE: lambda x, y: x >= y,
    # scalar range check
    Rel.INC_NEITHER: lambda x, lower, upper: (lower < x < upper),
    Rel.INC_LEFT: lambda x, lower, upper: (lower <= x < upper),
    Rel.INC_RIGHT: lambda x, lower, upper: (lower < x <= upper),
    Rel.INC_BOTH: lambda x, lower, upper: (lower <= x <= upper),
    # collection in, not in
    Rel.IN: lambda x, y: x in y,
    Rel.NOT_IN: lambda x, y: x not in y,
}

rel_strs = {
    # scalar compare
    Rel.EQ: "= {}",
    Rel.NE: "!= {}",
    Rel.LT: "< {}",
    Rel.LE: "<= {}",
    Rel.GT: "> {}",
    Rel.GE: ">= {}",
    # scalar range check
    Rel.INC_NEITHER: "({}, {})",
    Rel.INC_LEFT: "[{}, {})",
    Rel.INC_RIGHT: "({}, {}]",
    Rel.INC_BOTH: "[{}, {}]",
    # collection in, not in
    Rel.IN: "in {}",
    Rel.NOT_IN: "not in {}",
}
'''


@constexpr
def _check_is_int(arg_value, arg_name, op_name):
    arg_value = validator.check_is_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_positive_int(arg_value, arg_name, op_name):
    arg_value = validator.check_positive_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_axis_range(arg_value, limit, arg_name, op_name):
    arg_value = validator.check_int_range(arg_value, -limit, limit, validator.INC_LEFT, arg_name, op_name)
    return arg_value


@constexpr
def _cal_repeat_dims(x_rank, rep, expand_axis):
    rep_dims = [1] * (x_rank + 1)
    rep_dims[expand_axis] = rep
    return tuple(rep_dims)


@constexpr
def _cal_reshape(x_shape, rep, axis):
    x_reshape = list(x_shape)
    x_reshape[axis] *= rep
    return tuple(x_reshape)


class RepeatElement(nn.Cell):
    def __init__(self, rep, axis, parallel_config=None):
        super(RepeatElement, self).__init__()

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        rep = _check_positive_int(rep, "rep", "repeat_elements")
        axis = _check_is_int(axis, "axis", "repeat_elements")
        self.shape_op = P.Shape()
        self.tile_op = P.Tile().shard(((dp, 1, 1, 1),))
        self.expand_dims_op = P.ExpandDims().shard(((dp, 1, 1),))
        self.reshape_op = P.Reshape()
        self.rank = P.Rank().shard(((dp, 1, 1),))
        self.rep = rep
        self.axis = axis

    def construct(self, x):
        x_rank = self.rank(x)
        axis = _check_axis_range(self.axis, x_rank, "axis", "repeat_elements")
        expand_axis = axis + 1
        x_expand = self.expand_dims_op(x, expand_axis)
        rep_dims = _cal_repeat_dims(x_rank, self.rep, expand_axis)
        x_expand = self.tile_op(x_expand, rep_dims)
        x_shape = self.shape_op(x)
        x_reshape = _cal_reshape(x_shape, self.rep, self.axis)
        x_rep = self.reshape_op(x_expand, x_reshape)
        return x_rep


if __name__ == "__main__":
    from mindspore import Tensor, context
    import numpy as np

    context.set_context(device_target="Ascend", mode=0, device_id=6)
    input_x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]))
    re = RepeatElement(rep=2, axis=1)
    print(re(input_x))
