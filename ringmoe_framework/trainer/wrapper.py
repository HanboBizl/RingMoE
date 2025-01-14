from mindspore import nn
from mindspore import ops as P


class ClassificationMoeWrapper(nn.WithLossCell):
    def __init__(self, backbone, loss_fn):
        super(ClassificationMoeWrapper, self).__init__(backbone, loss_fn)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._add = P.Add().shard(((), ()))

    def construct(self, data, label):
        out, moe_loss = self._backbone(data)
        loss = self._loss_fn(out, label)
        return self._add(loss, moe_loss)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone
