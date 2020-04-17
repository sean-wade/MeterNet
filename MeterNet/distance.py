
import numpy as np

import torch
from torch.autograd import Function


class PairwiseDistance(Function):
    def __init__(self, p=2):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        if (x1.size() != x2.size()):
            return -1
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


class NumpyDistance(object):
    def __init__(self, p=2):
        super(NumpyDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        if (x1.size != x2.size):
            return - 1
        eps = 1e-4 / x1.size
        power_sum = np.sum(np.power(np.abs(x1 - x2), self.norm))
        return np.power(power_sum + eps, 1./self.norm)


if __name__ == "__main__":
    nd = NumpyDistance(5)
    a = np.array([1, 2, 3])
    b = np.array([3, 4, 5])
    print(nd.forward(a, b))

    pd = PairwiseDistance(5)
    c = torch.tensor([[1, 2, 3]])
    d = torch.tensor([[3, 4, 5]])
    print(pd.forward(c, d))
    