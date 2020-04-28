from torch import flatten
from torch.nn import Module, Sequential
from torch.nn import Conv2d, \
                     MaxPool2d, \
                     AdaptiveAvgPool2d, \
                     Linear, \
                     ReLU, \
                     Softmax


class Scratch_Model(Module):
    def __init__(self):
        super(Scratch_Model, self).__init__()
        self.feat_extract = Sequential(Conv2d(3, 16, (3, 3)),
                                       MaxPool2d(2),
                                       ReLU(),
                                       Conv2d(16, 32, (3, 3)),
                                       MaxPool2d(2),
                                       ReLU(),
                                       Conv2d(32, 64, (3, 3)),
                                       MaxPool2d(2),
                                       ReLU())

        self.classifier = Sequential(Linear(43264, 133),
                                     ReLU(),
                                     Softmax())

    def forward(self, x):
        x = self.feat_extract(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x