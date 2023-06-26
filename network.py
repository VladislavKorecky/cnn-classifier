from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Softmax


class CNN(Module):
    """
    Convolutional neural network for classification.
    """

    def __init__(self) -> None:
        super().__init__()

        """self.layers = Sequential(
            Conv2d(3, 6, 10),  # 32x32 -> 23x23
            LeakyReLU(),

            Conv2d(6, 12, 10),  # 23x23 -> 14x14
            LeakyReLU(),

            Conv2d(12, 24, 10),  # 14x14 -> 5x5
            LeakyReLU(),
            Flatten(),

            Linear(600, 100),  # 24 * 5 * 5
            LeakyReLU(),

            Linear(100, 10),
            Softmax()
        )"""

        self.layers = Sequential(
            Conv2d(1, 8, 5),  # 28x28 -> 24x24
            MaxPool2d(2),  # 24x24 -> 12x12
            LeakyReLU(),

            Conv2d(8, 24, 5),  # 12x12 -> 8x8
            MaxPool2d(2),  # 8x8 -> 4x4
            LeakyReLU(),
            Flatten(),

            Linear(384, 100),
            LeakyReLU(),

            Linear(100, 10),
            Softmax()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Do a forward pass on the network.

        Args:
            x (Tensor): Network's input.

        Returns:
            Tensor: Network's output.
        """

        return self.layers(x)
