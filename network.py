from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Softmax, ReLU


class CNN(Module):
    """
    Convolutional neural network for classification.
    """

    def __init__(self) -> None:
        super().__init__()

        self.layers = Sequential(
            Conv2d(3, 12, 5),  # 32x32 -> 28x28
            MaxPool2d(2),  # 28x28 -> 14x14
            LeakyReLU(),

            Conv2d(12, 48, 5),  # 14x14 -> 10x10
            MaxPool2d(2),  # 10x10 -> 5x5
            LeakyReLU(),
            Flatten(),

            Linear(1200, 500),
            LeakyReLU(),

            Linear(500, 100),
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
