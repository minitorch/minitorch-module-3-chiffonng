"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import minitorch  # noqa: E402


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """Manual implementation for matrix multiplication
        x @ self.weights.value + self.bias.value"""

        # Manually broadcasting x and weights for element-wise multiplication
        x_broadcasted = x.view(*x.shape, 1)  # (batch_size, in_size, 1)
        weights_broadcasted = self.weights.value.view(
            1, *self.weights.value.shape
        )  # (1, in_size, out_size)

        # Element-wise multiplication for 2 tensors of same dimensions
        # -> (batch_size, in_size, out_size)
        v_1 = x_broadcasted * weights_broadcasted

        # Summing over the input dimension (in_size)
        # then reshaping to (batch_size, out_size)
        v_2 = v_1.sum(dim=1).contiguous().view(x.shape[0], self.out_size)

        # Adding the (reshaped) bias. -> (batch_size, out_size)
        return v_2 + self.bias.value.view(1, self.out_size)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 5
    RATE = 0.05
    data = minitorch.datasets["Xor"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
