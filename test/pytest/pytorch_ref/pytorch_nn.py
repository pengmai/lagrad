import torch
from torch import nn

torch.manual_seed(0)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


torch_mlp = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

def torch_primal(X: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
    return loss_fn(torch_mlp(X), y)
