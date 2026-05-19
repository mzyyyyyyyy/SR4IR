import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, loss_weight=1.0,delta=1.0):
        super(HuberLoss, self).__init__()
        self.loss_weight = loss_weight
        self.delta = delta

    def forward(self, input, target):
        error = torch.abs(input - target)
        is_large_error = error > self.delta

        # L1 loss for large errors
        large_loss = self.delta * (error - 0.5 * self.delta)
        # L2 loss for small errors
        small_loss = 0.5 * error ** 2

        # Combine the two losses based on the boolean tensor
        # This is a key step to make the function differentiable
        loss = torch.where(is_large_error, large_loss, small_loss)

        return self.loss_weight * torch.mean(loss)