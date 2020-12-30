import torch
import torch.nn as nn


class CrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.softmax_cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, label):
        # NCHW->NHWC
        logits = logits.permute(0, 2, 3, 1)
        logits = logits.float()
        label = label.permute(0, 2, 3, 1)
        label = label.squeeze().long()

        loss = torch.mean(
            self.softmax_cross_entropy_loss(logits.reshape(-1, 2), label.reshape(-1)))
        return loss
