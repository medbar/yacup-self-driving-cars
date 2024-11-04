import logging
import json
import contextlib
import random
from typing import Optional
import torch
import torch.nn as nn
import math


class TargetLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, ref_loc, hyp_loc):
        assert ref_loc.shape == hyp_loc.shape, f"{ref_loc.shape=}, {hyp_loc.shape=}"
        assert len(ref_loc.shape) >=2, f"{ref_loc.shape=}, {hyp_loc.shape=}"
        loss_per_frame = ((ref_loc - hyp_loc) ** 2).mean(dim=-1).sqrt()
        if self.reduction == "mean":
            return loss_per_frame.mean()
        elif self.reduction is None or self.reduction == "none":
            return loss_per_frame[..., None]
        else:
            raise self.reduction
