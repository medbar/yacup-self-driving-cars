import torch
import webdataset as wds
import logging
import random
import os
import time
import json

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from webdataset.filters import pipelinefilter
from pathlib import Path
from typing import Union, List

def collate_with_pad_v1(batch, pad_value):
    padded = {}
    for k, v in batch[0].items():
        e = [s[k] for s in batch]
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 2:
                # loc.pth and control_feats.pth is F x T
                e = [s.float().T for s in e]
                e = pad_sequence(e, batch_first=True, padding_value=pad_value)
            else:
                e = torch.stack(e)
        padded[k] = e
    return padded

def _batching_constant_batch_size(
    data,
    batch_size=10,
    partial=True,
    collate_fn=collate_with_pad_v1,
    collate_fn_kwargs={},
):
    """Create batches of the given size.
    :param data: iterator
    :param partial: return partial batches
    :returns: iterator
    """
    assert batch_size > 0, f"Wrong batch size {batch_size}"
    batch = []
    for sample in data:
        batch.append(sample)
        if len(batch) == batch_size:
            batch = collate_fn(batch, **collate_fn_kwargs)
            yield batch
            batch = []
    if len(batch) == 0:
        return
    elif partial:
        batch = collate_fn(batch, **collate_fn_kwargs)
        yield batch


batching_constant_batch_size = pipelinefilter(_batching_constant_batch_size)



