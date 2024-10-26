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



def _filter_keys(data, keep):
    """
    input Any
    output all columns from keep
    """
    keep = set(keep)
    keep.add("__key__")
    for sample in data:
        yield {k: sample[k] for k in keep}


filter_keys = pipelinefilter(_filter_keys)


def _limit_total_steps(data, thread_limit, num_threads=1):
    limit = thread_limit // num_threads
    logging.info(f"For each thread limit is {limit}")
    assert isinstance(limit, int), f"{limit}, {type(limit)}"
    for i, s in enumerate(data):
        if i == limit:
            epoch = s.get("__epoch__", None)
            logging.info(f"Reaching limit {limit}. Epoch is {epoch}")
            return
        yield s


limit_total_steps = pipelinefilter(_limit_total_steps)


def _goto_element_by_id(data, start_id):
    """Skips all elements up to start_id"""
    for i, element in enumerate(data):
        if i < start_id:
            continue
        if i == start_id:
            logging.info(f"{start_id=} has been reached. {element['__key__']}")
        yield element


goto_element_by_id = pipelinefilter(_goto_element_by_id)
