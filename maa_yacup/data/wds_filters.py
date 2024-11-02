import torch
import webdataset as wds
import logging
import random
import os
import pandas as pd
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


def _select_loc_columns(data, columns=[0, 1, 5]):
    """Skips all elements up to start_id"""
    for e in data:
        e["loc.pth"] = e["loc.pth"][..., columns, :]
        yield e


select_loc_columns = pipelinefilter(_select_loc_columns)


def _chunking(data, chunk_len=1001, shift=250, T_is_last=True, pad_value=None):
    assert T_is_last
    for e in data:
        loc = e["loc.pth"]
        assert len(loc.shape) == 2, "{loc.shape=}"
        F, T = loc.shape
        assert T > F, f"{loc.shape=}, {T_is_last=}"
        for i in range(0, T, shift):
            end = i + chunk_len
            if end > T:
                i = max(T - chunk_len, 0)
                end = T
            # print(f"{e['__key__']=}, {loc.shape=}, {i=}, {end=}")
            if pad_value is not None and (loc[:, i:end] == pad_value).all():
                logging.debug(f"Skip {e['__key__']}__{i}__{end}")
                continue
            chunk = {
                "__key__": e["__key__"] + f"__{i}__{end}",
                "loc.pth": loc[:, i:end],
                "control_feats.pth": e["control_feats.pth"][:, i:end],
            }
            for k in ['x.pth', 'y.pth', 'yaw.pth']:
                if k in e:
                    chunk[k] = e[k][..., i:end]
            yield chunk
            if end == T:
                break


chunking = pipelinefilter(_chunking)


def _write_as_sharded_wds(
    dataset, out_printf_frmt, max_elements_per_shard=200, keys_subset=None
):
    """
    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    Path(out_printf_frmt).parent.mkdir(parents=True, exist_ok=True)
    if keys_subset is not None:
        keys_subset = set(keys_subset)
        keys_subset.add("__key__")

    with wds.ShardWriter(out_printf_frmt, maxcount=max_elements_per_shard) as sink:
        for e in dataset:
            assert "__key__" in e, f"Bad dataset format {e.keys()}"
            if keys_subset is not None:
                e = {k: v for k, v in e.items() if k in keys_subset}
            sink.write(e)
            yield e


write_as_sharded_wds = pipelinefilter(_write_as_sharded_wds)


def _save_submit(data, outd=None, loc_key="predicted.pth", frame_step=20000000):
    fname = str(time.strftime("%Y-%m-%d-%H:%M:%S.txt", time.gmtime()))
    outf = f"{outd}/{fname}"
    with open(outf + ".csv", "w") as f:
        f.write("testcase_id,stamp_ns,x,y,yaw\n")
        df = []
        for e in data:
            req = e["req_stamps.pth"]
            loc = e[loc_key]
            for t in req:
                t = int(t.item())
                left_frame_id = t // frame_step
                right_frame_id = left_frame_id + 1
                delta = t - left_frame_id * frame_step
                bounds = loc[left_frame_id : right_frame_id + 1]
                db = bounds[1] - bounds[0]
                ddot = db * delta / frame_step
                dot = bounds[0] + ddot
                id = int(e["__key__"].replace("YaCupTest__", ""))
                df.append(
                    {
                        "testcase_id": id,
                        "stamp_ns": t + e["start_ns.id"],
                        "x": dot[0].item(),
                        "y": dot[1].item(),
                        "yaw": dot[-1].item(),
                    }
                )
                f.write(f"{id},{t},{dot[0]},{dot[1]},{dot[-1]}\n")
            yield e
    df = pd.DataFrame(sorted(df, key=lambda row: (row["testcase_id"], row["stamp_ns"])))
    df.to_csv(outf + ".sorted.csv.gz", index=False, compression="gzip")


save_submit = pipelinefilter(_save_submit)


def _prepare_input_feats_v1(data):
    for e in data:
        al = e.pop("acceleration_level.pth")
        s = e.pop("steering.pth")
        T = al.shape[0]
        # 4 X T
        e["control_feats.pth"] = torch.stack(
            [
                al,
                s,
                torch.full((T,), e["vehicle_model.id"]),
                torch.full((T,), e["vehicle_model_modification.id"]),
            ]
        )
        x = e.pop("x.pth")
        y = e.pop("y.pth")
        z = e.pop("z.pth")
        roll = e.pop("roll.pth")
        pitch = e.pop("pitch.pth")
        yaw = e.pop("yaw.pth")
        e["loc.pth"] = torch.stack([x, y, z, roll, pitch, yaw])
        yield e


prepare_input_feats_v1 = pipelinefilter(_prepare_input_feats_v1)


def _prepare_input_feats_v2(data):
    for e in data:
        al = e.pop("acceleration_level.pth")
        s = e.pop("steering.pth")

        T = al.shape[0]
        # 4 X T
        e["control_feats.pth"] = torch.stack(
            [
                al,
                s,
                torch.full((T,), e['vehicle_model.id']),
                torch.full((T,), e['vehicle_model_modification.id'])
            ]
        )
        x = e["x.pth"]
        y = e["y.pth"]
        v_x = e.pop("v_x.pth")
        v_y = e.pop("v_y.pth")
        mod_v = e.pop("mod_v.pth")
        v_direct = e.pop("v_direct.pth")
        yaw = e["yaw.pth"]
        e["loc.pth"] = torch.stack([x, y, v_x, v_y, mod_v, v_direct, yaw])
        # remove unused loc
        z = e.pop("z.pth")
        roll = e.pop("roll.pth")
        pitch = e.pop("pitch.pth")
        v_yaw = e.pop("v_yaw.pth")
        ###
        yield e


prepare_input_feats_v2 = pipelinefilter(_prepare_input_feats_v2)
