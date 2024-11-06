import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import pickle
from sklearn.preprocessing import StandardScaler

DATA = [*glob("YandexCup2024v2/YaCupTest/*")]

print("---Normalizing---")
vehicle_model2data = defaultdict(list)
for d in tqdm(DATA):
    with open(f"{d}/metadata.json") as f:
        meta = json.load(f)
    for f in glob(f"{d}/*.csv"):
        bname = Path(f).stem
        df = pd.read_csv(f)
        for i in df.columns:
            v = df[i].values
            vehicle_model2data[(meta["vehicle_model"], i)].extend(v)

vert2acceleration_level = defaultdict(list)
vert2steering = defaultdict(list)
for (vert, key), v in vehicle_model2data.items():
    if key == "acceleration_level":
        vert2acceleration_level[vert].extend(v)
    elif key == "steering":
        vert2steering[vert].extend(v)
    else:
        print(f"Skip {vert}, {key}")


vert2feats = {
    k: [[a, s] for a, s in zip(v, vert2steering[k])]
    for k, v in vert2acceleration_level.items()
}

scaler_for_model = {}
for k, feats in vert2feats.items():
    feats = np.asarray(feats, dtype=float)
    scaler = StandardScaler()
    scaler.fit(feats)
    scaler_for_model[k] = scaler

print(scaler_for_model[1].scale_)
print(scaler_for_model[1].mean_)
print(scaler_for_model[1].var_)

with open("YandexCup2024v2/scaler.pkl", "wb") as f:
    pickle.dump(scaler_for_model, f)

for d in tqdm(DATA):
    with open(f"{d}/metadata.json") as f:
        meta = json.load(f)
    df = pd.read_csv(f"{d}/control.csv")
    normalized = scaler_for_model[meta["vehicle_model"]].transform(
        df[["acceleration_level", "steering"]].values
    )
    df["norm_acceleration_level"] = normalized[:, 0]
    df["norm_steering"] = normalized[:, 1]
    df.to_csv(f"{d}/control_norm_v1.csv", index=False)


print("---Quantisizing time to frames of constant duration---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import json
from glob import glob
from tqdm import tqdm
import os

NSECS_IN_SEC = 1000000000


def interpolate_2d(A_x, B_x, T_x, A_y, B_y):
    """Predicts T_y by linear interpolating A and B dots"""
    assert A_x <= B_x, f"{A_x=} must be less than {B_x}"
    assert A_x <= T_x, f"{A_x=} must be less than {T_x}"
    delta_y = (B_y - A_y) / (B_x - A_x) * (T_x - A_x)
    return A_y + delta_y


def inperpolate_dots(A_x, A, B_x, B, T_x):
    if A is None or B is None:
        # if A does not exist then just B returned
        return B
    assert len(A) == len(B), f"{len(A)=}, {len(B)=}"
    T = [interpolate_2d(A_x, B_x, T_x, A_y, B_y) for A_y, B_y in zip(A, B)]
    return T


# 20000000нс == 20мс
def quantisize_time(
    time, variables, start_time=None, frame_step=20000000, interpolate_type="linear"
):
    # time is (T,)
    # variables is (T, N)
    assert time.shape[0] == variables.shape[0], f"{time.shape=} {variables.shape=}"
    # print(f"Frame step is {frame_step/NSECS_IN_SEC}s == {frame_step/NSECS_IN_SEC*1000}mc")
    if start_time is None:
        start_time = np.min(time)
    # moves start to zero
    time = time - start_time
    num_frames = math.ceil(np.max(time) / frame_step)
    dots_in_frames = [[] for _ in range(num_frames)]
    frames = []
    prev_t = None
    prev_dot = None
    prev_frame_id = -1
    for t, v in zip(time, variables):
        frame_id = int(t // frame_step)
        if frame_id != prev_frame_id:
            for i in range(prev_frame_id + 1, frame_id + 1):
                frame_vars = inperpolate_dots(prev_t, prev_dot, t, v, i * frame_step)
                frames.append([i, *frame_vars])
        prev_t = t
        prev_dot = v
        prev_frame_id = frame_id
    return (start_time, np.asarray(frames, dtype=float))

VERSION='v2'
def quant_dir(d):
    cdf = pd.read_csv(f'{d}/control_norm_v1.csv')
    ldf = pd.read_csv(f'{d}/localization.csv')
    st = min(cdf.stamp_ns.min(), ldf.stamp_ns.min())
    _, data = quantisize_time(cdf.values[:, 0], cdf.values[:, 1:], start_time=st, frame_step=20000000)
    df = pd.DataFrame(data, columns=['frame_id', *cdf.columns[1:]])
    #df['stamp_ms'] = df['stamp_ns'].astype(float) / NSECS_IN_SEC * 1000
    df.to_csv(f'{d}/control_norm_v1-quant20mc_{VERSION}.csv', index=False)
    _, data = quantisize_time(ldf.values[:, 0], ldf.values[:, 1:], start_time=st, frame_step=20000000)
    df = pd.DataFrame(data, columns=['frame_id', *ldf.columns[1:]])
    #df['stamp_ms'] = df['stamp_ns'].astype(float) / NSECS_IN_SEC * 1000
    df.to_csv(f'{d}/localization-quant20mc_{VERSION}.csv', index=False)

    f=f'{d}/requested_stamps.csv'
    if os.path.exists(f):
        df = pd.read_csv(f'{d}/requested_stamps.csv')
        df.stamp_ns = df.stamp_ns - st
        df.to_csv(f'{d}/requested_stamps-quant20mc_{VERSION}.csv', index=False)
    with open(f"{d}/quant20mc_{VERSION}", 'w') as f:
        f.write(f'{st}')
dataset_info = []
for d in tqdm(DATA):
    quant_dir(d)


print("---Packing data into WebDataset format---")
import os
import webdataset as wds
import logging
import numpy
import json
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from itertools import islice
from glob import glob
from torch.utils.data import DataLoader, IterableDataset
from webdataset.filters import pipelinefilter
from webdataset.utils import pytorch_worker_info


def make_speed_csv(d):
    localization_df = pd.read_csv(f"{d}/localization-quant20mc_v2.csv")

    # to predict: x1 = v1 + x0
    dx = localization_df["x"].values[1:] - localization_df["x"].values[:-1]
    localization_df["v_x"] = np.concatenate([dx[:1], dx], axis=0)
    dy = localization_df["y"].values[1:] - localization_df["y"].values[:-1]
    localization_df["v_y"] = np.concatenate([dy[:1], dy], axis=0)
    localization_df["mod_v"] = (
        localization_df["v_x"] ** 2 + localization_df["v_y"] ** 2
    ) ** (1 / 2)

    dyaw = localization_df["yaw"].values[1:] - localization_df["yaw"].values[:-1]
    localization_df["v_yaw"] = np.concatenate([[0], dyaw], axis=0)

    angle_mod = np.arctan2(
        np.abs(localization_df.v_y.values).tolist(),
        np.abs(localization_df.v_x.values).tolist(),
    )
    direction = localization_df.v_x.values < 0
    angle_mod[direction] = np.pi - angle_mod[direction]
    direction = localization_df.v_y.values < 0
    angle_mod[direction] = -angle_mod[direction]
    localization_df["v_direct"] = angle_mod

    localization_df.to_csv(f"{d}/speed-quant20mc_v2.csv", index=False)


dataset_info = []
for d in tqdm(DATA):
    make_speed_csv(d)


class DirLoader(IterableDataset):
    def __init__(
        self,
        root_dir,
        control_bname="control_norm_v1-quant20mc_v2.csv",
        req_bname=None,
        meta_bname="metadata.json",
        start_bname="quant20mc_v2",
        speed_bname="speed-quant20mc_v2.csv",
        seed=None,
        pad_value=-1000000,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.control_bname = control_bname
        self.req_bname = req_bname
        self.meta_bname = meta_bname
        self.start_bname = start_bname
        self.speed_bname = speed_bname
        self.seed = seed
        self.pad_value = pad_value
        self.dirlist = [*os.listdir(root_dir)]

    def __len__(self):
        return len(self.dirlist)

    def __iter__(self):
        dirlist = self.dirlist
        rank, world_size, worker, num_workers = pytorch_worker_info()
        assert world_size == 1, "Do not use this class for DDP"
        if num_workers > 1:
            full_len = len(dirlist)
            dirlist = list(islice(dirlist, worker, None, num_workers))
            logging.info(
                f"Subset for {worker} worker contains {len(dirlist)}/{full_len} annotations"
            )
            logging.debug(f"First dir is {dirlist[0]}")
        if len(dirlist) == 0:
            logging.warning(
                f"Zero len dirs list! {worker=}, {num_workers=}, {len(dirlist)=}, {len(self.dirlist)}"
            )
            return

        if self.seed is not None:
            random.Random(self.seed).shuffle(dirlist)
        for dbname in dirlist:
            d = f"{self.root_dir}/{dbname}"
            control_df = pd.read_csv(f"{d}/{self.control_bname}")
            speed_ds = pd.read_csv(f"{d}/{self.speed_bname}")
            assert (control_df.frame_id >= 0).all(), f"{control_df.frame_id=}"
            assert (speed_ds.frame_id >= 0).all(), f"{speed_ds.frame_id=}"
            df = control_df.set_index("frame_id").join(
                speed_ds.set_index("frame_id"), lsuffix="_c", rsuffix="_r"
            )
            with open(f"{d}/{self.meta_bname}") as f:
                meta = json.load(f)
            df = df.fillna(self.pad_value)
            assert np.allclose(
                df.index, np.arange(len(df))
            ), f"{len(df)=}, \n{df.index=}"
            out = {
                "__key__": f"{Path(self.root_dir).stem}__{Path(d).stem}",
                "frame_ids.pth": torch.from_numpy(df.index.values),
                "control_frame_ids.pth": torch.from_numpy(control_df.frame_id.values),
                "loc_frame_ids.pth": torch.from_numpy(speed_ds.frame_id.values),
                "acceleration_level.pth": torch.from_numpy(
                    df.norm_acceleration_level.values
                ),
                "steering.pth": torch.from_numpy(df.norm_steering.values),
                "x.pth": torch.from_numpy(df.x.values),
                "y.pth": torch.from_numpy(df.y.values),
                "z.pth": torch.from_numpy(df.z.values),
                "roll.pth": torch.from_numpy(df.roll.values),
                "pitch.pth": torch.from_numpy(df.pitch.values),
                "yaw.pth": torch.from_numpy(df.yaw.values),
                "v_x.pth": torch.from_numpy(df.v_x.values),
                "v_y.pth": torch.from_numpy(df.v_y.values),
                "mod_v.pth": torch.from_numpy(df.mod_v.values),
                "v_yaw.pth": torch.from_numpy(df.v_yaw.values),
                "v_direct.pth": torch.from_numpy(df.v_direct.values),
                "ride_date.txt": meta["ride_date"],
                "tires.pickle": meta["tires"],
                "vehicle_id.id": meta["vehicle_id"],
                "vehicle_model.id": meta["vehicle_model"],
                "vehicle_model_modification.id": meta["vehicle_model_modification"],
                "location_reference_point_id.id": meta["location_reference_point_id"],
            }
            if self.req_bname is not None and os.path.exists(f"{d}/{self.req_bname}"):
                req_df = pd.read_csv(f"{d}/{self.req_bname}")
                out["req_stamps.pth"] = torch.from_numpy(req_df.stamp_ns.values)
            if self.start_bname is not None:
                with open(f"{d}/{self.start_bname}") as f:
                    start_ns = int(float(f.read()))
                out["start_ns.id"] = start_ns

            yield out


for d in ['YandexCup2024v2/YaCupTest/']:
    out_d = f"exp_v2/{d}"
    os.makedirs(out_d, exist_ok=True)
    max_elements_per_shard = 50
    num_e = len(os.listdir(d))
    with wds.ShardWriter(f"{out_d}/dump-%06d.tar", maxcount=400) as sink:
        for e in tqdm(
            DataLoader(
                DirLoader(d, req_bname="requested_stamps-quant20mc_v2.csv", seed=None),
                batch_size=None,
                num_workers=6,
                sampler=None,
            ),
            total=num_e,
        ):
            sink.write(e)
    print(f"Done {out_d}")
