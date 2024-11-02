import torch
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import webdataset as wds
from webdataset.filters import pipelinefilter

SEGMENT_LENGTH = 1.0


def yaw_direction(yaw_value):
    return np.array([np.cos(yaw_value), np.sin(yaw_value)])


def build_car_points(x_y_yaw):
    directions = np.vstack(yaw_direction(x_y_yaw[:, -1]))
    front_points = x_y_yaw[:, :-1] + SEGMENT_LENGTH * directions.T
    points = np.vstack([x_y_yaw[:, :-1], front_points])
    return points


def calculate_metric_testcase(hyp_x_y_yaw, ref_x_y_yaw):
    assert (
        hyp_x_y_yaw.shape == ref_x_y_yaw.shape
    ), f"{hyp_x_y_yaw.shape=} {ref_x_y_yaw.shape=}"
    assert len(hyp_x_y_yaw.shape) == 2, f"{hyp_x_y_yaw.shape=}"
    if isinstance(hyp_x_y_yaw, torch.Tensor):
        hyp_x_y_yaw = hyp_x_y_yaw.numpy()
    if isinstance(ref_x_y_yaw, torch.Tensor):
        ref_x_y_yaw = ref_x_y_yaw.numpy()

    points_gt = build_car_points(ref_x_y_yaw)
    points_pred = build_car_points(hyp_x_y_yaw)
    xy_mse = np.mean(np.sqrt(2.0 * (ref_x_y_yaw - hyp_x_y_yaw) ** 2))
    # metric_per_each = np.mean(np.sqrt(2.0 * (points_gt - points_pred) ** 2), axis=0)
    metric = np.mean(np.sqrt(2.0 * np.mean((points_gt - points_pred) ** 2, axis=1)))
    return metric, xy_mse



def _measure_score(
    data,
    predicted_key="predicted.pth",
    ref_x_key="x.pth",
    ref_y_key="y.pth",
    ref_yaw_key="yaw.pth",
    pad_value=-1000000,
):
    ref = []
    hyp = []
    for e in data:
        mask = (e["loc.pth"] != pad_value).all(dim=-1)
        pred = e[predicted_key][mask].cpu().numpy()
        source = (
            torch.stack(
                [e[ref_x_key][mask], e[ref_y_key][mask], e[ref_yaw_key][mask]], dim=-1
            )
            .cpu()
            .numpy()
        )
        assert len(pred) == len(
            source
        ), f"{pred.shape=}, {mask.shape=}, {e[predicted_key].shape=}, {e[ref_x_key].shape=}"
        ref.append(source)
        hyp.append(pred)
        m, me = calculate_metric_testcase(pred, source)
        logging.info(f"Metric: {e['__key__']=}, {m=}, {me=}")
        e["metric.npy"] = m
        e["metrics.npy"] = me
        yield e
    m, me = calculate_metric_testcase(
        np.concatenate(hyp, axis=0), np.concatenate(ref, axis=0)
    )
    print(f"Metric {m}, for each axis separatly {me}")


measure_score = pipelinefilter(_measure_score)
