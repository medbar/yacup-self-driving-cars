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
    points_gt = build_car_points(ref_x_y_yaw.numpy())
    points_pred = build_car_points(hyp_x_y_yaw.numpy())

    metric = np.mean(np.sqrt(2.0 * np.mean((points_gt - points_pred) ** 2, axis=1)))
    return metric


@torch.no_grad()
@torch.inference_mode()
def _predict(dl, module, device="cpu", limit=-1, start_frame="auto", end_frame=1000, aprox_yaw=False):
    module = module.to(device).eval()
    pbar = tqdm(dl)
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(pbar):
        orig_loc = batch["loc.pth"][:, :end_frame]
        mask = orig_loc != module.pad_value
        if start_frame == "auto":
            # finding pad
            for batch_start_frame in range(end_frame):
                if not mask[:, batch_start_frame].all():
                    break
        else:
            batch_start_frame = start_frame
        start_loc = orig_loc[:, :start_frame, :]
        assert start_loc.shape[1] == start_frame, f"{orig_loc.shape=}"
        assert (
            start_loc[:, -1] != module.pad_value
        ).all(), f"{start_frame=}, {batch['__key__']=}, {start_loc[:, -1]=}"
        control_feats = batch["control_feats.pth"][:, :end_frame]
        out = module.predict_step(
            {
                "loc.pth": start_loc.to(device),
                "control_feats.pth": control_feats.to(device),
            }
        )
        batch["predicted.pth"] = out["loc.pth"][:, :end_frame].cpu()
        #batch["predicted.pth"][mask] = orig_loc[mask]
        if aprox_yaw:
            x = batch["predicted.pth"][:, :, 0]
            y = batch["predicted.pth"][:, :, 1]
            y2 = y[:, 2:] - y[:, :-2]
            x2 = x[:, 2:] - x[:, :-2]
            angle2_mod = torch.atan2(y2, x2)
            pred_yaw = torch.cat(
                    [angle2_mod[:, :1], angle2_mod, angle2_mod[:, -1:]], axis=1
            )
            batch["predicted.pth"] = torch.stack([x, y, pred_yaw], dim=2)
        # batch['gen_loss.pth'] = out["loss.pth"].cpu()
        pbar.set_description(f"{out['losses.pth'][-1]=}, {batch_start_frame=}")
        yield batch
        if batch_idx == limit:
            logging.info(f"Stopping. {limit=}")
            break


predict = pipelinefilter(_predict)



@torch.no_grad()
@torch.inference_mode()
def _predict_aed(dl, module, device="cpu", limit=-1, start_frame=245, end_frame=1000, aprox_yaw=True):
    module = module.to(device).eval()
    pbar = tqdm(dl)
    #pbar=dl
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(pbar):
        orig_loc = batch["loc.pth"][:, :end_frame]
        mask = orig_loc != module.pad_value
        start_loc = orig_loc[:, :start_frame, :]
        assert start_loc.shape[1] == start_frame, f"{orig_loc.shape=}"
        assert (
            start_loc[:, -1] != module.pad_value
        ).all(), f"{start_frame=}, {batch['__key__']=}, {start_loc[:, -1]=}"
        control_feats = batch["control_feats.pth"][:, :end_frame]
        out = module.predict_step(
            {
                "loc_encoder.pth": start_loc.to(device),
                "control_feats.pth": control_feats.to(device),
            }
        )
        batch["predicted.pth"] = out["loc.pth"][:, :end_frame].cpu()
        #batch["predicted.pth"][mask] = orig_loc[mask]
        if aprox_yaw:
            x = batch["predicted.pth"][:, :, 0]
            y = batch["predicted.pth"][:, :, 1]
            y2 = y[:, 2:] - y[:, :-2]
            x2 = x[:, 2:] - x[:, :-2]
            angle2_mod = torch.atan2(y2, x2)
            pred_yaw = torch.cat(
                    [angle2_mod[:, :1], angle2_mod, angle2_mod[:, -1:]], axis=1
            )
            batch["predicted.pth"] = torch.stack([x, y, pred_yaw], dim=2)
        # batch['gen_loss.pth'] = out["loss.pth"].cpu()
        #logging.info(f"{batch_idx=}, {batch['__key__']=}")
        yield batch
        if batch_idx == limit:
            logging.info(f"Stopping. {limit=}")
            break


predict_aed = pipelinefilter(_predict_aed)
