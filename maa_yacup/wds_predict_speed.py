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


@torch.no_grad()
@torch.inference_mode()
def _predict(
    dl,
    module,
    device="cpu",
    limit=-1,
    start_frame=250,
    end_frame=1000,
    v_x_id=0,
    v_y_id=1,
    debug_check=False,
):
    module = module.to(device).eval()
    pbar = tqdm(dl)
    criterion = torch.nn.MSELoss()
    step = module.model.step()
    for batch_idx, batch in enumerate(pbar):
        orig_loc = batch["loc.pth"][:, :end_frame]
        start_loc = orig_loc[:, :start_frame]
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
        if debug_check:
            logging.info(f"Replace model output with ideal speeds")
            out = {
                "loc.pth": torch.cat([orig_loc[:, step:], orig_loc[:, -step:]], dim=1),
                "losses.pth": out["losses.pth"],
            }
        # ploc: M(V_n) = V_{n+step}
        ploc = out["loc.pth"].cpu()
        B, T, F = ploc.shape
        batch["predicted_speed_loc.pth"] = ploc
        chunks_ploc = torch.nn.functional.pad(
            ploc, (0, 0, 0, step - (ploc.shape[1] % step))
        ).view(B, -1, step, F)
        num_chunks = chunks_ploc.shape[1]
        # add_to_start_point = chunks_ploc.cumsum(dim=1)
        # add_to_start_point = ploc[:, start_frame:, :].cumsum(dim=1)
        # X_n = X_{n-step} + M(V_{n-2*step})
        #
        start_chunks = start_frame // step
        startx = batch["x.pth"][:, : start_chunks * step].view(B, -1, step)
        starty = batch["y.pth"][:, : start_chunks * step].view(B, -1, step)
        assert startx.shape == starty.shape, f"{startx.shape=} {starty.shape=}"
        start_chunk_id = startx.shape[1]
        pred_xy = startx.new_zeros(
            (
                B,
                num_chunks + 2,
                step,
                F
            ),
        )
        pred_xy[:, :start_chunk_id, :, v_x_id] = startx
        pred_xy[:, :start_chunk_id, :, v_y_id] = starty
        # after start_chunk_id X_2== x_1 + M(v_0)
        #pred_xy[:, start_chunk_id:] = pred_xy[:, start_chunk_id - 1]
        add_to_start_point = chunks_ploc[:, start_chunk_id - 2 :].cumsum(dim=1)
        pred_xy[:, start_chunk_id:] = (
            pred_xy[:, start_chunk_id - 1 : start_chunk_id] + add_to_start_point
        )

        # start_x = batch["x.pth"][:, :start_frame]
        # start_y = batch["y.pth"][:, :start_frame]
        # start_chunk_x = start_x[:, -step:]
        # start_chunk_y = start_y[:, -step:]
        # pred_x = torch.cat(
        #    [start_x, start_chunk_x + add_to_start_point[:, :, v_x_id]], dim=1
        # )
        # pred_y = torch.cat(
        #    [start_y, start_chunk_y + add_to_start_point[:, :, v_y_id]], dim=1
        # )
        pred_x = pred_xy[:, :, :, v_x_id].view(B, -1)[:, :end_frame]
        pred_y = pred_xy[:, :, :, v_y_id].view(B, -1)[:, :end_frame]
        y2 = pred_y[:, 2:] - pred_y[:, :-2]
        x2 = pred_x[:, 2:] - pred_x[:, :-2]
        angle2_mod = torch.atan2(y2, x2)
        # direction = x2 < 0
        # angle2_mod[direction] = np.pi - angle2_mod[direction]
        # direction = y2 < 0
        # angle2_mod[direction] = - angle2_mod[direction]
        pred_yaw = torch.cat(
            [angle2_mod[:, :1], angle2_mod, angle2_mod[:, -1:]], axis=1
        )
        batch["predicted.pth"] = torch.stack([pred_x, pred_y, pred_yaw], axis=-1)

        assert (
            batch["predicted.pth"].shape[1]
            == batch["x.pth"].shape[1]
            == batch["y.pth"].shape[1]
            == batch["yaw.pth"].shape[1]
        ), f'{batch["predicted.pth"].shape[1]=}, {batch["x.pth"].shape[1]=},  {batch["y.pth"].shape[1]=}, {batch["yaw.pth"].shape[1]=}'
        # batch['gen_loss.pth'] = out["loss.pth"].cpu()
        mse = criterion(batch["predicted_speed_loc.pth"], orig_loc)
        pbar.set_description(f"{out['losses.pth'][-1]=}, {mse=}")
        yield batch
        if batch_idx == limit:
            logging.info(f"Stopping. {limit=}")
            break


predict = pipelinefilter(_predict)


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
