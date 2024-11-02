import torch
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd


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


def predict_and_validate(module, dl, out_f, device="cpu", limit=20, start_frame=250, end_frame=1000):
    module = module.to(device).eval()
    losses = []
    total = 0
    pbar = tqdm(dl)
    refs = []
    hyps = []
    criterion = torch.nn.MSELoss()
    with open(out_f, "w") as f:
        for batch_idx, batch in enumerate(pbar):
            orig_loc = batch["loc.pth"][:, :end_frame]
            batch["loc.pth"] = orig_loc[:, :start_frame, :]
            batch["control_feats.pth"] = batch["control_feats.pth"][:, :end_frame+1]
            loc = module.predict_step(batch)["loc.pth"][:, :end_frame]
            assert loc.shape == orig_loc.shape, f"{loc.shape=}. {orig_loc.shape=}"
            mask = (orig_loc != module.pad_value).any(dim=-1)
            mask[:, : orig_loc.shape[1] // 4] = False
            loc_masked = loc[mask]
            ref_masked = orig_loc[mask]
            loss = criterion(loc_masked, ref_masked).cpu().item()
            refs.extend(ref_masked[:, [0, 1, -1]])
            hyps.extend(loc_masked[:, [0, 1, -1]])
            losses.append(loss * orig_loc.shape[0])
            total += orig_loc.shape[0]
            # logging.info(f"{loss=}")
            pbar.set_description(f"loss={loss}, {batch_idx=}")
            for l in torch.concatenate([orig_loc, loc], dim=-1).tolist():
                f.write("\n".join(" ".join(map(str, line)) for line in l))
                f.write("\n\n")
            f.flush()
            if batch_idx == limit:
                logging.info(f"Stopping. {limit=}")
                break
        loss = sum(losses) / total
        s = f"Average loss is {loss}"
        print(s)
        f.write(s)
        metric = calculate_metric_testcase(
            torch.stack(hyps), torch.stack(refs)
        )
        s = f"Metric is {metric}"
        print(s)
        f.write(s)
