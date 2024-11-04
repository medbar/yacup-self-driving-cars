import logging
import json
import contextlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from pathlib import Path


from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric

from pytorch_lightning import LightningModule


class CoordsPredictorAR(LightningModule):
    # enable loading from ckpt without LLM weights
    strict_loading = False

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(
        self,
        model,
        criterion=None,
        optimizers_config=None,
        debug_dir=None,
        DEBUG=False,
        pad_value=-1000000,
        prefix_len=245,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.DEBUG = DEBUG
        self.model = model
        if criterion is None:
            criterion = torch.nn.MSELoss()

        self.criterion = criterion
        self.debug_dir = debug_dir
        self.validation_step_outputs = []
        self.optimizers_config = optimizers_config
        self.pad_value = pad_value

    def set_optimizers_config(self, optimizers_config):
        self.optimizers_config = optimizers_config

    def configure_optimizers(self):
        return self.optimizers_config

    def forward(self, batch):
        """
        batch = {
        "loc.pth": [BxTxCoords] coords,
        "loc_attention_mask.pth": [BxT] attention mask. 1 not mask, 0 - masking this element
        "control_feats.pth" : [B x T X Feats], input features
        "control_feats_attention_mask.pth" : [B x T]
        }
        """

        control_feats = batch["control_feats.pth"]
        cf_attention_mask = (control_feats != self.pad_value).any(dim=-1)
        loc_encoder = batch["loc_encoder.pth"]
        outputs = self.model(
            loc=loc_encoder,
            control_feats=control_feats,
            loc_attention_mask=(loc_encoder != self.pad_value).any(dim=-1),
            cf_attention_mask=cf_attention_mask,
        )
        if "loc.pth" in batch:
            prefix_len = loc_encoder.shape[1]
            ideal_loc = batch["loc.pth"][:, prefix_len:]
            pred_loc = outputs["loc.pth"][:, prefix_len:]
            mask = (ideal_loc != self.pad_value).any(dim=-1)
            mask = (mask & cf_attention_mask[:, prefix_len:])
            hyp_w_mask = pred_loc[mask]
            ref_w_mask = ideal_loc[mask]
            loss = self.criterion(hyp_w_mask, ref_w_mask)
            outputs["loss.pth"] = loss
            outputs["num.pth"] = mask.sum()
        else:
            loss = 0
            outputs["num.pth"] = 0
        if self.DEBUG:
            torch.save(
                {
                    "batch": batch,
                    "outputs": outputs,
                    "loss": loss,
                },
                f'tmp/{batch["__key__"][0]}.pth',
            )
        return outputs

    def training_step(self, batch, batch_idx):
        # logging.debug(f"Process {batch_idx}, {batch['loc.pth'].shape=}")
        batch["loc_encoder.pth"] = batch["loc.pth"][:, : self.prefix_len]
        outputs = self.forward(batch)
        loss = outputs["loss.pth"]
        self.log(
            "loss_train",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=outputs["num.pth"],
        )
        # logging.debug(f"loss_train for {batch_idx} is {loss.item()}")
        return {
            "loss": loss,
        }

    @torch.no_grad()
    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # logging.debug(f"valid started for {batch_idx}")
        # outputs = self.forward(batch)
        batch["loc_encoder.pth"] = batch["loc.pth"][:, : self.prefix_len]
        outputs = self.forward(batch)
        loss = outputs["loss.pth"]

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=outputs["num.pth"],
        )
        if self.debug_dir is not None and batch_idx == 0:
            # all batches is too long
            preds = outputs["loc.pth"]
            self.validation_step_outputs.append(
                (batch_idx, batch["loc.pth"].cpu(), preds.cpu(), loss.cpu().item())
            )

    def on_validation_epoch_end(self):
        if self.debug_dir is not None:
            Path(self.debug_dir).mkdir(exist_ok=True)
            fname = str(time.strftime("%Y-%m-%d-%H:%M:%S.txt", time.gmtime()))
            with open(f"{self.debug_dir}/{fname}", "w") as f:
                for (
                    batch_idx,
                    ref_batch,
                    hyp_batch,
                    loss,
                ) in self.validation_step_outputs:
                    f.write(f"{batch_idx=} {loss=}\n")
                    for ref, hyp in zip(ref_batch, hyp_batch):
                        for r, h in zip(ref, hyp):
                            f.write(
                                " ".join(f"{e.item()}" for e in r)
                                + " | "
                                + " ".join(f"{e.item()}" for e in h)
                                + "\n"
                            )
                        f.write("\n\n")
        self.validation_step_outputs.clear()  # free memory

    @torch.no_grad()
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx=None, dataloader_idx=0):
        assert "loc_encoder.pth" in batch, f"{batch.keys()=}"
        return self.forward(batch)

    def inplace_load_from_checkpoint(self, ckpt_path):
        data = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missed, unexpected = self.load_state_dict(data, strict=self.strict_loading)
        logging.debug(f"Loaded model from {ckpt_path}\n{missed=}\n{unexpected=}")
        return self
