import logging
import json
import contextlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ):
        super().__init__()
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

        mask = (batch["loc.pth"] != self.pad_value).any(dim=-1)
        attention_mask = (batch["control_feats.pth"] != self.pad_value).any(dim=-1) & mask
        #mask = torch.nn.functional.pad(mask[:, 1:], (0, 1), value=False)
        # the first frame as a zero point
        zero_point = batch["loc.pth"][:, 0]
        assert (zero_point > self.pad_value).all() , f"{zero_point=}, {self.pad_value}"
        loc = batch['loc.pth'] - zero_point[:, None, :]
        #labels = torch.nn.functional.pad(
        #    loc[:, 1:], (0, 0, 0, 1), value=self.pad_value
        #)
        inputs_embeds = torch.concatenate(
            [loc, batch["control_feats.pth"]], dim=-1
        )
        inputs_embeds = inputs_embeds * attention_mask[:, :, None]
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        #logging.debug(f"{outputs['logits.pth'].shape=}, {outputs['embs.pth'].shape=}")
        outputs['loc.pth'] = outputs['logits.pth'] + zero_point[:, None, :]
        mask = mask[:, 1:]
        outputs["num.pth"] = mask.sum()
        logits_w_mask = outputs["logits.pth"][:, :-1][mask]
        labels_w_mask = loc[:, 1:][mask]
        loss = self.criterion(logits_w_mask, labels_w_mask)
        outputs["loss.pth"] = loss
        logging.debug(f"{loss=}, {outputs['num.pth']=}, {logits_w_mask[-6:]=}, {labels_w_mask[-6:]=}")
        # DEBUG
        if self.DEBUG:
            torch.save(
                {
                    "batch": batch,
                    "attention_mask": attention_mask,
                    "mask": mask,
                    "inputs_embeds": inputs_embeds,
                    "labels_w_mask": labels_w_mask,
                    "outputs": outputs,
                    "mask": mask,
                    "logits_w_mask": logits_w_mask,
                    "loss": loss,
                },
                f'tmp/{batch["__key__"][0]}.pth',
            )
        return outputs

    def training_step(self, batch, batch_idx):
        logging.debug(f"Process {batch_idx}, {batch['loc.pth'].shape=}")
        outputs = self.forward(batch)
        loss = outputs["loss.pth"]
        self.log(
            "loss_train",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=outputs["num.pth"],
        )
        logging.debug(f"loss_train for {batch_idx} is {loss.item()}")
        return {
            "loss": loss,
        }

    @torch.no_grad()
    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # logging.debug(f"valid started for {batch_idx}")
        outputs = self.forward(batch)
        self.log(
            "loss_valid",
            outputs["loss.pth"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=outputs["num.pth"],
        )
        if self.debug_dir is not None:
            # remove last symbol, because  rprompt applied as a first item in labels in the forward
            # removes the last because the rprompt[-1] is used as the first element in labels in the forward
            preds = outputs["loc.pth"]
            self.validation_step_outputs.append(
                (
                    batch_idx,
                    batch["loc.pth"].cpu(),
                    preds.cpu(),
                    outputs["loss.pth"].cpu().item(),
                )
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
                    if batch_idx != 0:
                        # all batches is too long
                        continue
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
        B, T, _ = batch["loc.pth"]
        loc = batch["loc.pth"]
        control = batch["control_feats.pth"]
        maxT = control.shape[1]
        losses = []
        logging.info(f"Start from {T} and generate untill {maxT}")
        for i in range(T, maxT):
            pred_out = self.forward({"loc.pth": loc, "control_feats.pth": control})
            losses.append(pred_out['loss.pth'])
            next_frame = pred_out["loc.pth"][:, -1:, :]
            loc = torch.concatenate([loc, next_frame], dim=1)
            logging.debug(f"{loc.shape=} {control.shape=}, {losses=}")
        return loc

    def inplace_load_from_checkpoint(self, ckpt_path):
        data = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missed, unexpected = self.load_state_dict(data, strict=self.strict_loading)
        logging.debug(f"Loaded model from {ckpt_path}\n{missed=}\n{unexpected=}")
        return self
