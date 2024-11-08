import logging
import json
import contextlib
import random
from typing import Optional
import torch
import torch.nn as nn
import math

# pip install torchtune
# from torchtune.module import RotaryPositionalEmbeddings


class Transformer(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(
        self,
        loc_dim=6,
        control_dim=4,
        # cnn_stride=3,
        cnn_kernel=5,
        emb_dim=256,
        out_dim=6,
        dropout=0.1,
        num_layers=4,
        proj_size=0,
        mask_after=None,
        mask_prob=0.5,
        mask_in_eval=False,
    ):
        super().__init__()
        self.mask_after = mask_after
        self.mask_prob = mask_prob
        self.mask_in_eval = mask_in_eval
        self.loc_dim = loc_dim
        self.control_dim = control_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.dropout = 0.1
        self.num_layers = num_layers
        self.cnn_kernel = cnn_kernel
        self.padding = cnn_kernel // 2
        self.trans_proj = nn.Sequential(
            nn.Linear(loc_dim * cnn_kernel + control_dim * cnn_kernel * 2, emb_dim),
            nn.SiLU(),
        )
        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            proj_size=proj_size,
        )
        head_dim = proj_size if proj_size else emb_dim
        self.head = nn.Sequential(
            nn.Linear(head_dim, out_dim * cnn_kernel),
        )

    def mask(self, emb):
        if self.mask_after is None:
            return emb
        B, T, F = emb.shape
        if self.mask_after >= T:
            return emb

        if self.training:
            mask = emb.new_empty(
                B, dtype=bool
            ).bernoulli_(1-self.mask_prob)
        elif self.mask_in_eval:
            mask = emb.new_zeros(B, dtype=bool)
        else:
            return emb
        l, r = emb[:, : self.mask_after], emb[:, self.mask_after :]
        #print(f"{mask.shape=}, {(~mask).sum()=}")
        return torch.cat([l, r * mask[:, None, None]], dim=1)

    def forward(self, loc, control_feats, attention_mask=None, h=None):
        # inputs_embeds = torch.concatenate([loc, control_feats], dim=-1)
        # inputs_embeds = inputs_embeds * attention_mask[:, :, None]
        loc = loc * attention_mask[:, :, None]
        control_feats = control_feats * attention_mask[:, :, None]
        B, T, F = loc.shape
        control_right_pad = 0
        if control_feats.shape[1] < loc.shape[1] + self.cnn_kernel:
            # logging.warning(f"{control_feats.shape[1]=} < {loc.shape[1]+self.cnn_kernel=}")
            control_right_pad = loc.shape[1] + self.cnn_kernel - control_feats.shape[1]
        pad_left = 0
        if attention_mask is None:
            attention_mask = loc.new_ones((B, T))
        assert attention_mask.shape == (B, T), f"{attention_mask.shape=}, {B=}, {T=}"
        if T % self.cnn_kernel != 0:
            pad_left = self.cnn_kernel - (T % self.cnn_kernel)
            loc = torch.nn.functional.pad(loc, (0, 0, pad_left, 0), value=0)
            attention_mask = torch.nn.functional.pad(
                attention_mask, (pad_left, 0), value=False
            )
        control_feats = torch.nn.functional.pad(
            control_feats, (0, 0, pad_left, control_right_pad), value=0
        ).view(B, -1, self.control_dim * self.cnn_kernel)

        dloc = torch.nn.functional.pad(loc[:, 1:] - loc[:, :-1], (0, 0, 1, 0))
        anchor_loc = loc.view(B, -1, self.cnn_kernel, F)[:, :, -1:, :]  # .mean(dim=2)
        dloc = self.mask(dloc)
        dloc = dloc.view(B, -1, self.cnn_kernel * F)
        # assert (
        #    anchor_loc.shape[1] == control_feats.shape[1] - 1
        # ), f"{anchor_loc.shape[1]=}, {control_feats.shape[1]=}"
        embs = torch.concatenate(
            [dloc, control_feats[:, :-1], control_feats[:, 1:]], dim=-1
        )
        embs = self.trans_proj(embs)
        sT = embs.shape[1]
        embs, h = self.encoder(embs, h)
        # embs = embs * self.residual_w + embs_after_t
        logits_dloc = self.head(embs).view(B, -1, self.cnn_kernel, self.out_dim)
        logits_danchor = logits_dloc.cumsum(dim=2)
        logits = (logits_danchor + anchor_loc).view(B, loc.shape[1], self.out_dim)
        logits = logits[:, pad_left:, :]
        assert logits.shape == (
            B,
            T,
            self.out_dim,
        ), f"{logits.shape=}, {B=}, {T=}, {sT=}, {self.out_dim=}"
        return {"logits.pth": logits,
                "embs.pth": embs,
                "h.pth": h,
                "anchor_loc.pth": anchor_loc,
                "logits_danchor.pth": logits_danchor,
                "pad_left.id": pad_left
               }

    def step(self):
        """return autoregress step"""
        return self.cnn_kernel
