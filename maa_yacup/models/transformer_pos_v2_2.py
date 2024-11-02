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
        ff_dim=512,
        out_dim=6,
        dropout=0.1,
        num_layers=4,
        nhead=4,
        residual_w=0.2,
        max_len=3000,  # 20s, frame every 20mc
        mask_after=None
    ):
        super().__init__()
        self.loc_dim = loc_dim
        self.control_dim = control_dim
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.out_dim = out_dim
        self.dropout = 0.1
        self.num_layers = num_layers
        self.cnn_kernel = cnn_kernel
        self.padding = cnn_kernel // 2
        self.residual_w = residual_w
        self.mask_after = mask_after
        self.loc_proj = nn.Sequential(
            nn.Linear(loc_dim * cnn_kernel, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.SiLU(),
        )
        self.control_proj = nn.Sequential(
            nn.Linear(control_dim * cnn_kernel, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.SiLU(),
        )
        self.pos_emb = nn.Embedding(max_len // cnn_kernel + 1, emb_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=nn.functional.silu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, out_dim * cnn_kernel),
        )
    def mask_loc(self, loc):
        if self.mask_after is None:
            return loc
        raise NotImplementedError()

    def forward(self, loc, control_feats, attention_mask=None):
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
        )[:, : loc.shape[1] + self.cnn_kernel]
        loc = loc.view(B, -1, self.cnn_kernel, F)
        # последняя точка каждого фрейма - это точка отсчета для следующего фрейма
        anchor_locs = loc[:, :, -1:, :]
        loc = torch.cat([loc.new_zeros((B, 1, self.cnn_kernel, F)), loc[:, 1:] - anchor_locs[:, :-1]], dim=1)
        loc = self.mask_loc(loc * attention_mask.view(B, -1, self.cnn_kernel, 1))

        loc_embs = self.loc_proj(loc.view(B, -1, F * self.cnn_kernel))
        control_embs = self.control_proj(
            control_feats.view(B, -1, control_feats.shape[-1] * self.cnn_kernel)
        )
        embs = torch.concatenate(
            [(loc_embs + control_embs[:, :-1]), control_embs[:, 1:]], dim=2
        )
        sT = embs.shape[1]
        embs_attention_mask = attention_mask.view(B, sT, -1).any(dim=-1)
        # logging.debug(f"{inputs_embeds.shape=}, {embs.shape=}, {sT=}")
        pos_emb = self.pos_emb.weight[:sT].repeat(B, 1, 1)
        src_mask = torch.triu(embs.new_full((sT, sT), float("-inf")), diagonal=1)
        embs_after_t = self.encoder(
            embs + pos_emb,
            mask=src_mask,
            src_key_padding_mask=~embs_attention_mask,
            is_causal=True,
        )
        embs = embs * self.residual_w + embs_after_t
        logits_delta = self.head(embs).view(B, -1, self.cnn_kernel, self.out_dim)
        logits = logits_delta  + anchor_locs
        logits = logits.view(B, -1, self.out_dim)[:, pad_left:, :]
        assert logits.shape == (
            B,
            T,
            self.out_dim,
        ), f"{logits.shape=}, {B=}, {T=}, {sT=}, {self.out_dim=}"
        return {
            "logits.pth": logits,
            "logits_delata.pth": logits_delta,
            "embs.pth": embs,
        }

    def step(self):
        """return autoregress step"""
        return self.cnn_kernel
