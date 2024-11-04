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
        loc_dim=2,
        control_dim=4,
        emb_dim=256,
        ff_dim=512,
        dropout=0.1,
        num_encoder_layers=2,
        num_decoder_layers=6,
        nhead=4,
        max_decoder_len=760,
    ):
        super().__init__()
        self.loc_dim = loc_dim
        self.control_dim = control_dim
        self.emb_dim = emb_dim
        self.max_decoder_len = max_decoder_len
        self.encoder_proj = nn.Linear(emb_dim * 2, emb_dim)
        self.control_proj = nn.Sequential(
            nn.Conv1d(control_dim, emb_dim // 4, 3, padding="same"),
            nn.BatchNorm1d(emb_dim // 4),
            nn.ReLU(),
            nn.Conv1d(emb_dim // 4, emb_dim // 2, 3, padding="same"),
            nn.BatchNorm1d(emb_dim // 2),
            nn.ReLU(),
            nn.Conv1d(emb_dim // 2, emb_dim, 3, padding="same"),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        self.pos_emb = nn.Embedding(max_decoder_len, emb_dim)
        self.aed = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(emb_dim, loc_dim, bias=False)

    def forward(
        self,
        loc,
        control_feats,
        loc_attention_mask=None,
        cf_attention_mask=None,
    ):
        """loc is encoder input (250 default values)"""
        if loc_attention_mask is None:
            loc_attention_mask = loc.new_ones((loc.shape[0], loc.shape[1]), dtype=bool)
        if cf_attention_mask is None:
            cf_attention_mask = control_feats.new_ones(
                (control_feats.shape[0], control_feats.shape[1]), dtype=bool
            )
        # dloc = nn.functiona.pad(loc[:, 1:, :] - loc[:, :-1, :], (0,0,0,1), mode='replicate')
        dloc_attention_mask = loc_attention_mask[:, 1:] & loc_attention_mask[:, :-1]
        dloc = (loc[:, 1:, :] - loc[:, :-1, :]) * dloc_attention_mask[:, :, None]
        dloc_emb = nn.functional.linear(dloc, self.head.weight.T)
        B, eT, F = dloc.shape

        control_feats = control_feats * cf_attention_mask[:, :, None]
        cf_emb = self.control_proj(control_feats.transpose(1, 2)).transpose(1, 2)
        cf_encoder_emb, cf_decoder_emb = cf_emb[:, :eT, :], cf_emb[:, eT:, :]

        encoder_attention_mask = cf_attention_mask[:, :eT] & dloc_attention_mask
        decoder_attention_mask = cf_attention_mask[:, eT:]

        encoder_in = self.encoder_proj(torch.cat([dloc_emb, cf_encoder_emb], dim=2))
        sT = cf_decoder_emb.shape[1]
        pos_emb = self.pos_emb.weight[:sT].repeat(B, 1, 1)
        decoder_in = cf_decoder_emb + pos_emb
        decoder_out = self.aed(
            src=encoder_in,
            tgt=decoder_in,
            src_key_padding_mask=~encoder_attention_mask,
            tgt_key_padding_mask=~decoder_attention_mask,
            memory_key_padding_mask=~encoder_attention_mask,
        )
        pred_dloc = self.head(decoder_out)
        pred_diff_loc_from_zero = pred_dloc.cumsum(dim=1)
        pred_loc = (
            loc[:, -1:] + pred_diff_loc_from_zero[:, :-1]
        )  # skip last predict (vn+1)
        pred_loc = torch.cat([loc, pred_loc], dim=1)
        assert (
            pred_loc.shape[1] == control_feats.shape[1]
        ), f"{pred_loc.shape=}, {control_feats.shape=}, {eT=}, {sT=}"
        return {
            "loc.pth": pred_loc,
            "dloc.pth": dloc,
        }

    def step(self):
        """return autoregress step"""
        return self.max_decoder_len
