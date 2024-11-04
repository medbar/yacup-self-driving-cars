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
        yaw_last=False,
        control_dim=4,
        emb_dim=256,
        ff_dim=512,
        dropout=0.1,
        num_encoder_layers=2,
        num_decoder_layers=6,
        nhead=4,
        max_encoder_len=250,
        max_decoder_len=760,
        cnn_kernel=10,
    ):
        super().__init__()
        self.loc_dim = loc_dim
        self.yaw_last = yaw_last
        self.cnn_kernel = cnn_kernel
        self.control_dim = control_dim
        self.emb_dim = emb_dim
        self.max_decoder_len = max_decoder_len
        self.loc_prof = nn.Linear(loc_dim*cnn_kernel, emb_dim)
        self.control_proj = nn.Linear(control_dim * cnn_kernel, emb_dim)
        self.encoder_proj = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 2, emb_dim))
        self.pos_enc_emb = nn.Embedding(max_encoder_len // cnn_kernel + 1, emb_dim)
        self.pos_emb = nn.Embedding(max_decoder_len // cnn_kernel + 1, emb_dim)
        self.aed = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(emb_dim, loc_dim * cnn_kernel, bias=False)

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
        ### DLOC
        dloc_attention_mask = loc_attention_mask[:, 1:] & loc_attention_mask[:, :-1]
        dloc = (loc[:, 1:, :] - loc[:, :-1, :]) * dloc_attention_mask[:, :, None]
        B, T, F = dloc.shape
        pad_left = 0
        if dloc.shape[1] % self.cnn_kernel != 0:
            pad_left = self.cnn_kernel - (dloc.shape[1] % self.cnn_kernel)
            dloc = torch.nn.functional.pad(dloc, (0, 0, pad_left, 0), value=0)
            dloc_attention_mask = torch.nn.functional.pad(
                dloc_attention_mask, (pad_left, 0), value=False
            )
        dloc = dloc.view(B, -1, self.cnn_kernel * F)
        dloc_attention_mask = dloc_attention_mask.view(
            B, dloc.shape[1], self.cnn_kernel
        ).any(dim=-1)
        eT = dloc.shape[1]
        #dloc_emb = nn.functional.linear(dloc, self.head.weight.T)
        dloc_emb = self.loc_prof(dloc)
        ## END

        #### CONTROL
        target_T = control_feats.shape[1]
        control_feats = control_feats * cf_attention_mask[:, :, None]
        control_right_pad = 0
        if (target_T + pad_left) % self.cnn_kernel != 0:
            control_right_pad = (
                self.cnn_kernel - (target_T + pad_left) % self.cnn_kernel
            )
        if pad_left != 0 or control_right_pad != 0:
            control_feats = torch.nn.functional.pad(
                control_feats, (0, 0, pad_left, control_right_pad), mode="replicate"
            )
            cf_attention_mask = torch.nn.functional.pad(
                cf_attention_mask, (pad_left, control_right_pad), value=False
            )
        control_feats = control_feats.view(
            B, -1, self.cnn_kernel * control_feats.shape[-1]
        )
        cf_attention_mask = cf_attention_mask.view(
            B, control_feats.shape[1], self.cnn_kernel
        ).any(dim=-1)
        cf_emb = self.control_proj(control_feats)
        cf_encoder_emb, cf_decoder_emb = cf_emb[:, :eT, :], cf_emb[:, eT:, :]
        ### END

        encoder_attention_mask = cf_attention_mask[:, :eT] & dloc_attention_mask
        decoder_attention_mask = cf_attention_mask[:, eT:]


        pos_enc_emb = self.pos_enc_emb.weight[:eT].repeat(B, 1, 1)
        encoder_in = self.encoder_proj(torch.cat([dloc_emb, cf_encoder_emb], dim=2)) + pos_enc_emb
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
        pred_dloc = pred_dloc.view(B, -1, F)
        pred_diff_loc_from_zero = pred_dloc.cumsum(dim=1)
        pred_loc = loc[:, -1:] + pred_diff_loc_from_zero  # skip last predict (vn+1)
        pred_loc = torch.cat([loc, pred_loc], dim=1)[:, :target_T]
        return {
            "loc.pth": pred_loc,
            "dloc.pth": dloc,
        }

    def step(self):
        """return autoregress step"""
        return self.max_decoder_len
