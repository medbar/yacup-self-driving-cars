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
        input_dim=10,
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.out_dim = out_dim
        self.dropout = 0.1
        self.num_layers = num_layers
        self.cnn_kernel = cnn_kernel
        self.padding = cnn_kernel // 2
        self.residual_w = residual_w
        # self.emb_proj = nn.Sequential(nn.Conv1d(input_dim, emb_dim, cnn_kernel, stride=1, padding=padding, padding_mode='replicate'),
        #                                nn.SiLU(),
        #                                nn.Conv1d(input_dim, emb_dim, cnn_kernel, stride=cnn_kernel, padding=padding, padding_mode='replicate'),
        #                                nn.SiLU())
        self.emb_proj = nn.Sequential(
            nn.Linear(input_dim * cnn_kernel, emb_dim), nn.SiLU()
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
        self.head = nn.Linear(emb_dim, out_dim * cnn_kernel)

    def forward(self, inputs_embeds, attention_mask=None):
        B, T, F = inputs_embeds.shape
        pad_left = 0
        if attention_mask is None:
            attention_mask = inputs_embeds.new_ones((B, T))
        assert attention_mask.shape == (B,T), f"{attention_mask.shape=}, {B=}, {T=}"
        if T % self.cnn_kernel != 0:
            pad_left = self.cnn_kernel - (T % self.cnn_kernel)
            inputs_embeds = torch.nn.functional.pad(
                inputs_embeds, (0, 0, pad_left, 0), value=0
            )
            attention_mask = torch.nn.functional.pad(
                                attention_mask, (pad_left, 0), value=False
            )
        embs = self.emb_proj(inputs_embeds.view(B, -1, F * self.cnn_kernel))
        embs_attention_mask = attention_mask.view(B, embs.shape[1], -1).any(dim=-1)
        sT = embs.shape[1]
        #logging.debug(f"{inputs_embeds.shape=}, {embs.shape=}, {sT=}")
        pos_emb = self.pos_emb.weight[:sT].repeat(B, 1, 1)
        src_mask = torch.triu(embs.new_full((sT, sT), float('-inf')), diagonal=1)
        embs_after_t = self.encoder(embs + pos_emb, mask=src_mask, src_key_padding_mask=~embs_attention_mask, is_causal=True)
        embs = embs * self.residual_w + embs_after_t
        logits = self.head(embs).view(B, -1, self.out_dim)[:, pad_left:, :]
        assert logits.shape == (
            B,
            T,
            self.out_dim,
        ), f"{logits.shape=}, {B=}, {T=}, {sT=}, {self.out_dim=}"
        return {
            "logits.pth": logits,
            "embs.pth": embs,
        }
