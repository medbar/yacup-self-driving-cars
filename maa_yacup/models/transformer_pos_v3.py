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
    

    def pad_input(self, loc_feats, control_feats, attention_mask=None):
        B, T, F = loc_feats.shape
        control_right_pad = 0
        if control_feats.shape[1] < loc_feats.shape[1] + self.cnn_kernel:
            # logging.warning(f"{control_feats.shape[1]=} < {moda_dyaw.shape[1]+self.cnn_kernel=}")
            control_right_pad = loc_feats.shape[1] + self.cnn_kernel - control_feats.shape[1]
        pad_left = 0
        if attention_mask is None:
            attention_mask = loc_feats.new_ones((B, T))
        assert attention_mask.shape == (B, T), f"{attention_mask.shape=}, {B=}, {T=}"
        if T % self.cnn_kernel != 0:
            pad_left = self.cnn_kernel - (T % self.cnn_kernel)
            loc_feats = torch.nn.functional.pad(loc_feats, (0, 0, pad_left, 0), value=0)
            attention_mask = torch.nn.functional.pad(
                attention_mask, (pad_left, 0), value=False
            )
        control_feats = torch.nn.functional.pad(
            control_feats, (0, 0, pad_left, control_right_pad), value=0
        )[:, : loc_feats.shape[1] + self.cnn_kernel]
        return loc_feats, control_feats, attention_mask


    def forward(self, loc, control_feats, attention_mask=None):
        # inputs_embeds = torch.concatenate([loc, control_feats], dim=-1)
        # inputs_embeds = inputs_embeds * attention_mask[:, :, None]
        loc = loc * attention_mask[:, :, None]
        control_feats = control_feats * attention_mask[:, :, None]
        # todo
        # в каждой точке x,y,yaw мы предсказываем ускорение модуля направления и угловое ускорение
        # для этого надо вычислисть среднюю скорость между точками A и B, после чего посчитать дельту сокрости для точка A
        # тоже самое надо сделать для угла поворота
        # модель должна предсказывать модуль ускорения по направлению машины + уголовую скорость изменения направления 
        avg_v = loc[:, 1:] - loc[:, :-1] * attention_mask[:, 1:, None]
        avg_a = (avg_v[:, 1:, :2] - avg_v[:, :-1, :2]) * attention_mask[:, 1:-1, None]
        # высчитываем проекцию ускорения на вектор направления машины
        alpha = torch.atan(avg_a[:, :, 1]/avg_a[:, :, 0])
        cos_da = torch.cos(loc[:, 1:-1, -1] - alpha)
        a_proj_yaw = cos_da * torch.sqrt(avg_a[:, :, :2].pow(2).sum(dim=-1))
        v_proj_yaw = cos_da * torch.sqrt(avg_v[:, :, :2].pow(2).sum(dim=-1))
        # паддим нулями
        a_proj_yaw = torch.nn.functional.pad(a_proj_yaw, (0,0, 1, 1), value=0)
        v_proj_yaw = torch.nn.functional.pad(v_proj_yaw, (0,0, 1, 0), value=0) # слева тк v это среднее между точками i и i+1. А нам в будущее смотреть нельзя
        moda_modv_dyaw = torch.concatenate([a_proj_yaw[:,:,None], v_proj_yaw[:, :, None], avg_v[:, :, -1:]], dim=-1)

        moda_modv_dyaw, control_feats, attention_mask = self.pad_input(moda_modv_dyaw, control_feats, attention_mask)
        loc_feats = moda_modv_dyaw[:, [0,2], :]
        B,T,F = loc_feats.shape
        #moda_dyaw = moda_dyaw.view(B, -1, self.cnn_kernel, F)
        moda_dyaw_embs = self.loc_proj(moda_dyaw.view(B, -1, F * self.cnn_kernel))
        control_embs = self.control_proj(
            control_feats.view(B, -1, control_feats.shape[-1] * self.cnn_kernel)
        )
        embs = torch.concatenate(
            [(moda_dyaw_embs + control_embs[:, :-1]), control_embs[:, 1:]], dim=2
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
        logits = self.head(embs).view(B, -1, self.cnn_kernel, self.out_dim)
        #logits = logits_delta + anchor_moda_dyaws
        logits = logits.view(B, -1, self.out_dim)[:, pad_left:, :]
        assert logits.shape == (
            B,
            T,
            self.out_dim,
        ), f"{logits.shape=}, {B=}, {T=}, {sT=}, {self.out_dim=}"

        # reconstruct loc

        return {
            "logits.pth": logits,
            "logits_delata.pth": logits_delta,
            "embs.pth": embs,
        }

    def step(self):
        """return autoregress step"""
        return self.cnn_kernel
