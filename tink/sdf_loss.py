# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.append("./DeepSDF_OakInk")

import DeepSDF_OakInk.deep_sdf.workspace as ws
import torch

# import deep_sdf.workspace as ws
from termcolor import cprint


def sdf_loss(decoder, obj_latent, hand_verts, center, rescale):
    hand_verts = hand_verts - center[None]
    hand_verts = hand_verts / rescale
    latent = obj_latent.expand(hand_verts.shape[0], -1)
    inp = torch.cat([latent, hand_verts], 1)
    sdf = decoder(inp).squeeze(1)

    neg_sdf = sdf[sdf < 0]
    if neg_sdf.shape[0] == 0:
        return torch.tensor(0, device=neg_sdf.device)

    return torch.mean(torch.pow(neg_sdf, 2), dim=0)


def load_sdf_decoder(data_path, resume):
    specs = json.load(open(os.path.join(data_path, "specs.json"), "r"))
    arch = __import__("DeepSDF_OakInk.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(os.path.join(data_path, "network", ws.model_params_subdir, resume + ".pth"))
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()
    return decoder


def load_obj_latent(data_path, obj):
    sdf_code_path = os.path.join(data_path, ws.reconstructions_subdir, ws.reconstruction_codes_subdir)
    target = f"{obj}.pth"
    latent = torch.load(os.path.join(sdf_code_path, target))[0].cuda()
    latent.requires_grad = False
    return latent
