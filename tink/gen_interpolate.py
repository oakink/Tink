import json
import os
import sys

sys.path.append("./DeepSDF_OakInk")

import DeepSDF_OakInk.deep_sdf.workspace as ws
import numpy as np
import torch
from DeepSDF_OakInk.deep_sdf.mesh import create_mesh
from tqdm import tqdm
from termcolor import cprint


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source", "-s", type=str, required=False)
    parser.add_argument("--target", "-t", type=str, required=False)
    parser.add_argument("--interpolate", "-i", default=10, type=int)
    parser.add_argument("--resume", type=str, default="latest")
    arg = parser.parse_args()

    assert arg.all or (arg.source and arg.target)

    handle_list = []

    if arg.all:
        split = json.load(open(os.path.join(arg.data, "split.json"), "r"))
        for cid in split.keys():
            real_ids = split[cid]["real"]
            virtual_ids = split[cid]["virtual"]
            for r_oid in real_ids:
                for v_oid in virtual_ids:
                    handle_list.append((r_oid, v_oid))
    else:
        handle_list.append((arg.source, arg.target))

    print(len(handle_list))

    for idx, (soruce_idx, target_idx) in enumerate(handle_list):

        cprint(f"handling {soruce_idx} -> {target_idx}", "blue")

        specs = json.load(open(os.path.join(arg.data, "specs.json"), "r"))
        arch = __import__("DeepSDF_OakInk.networks." + specs["NetworkArch"], fromlist=["Decoder"])
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
        decoder = torch.nn.DataParallel(decoder)
        saved_model_state = torch.load(os.path.join(arg.data, "network", ws.model_params_subdir, arg.resume + ".pth"))
        saved_model_epoch = saved_model_state["epoch"]
        decoder.load_state_dict(saved_model_state["model_state_dict"])

        decoder = decoder.module.cuda()

        sdf_code_path = os.path.join(arg.data, ws.reconstructions_subdir, ws.reconstruction_codes_subdir)
        sdf_mesh_path = os.path.join(arg.data, ws.reconstructions_subdir, ws.reconstruction_meshes_subdir)
        source = f"{soruce_idx}.pth"
        target = f"{target_idx}.pth"
        interpolate_path = os.path.join(arg.data, "interpolate", f"{soruce_idx}-{target_idx}")
        if os.path.exists(interpolate_path) and len(os.listdir(interpolate_path)) == arg.interpolate:
            continue

        os.makedirs(interpolate_path, exist_ok=True)

        k = arg.interpolate

        with torch.no_grad():
            latent_source = torch.load(os.path.join(sdf_code_path, source))
            latent_target = torch.load(os.path.join(sdf_code_path, target))
            if not os.path.isfile(os.path.join(sdf_mesh_path, source.replace(".pth", ".ply"))):
                create_mesh(
                    decoder,
                    latent_source[0].type(torch.FloatTensor).cuda(),
                    os.path.join(sdf_mesh_path, source),
                    N=256,
                    max_batch=int(2 ** 18),
                )
            if not os.path.isfile(os.path.join(sdf_mesh_path, target.replace(".pth", ".ply"))):
                create_mesh(
                    decoder,
                    latent_target[0].type(torch.FloatTensor).cuda(),
                    os.path.join(sdf_mesh_path, target),
                    N=256,
                    max_batch=int(2 ** 18),
                )

            latent_source = latent_source.cpu().numpy()[0][0]
            latent_target = latent_target.cpu().numpy()[0][0]

        inter_list = []
        for i in range(256):
            y = np.array([latent_source[i], latent_target[i]])
            x = np.array([0, k + 1])
            xl = np.arange(1, k + 1)
            interp = np.interp(xl, x, y)
            inter_list.append(interp)
        interpolation = np.vstack(inter_list)

        with torch.no_grad():
            for i in range(k):
                interp_obj = os.path.join(interpolate_path, f"interp{i + 1:02d}")

                latent = torch.from_numpy(interpolation[:, i].T)
                create_mesh(
                    decoder,
                    latent.type(torch.FloatTensor).cuda(),
                    interp_obj,
                    N=256,
                    max_batch=int(2 ** 18),
                )
