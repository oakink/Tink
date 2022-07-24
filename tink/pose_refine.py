# ----------------------------------------------
# Written by Kailin Li (kailinli@sjtu.edu.cn)
# ----------------------------------------------
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from torch.nn import parameter
import trimesh
from manotorch.manolayer import ManoLayer
from manotorch.utils.anchorutils import anchor_load, get_region_palm_mask, masking_load_driver
from manotorch.utils.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis
from termcolor import colored, cprint
from tqdm import trange

from tink.cal_contact_info import to_pointcloud
from tink.hand_optimizer import GeOptimizer, init_runtime_viz
from tink.info_transform import get_obj_path
from tink.sdf_loss import load_obj_latent, load_sdf_decoder


def scatter_array(ax, array, c=None, cmap=None):
    ax.scatter3D(array[:, 0], array[:, 1], array[:, 2], c=c, cmap=cmap)
    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval Hand Manipulation")

    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--contact_path", "-p", type=str, required=True)
    parser.add_argument("--source", "-s", type=str)
    parser.add_argument("--target", "-t", type=str)
    parser.add_argument("--resume", type=str, default="latest")

    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--iters", default=1500, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--center_obj", action="store_false")

    parser.add_argument("--fix_tsl", action="store_true")

    parser.add_argument("--vis", action="store_true")

    arg = parser.parse_args()

    interpolate_path = os.path.join(arg.data, "interpolate", f"{arg.source}-{arg.target}")
    contact_info = pickle.load(open(os.path.join(arg.contact_path, arg.target, "contact_info.pkl"), "rb"))
    trans_contact_path = os.path.join(arg.contact_path, arg.target)

    # if not arg.overwrite and os.path.exists(os.path.join(trans_contact_path, "hand_param.pkl")):
    #     cprint(f"{os.path.join(trans_contact_path, 'hand_param.pkl')} exists, skip.", "yellow")
    #     exit(0)

    try:
        mesh = trimesh.load(get_obj_path(arg.target), process=False, force="mesh", skip_materials=True)

        if arg.center_obj:
            bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
            mesh.vertices = mesh.vertices - bbox_center

        obj_pc = to_pointcloud(mesh)
    except:
        pc_data = pickle.load(open(os.path.join(arg.data, "virtual_pc_cache", f"{arg.target}.pkl"), "rb"))
        obj_pc = o3d.geometry.PointCloud()
        obj_pc.points = o3d.utility.Vector3dVector(pc_data["points"])
        obj_pc.normals = o3d.utility.Vector3dVector(pc_data["normals"])

    rescale = pickle.load(open(os.path.join(arg.data, "rescale.pkl"), "rb"))
    rescale = rescale["max_norm"] * rescale["scale"]

    if arg.center_obj:
        # * if the object is centered, we do not need to center it again.
        center = np.zeros(3, dtype=np.float32)
    else:
        center = pickle.load(
            open(list(glob.iglob(f"{os.path.join(arg.data, 'SdfSamples_resize', arg.target)}/*.pkl"))[0], "rb")
        )

    device = torch.device("cuda")
    obj_verts_np, obj_normals_np = np.asarray(obj_pc.points), np.asarray(obj_pc.normals)
    obj_verts, obj_normals = torch.tensor(obj_verts_np).to(device), torch.tensor(obj_normals_np).to(device)

    hand_param = pickle.load(open(os.path.join(arg.contact_path, "hand_param.pkl"), "rb"))

    if arg.fix_tsl:
        contact_region_center = np.asfarray(obj_pc.points, dtype=np.float32)[contact_info["vertex_contact"] == 1].mean(0)
        raw_mesh = trimesh.load(get_obj_path(arg.source), process=False, force="mesh", skip_materials=True)
        raw_obj_pc = to_pointcloud(raw_mesh)
        raw_contact_info = pickle.load(open(os.path.join(arg.contact_path, "contact_info.pkl"), "rb"))

        raw_contact_region_center = np.asfarray(raw_obj_pc.points, dtype=np.float32)[
            raw_contact_info["vertex_contact"] == 1
        ].mean(0)

        hand_param["tsl"] = hand_param["tsl"] + (contact_region_center - raw_contact_region_center)

    hand_pose = angle_axis_to_quaternion(torch.tensor(hand_param["pose"].reshape(-1, 3))).to(device)
    hand_shape = torch.tensor(hand_param["shape"]).to(device)
    hand_tsl = torch.tensor(hand_param["tsl"]).to(device)

    vertex_contact = torch.from_numpy(contact_info["vertex_contact"]).long().to(device)
    contact_region = torch.from_numpy(contact_info["hand_region"]).long().to(device)
    anchor_id = torch.from_numpy(contact_info["anchor_id"]).long().to(device)
    anchor_elasti = torch.from_numpy(contact_info["anchor_elasti"]).float().to(device)
    anchor_padding_mask = torch.from_numpy(contact_info["anchor_padding_mask"]).long().to(device)

    hra, hpv = masking_load_driver("./assets/anchor", "./assets/hand_palm_full.txt")

    hand_region_assignment = torch.from_numpy(hra).long().to(device)
    hand_palm_vertex_mask = torch.from_numpy(hpv).long().to(device)

    opt = GeOptimizer(
        device=device,
        lr=arg.lr,
        n_iter=arg.iters,
        verbose=False,
        lambda_contact_loss=500.0,
        lambda_repulsion_loss=200.0,
    )
    mano_out = opt.mano_layer(hand_pose[None], hand_shape[None])

    if arg.vis:
        runtime_viz = init_runtime_viz(
            hand_verts=mano_out.verts.cpu().squeeze().numpy(),
            hand_faces=opt.mano_layer.th_faces.cpu().numpy(),
            obj_verts=obj_verts_np,
            obj_normals=obj_normals_np,
            contact_info=contact_info,
        )
    else:
        runtime_viz = None
    opt.set_opt_val(
        vertex_contact=vertex_contact,
        contact_region=contact_region,
        anchor_id=anchor_id,
        anchor_elasti=anchor_elasti,
        anchor_padding_mask=anchor_padding_mask,
        hand_shape_init=hand_shape,
        hand_tsl_init=hand_tsl,
        hand_pose_init=([i for i in range(16)], hand_pose),
        obj_verts_3d_gt=obj_verts,
        obj_normals_gt=obj_normals,
        sdf_decoder=load_sdf_decoder(arg.data, arg.resume),
        sdf_latent=load_obj_latent(arg.data, arg.target),
        sdf_center=torch.tensor(center).to(device),
        sdf_rescale=torch.tensor(rescale).to(device),
        runtime_vis=runtime_viz,
    )

    opt.optimize(progress=True)

    hand_pose, hand_shape, hand_tsl = opt.recover_hand_param()
    hand_pose, hand_shape, hand_tsl = hand_pose.cpu(), hand_shape.cpu().numpy(), hand_tsl.cpu().numpy()
    hand_pose = quaternion_to_angle_axis(hand_pose).reshape(48).numpy()

    pickle.dump(
        {"pose": hand_pose, "shape": hand_shape, "tsl": hand_tsl},
        open(os.path.join(trans_contact_path, "hand_param.pkl"), "wb"),
    )
    print("finish")
