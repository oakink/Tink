import os
import argparse
import os
import pickle
import re

import numpy as np
import open3d as o3d
import torch
import trimesh
from manotorch.manolayer import ManoLayer, MANOOutput
from termcolor import cprint
from tink.cal_contact_info import load_pointcloud, to_pointcloud
from tink.info_transform import get_obj_path
from tink.vis_contact_info import open3d_show, create_vertex_color
from tink.info_transform import cal_closest_idx


def get_hand_parameter(path):
    pose = pickle.load(open(path, "rb"))
    return pose["pose"], pose["shape"], pose["tsl"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="selector")
    parser.add_argument("--path", "-p", type=str)
    args = parser.parse_args()

    mano_layer = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2")
    hand_faces = mano_layer.th_faces.numpy()

    matcher = re.compile(r"DeepSDF/data/sdf/(.+)/contact/(.{6})/.+/(.{6})/hand_param.pkl")
    match_res = matcher.findall(args.path)
    cat, soid, toid = match_res[0][0], match_res[0][1], match_res[0][2]
    interpolate_path = os.path.join("DeepSDF/data/sdf", cat, "interpolate", f"{soid}-{toid}")
    interpolate_objs = []
    for i in range(10):
        interpolate_obj_path = os.path.join(interpolate_path, f"interp{i + 1:02d}.ply")
        interpolate_objs.append(trimesh.load(interpolate_obj_path, process=False))
    interpolate_pc_path = os.path.join("DeepSDF/data/sdf", cat, "interpolate_pc_cache", f"{soid}-{toid}")
    interpolate_pcs = []
    for i in range(10):
        interpolate_pc = os.path.join(interpolate_pc_path, f"interp{i + 1:02d}_pc5000.pkl")
        interpolate_pcs.append(pickle.load(open(interpolate_pc, "rb")))
    interpolate_cr_path = os.path.split(args.path)[0]
    interpolate_contacts = []
    for i in range(10):
        interpolate_contact_path = os.path.join(interpolate_cr_path, f"contact_interp{i + 1:02d}.pkl")
        interpolate_contacts.append(pickle.load(open(interpolate_contact_path, "rb")))

    rescale = pickle.load(open(os.path.join("DeepSDF/data/sdf", cat, "rescale.pkl"), "rb"))
    rescale = rescale["max_norm"] * rescale["scale"]

    source_mesh = trimesh.load(
        get_obj_path(soid, use_downsample=False), process=False, force="mesh", skip_materials=True
    )
    target_mesh = trimesh.load(
        get_obj_path(toid, use_downsample=False), process=False, force="mesh", skip_materials=True
    )
    source_mesh.vertices = (
        source_mesh.vertices - (np.array(source_mesh.vertices).max(0) + np.array(source_mesh.vertices).min(0)) / 2
    )  # center
    source_mesh.vertices = source_mesh.vertices / rescale
    target_mesh.vertices = (
        target_mesh.vertices - (np.array(target_mesh.vertices).max(0) + np.array(target_mesh.vertices).min(0)) / 2
    )  # center
    target_mesh.vertices = target_mesh.vertices / rescale

    # def upsample(m):
    #     for _ in range(3):
    #         faces = m.faces
    #         obj_verts = m.vertices
    #         obj_verts, faces = trimesh.remesh.subdivide(obj_verts, faces)
    #         tmp_mesh = o3d.geometry.TriangleMesh()
    #         tmp_mesh.triangles = o3d.utility.Vector3iVector(faces)
    #         tmp_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
    #         tmp_mesh = tmp_mesh.simplify_quadric_decimation(50000)
    #         m.vertices = np.array(tmp_mesh.vertices)
    #         m.faces = np.array(tmp_mesh.triangles)
    #         print(m.vertices.shape)
    #     return m

    interpolate_objs = [source_mesh] + interpolate_objs + [target_mesh]

    source_mesh = trimesh.load(get_obj_path(soid), process=False, force="mesh", skip_materials=True)
    target_mesh = trimesh.load(get_obj_path(toid), process=False, force="mesh", skip_materials=True)
    source_mesh.vertices = (
        source_mesh.vertices - (np.array(source_mesh.vertices).max(0) + np.array(source_mesh.vertices).min(0)) / 2
    )  # center
    source_mesh.vertices = source_mesh.vertices / rescale
    target_mesh.vertices = (
        target_mesh.vertices - (np.array(target_mesh.vertices).max(0) + np.array(target_mesh.vertices).min(0)) / 2
    )  # center
    target_mesh.vertices = target_mesh.vertices / rescale

    source_pc = to_pointcloud(source_mesh)
    target_pc = to_pointcloud(target_mesh)
    interpolate_pcs = (
        [{"points": np.asarray(source_pc.points), "normals": np.asarray(source_pc.normals)}]
        + interpolate_pcs
        + [{"points": np.asarray(target_pc.points), "normals": np.asarray(target_pc.normals)}]
    )
    source_contact = pickle.load(open(os.path.join(*args.path.split(os.sep)[:7], "contact_info.pkl"), "rb"))
    source_contact["hand_region"] = np.ones_like(source_contact["hand_region"]) * 17
    target_contact = pickle.load(open(os.path.join(*args.path.split(os.sep)[:8], "contact_info.pkl"), "rb"))
    target_contact["hand_region"] = np.ones_like(target_contact["hand_region"]) * 17
    interpolate_contacts = [source_contact] + interpolate_contacts + [target_contact]

    source_hand = os.path.join(*args.path.split(os.sep)[:7], "hand_param.pkl")
    hand_pose, hand_shape, hand_tsl = get_hand_parameter(source_hand)
    mano_output: MANOOutput = mano_layer(
        torch.from_numpy(hand_pose).unsqueeze(0), torch.from_numpy(hand_shape).unsqueeze(0)
    )
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.vertices = o3d.utility.Vector3dVector(
        (
            (mano_output.verts.squeeze().numpy() + hand_tsl[None])
            - (np.array(source_mesh.vertices).max(0) + np.array(source_mesh.vertices).min(0)) / 2
        )
        / rescale
    )
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[102.0 / 255.0, 209.0 / 255.0, 243.0 / 255.0]] * mano_output.verts.shape[1])
    )
    hand_mesh.compute_triangle_normals()
    hand_mesh.compute_vertex_normals()

    target_hand = os.path.join(*args.path.split(os.sep)[:8], "hand_param.pkl")
    hand_pose_tar, hand_shape_tar, hand_tsl_tar = get_hand_parameter(target_hand)
    mano_output_tar: MANOOutput = mano_layer(
        torch.from_numpy(hand_pose_tar).unsqueeze(0), torch.from_numpy(hand_shape_tar).unsqueeze(0)
    )
    hand_mesh_tar = o3d.geometry.TriangleMesh()
    hand_mesh_tar.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_tar.vertices = o3d.utility.Vector3dVector(
        (
            (mano_output_tar.verts.squeeze().numpy() + hand_tsl_tar[None])
            - (np.array(target_mesh.vertices).max(0) + np.array(target_mesh.vertices).min(0)) / 2
        )
        / rescale
    )
    hand_mesh_tar.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[102.0 / 255.0, 209.0 / 255.0, 243.0 / 255.0]] * mano_output_tar.verts.shape[1])
    )
    hand_mesh_tar.compute_triangle_normals()
    hand_mesh_tar.compute_vertex_normals()

    def remove_g(vis):
        vis.remove_geometry(hand_mesh_tar, reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    def add_g(vis):
        vis.add_geometry(hand_mesh_tar, reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    def remove_a(vis):
        vis.remove_geometry(hand_mesh, reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    def add_a(vis):
        vis.add_geometry(hand_mesh, reset_bounding_box=False)
        vis.update_renderer()
        vis.poll_events()

    idx = -1
    obj_mesh = None
    # obj_pc = None
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Runtime HAND + OBJ",
        width=1024,
        height=1024,
    )
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
    vis.add_geometry(axes)

    mano_layer = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2")
    hand_faces = mano_layer.th_faces.numpy()

    def vis_mesh(idx):
        global obj_mesh
        global hand_mesh
        # global obj_pc

        obj_trimesh = interpolate_objs[idx]
        if obj_mesh is None:
            obj_mesh = o3d.geometry.TriangleMesh()
            vis.add_geometry(obj_mesh)
            vis.remove_geometry(axes, reset_bounding_box=False)
            # obj_pc = o3d.geometry.PointCloud()
            # vis.add_geometry(obj_pc)
            vis.update_renderer()

        target_idx = cal_closest_idx(interpolate_pcs[idx]["points"], np.asarray(obj_trimesh.vertices))
        contact_info = {
            "vertex_contact": interpolate_contacts[idx]["vertex_contact"][target_idx],
            "hand_region": interpolate_contacts[idx]["hand_region"][target_idx],
        }

        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_trimesh.faces)
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_trimesh.vertices)
        obj_mesh.vertex_colors = o3d.utility.Vector3dVector(create_vertex_color(contact_info, mode="contact_region"))
        obj_mesh.compute_triangle_normals()
        obj_mesh.compute_vertex_normals()

        # obj_pc.points = o3d.utility.Vector3dVector(interpolate_pcs[idx]["points"][mask])
        # obj_pc.normals = o3d.utility.Vector3dVector(interpolate_pcs[idx]["normals"][mask])
        # obj_pc.colors = o3d.utility.Vector3dVector(
        #     create_vertex_color(interpolate_contacts[idx], mode="contact_region")[mask]
        # )
        # vis.update_geometry(obj_pc)
        vis.update_geometry(obj_mesh)
        vis.update_renderer()

        # hand_pose, hand_shape, hand_tsl = get_hand_parameter(hand_path)
        # mano_output: MANOOutput = mano_layer(
        #     torch.from_numpy(hand_pose).unsqueeze(0), torch.from_numpy(hand_shape).unsqueeze(0)
        # )
        # if hand_mesh is None:
        #     hand_mesh = o3d.geometry.TriangleMesh()
        #     vis.add_geometry(hand_mesh)
        #     vis.update_renderer()
        # hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
        # hand_mesh.vertices = o3d.utility.Vector3dVector(mano_output.verts.squeeze().numpy() + hand_tsl[None])
        # hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        #     np.array([[102.0 / 255.0, 209.0 / 255.0, 243.0 / 255.0]] * mano_output.verts.shape[1])
        # )
        # hand_mesh.compute_triangle_normals()
        # hand_mesh.compute_vertex_normals()
        # vis.update_geometry(hand_mesh)
        # vis.update_renderer()

    def get_next(vis):
        global obj_mesh
        global hand_mesh
        global idx

        # if oid is not None and hand_path is not None:
        #     cprint(f"finish {oid} {hand_path}", "blue")
        try:
            idx = idx + 1
            if idx >= 10 + 2:
                idx = 0
            # cur_oid, hand_path = next(handle_list)
        except:
            print("Finish!")
            exit(0)
        vis_mesh(idx)

    def set_view(vis):
        ctl = vis.get_view_control()
        import random

        t = random.randint(0, 2)
        v = np.zeros(3)
        v[t] = 1
        ctl.set_lookat(v[None].T)

    vis.register_key_callback(ord("N"), get_next)

    vis.register_key_callback(ord("1"), remove_g)
    vis.register_key_callback(ord("2"), add_g)
    vis.register_key_callback(ord("3"), remove_a)
    vis.register_key_callback(ord("4"), add_a)

    vis.register_key_callback(ord("T"), set_view)
    while True:
        vis.update_renderer()
        vis.poll_events()
