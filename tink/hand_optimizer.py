from pprint import pprint

import numpy as np
import open3d as o3d
import torch
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayer
from manotorch.manolayer import ManoLayer
from manotorch.utils.quatutils import (
    angle_axis_to_quaternion,
    normalize_quaternion,
    quaternion_mul,
    quaternion_to_angle_axis,
)
from manotorch.utils.rodrigues import rodrigues
from termcolor import colored
from tqdm import trange

from tink.grasp_loss import FieldLoss, HandLoss, ObjectLoss
from tink.sdf_loss import sdf_loss
from tink.vis_contact_info import create_vertex_color


class GeOptimizer:
    def __init__(
        self,
        device,
        lr=1e-2,
        n_iter=2500,
        verbose=True,
        mano_root="assets/mano_v1_2",
        anchor_path="assets/anchor",
        # values to initialize coef_val
        lambda_contact_loss=20.0,
        lambda_repulsion_loss=10.0,
    ):
        self.device = device
        self.lr = lr
        self.n_iter = n_iter

        # options
        self.verbose = verbose
        self.runtime_vis = None

        # layers and loss utils
        self.mano_layer = ManoLayer(
            rot_mode="quat",
            mano_assets_root=mano_root,
            center_idx=0,
            flat_hand_mean=True,
        ).to(self.device)

        self.anchor_layer = AnchorLayer(anchor_path).to(self.device)
        self.axis_layer = AxisLayer().to(self.device)

        # opt val dict, const val dict
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {}
        self.coef_val = {
            "lambda_contact_loss": lambda_contact_loss,
            "lambda_repulsion_loss": lambda_repulsion_loss,
        }

        # creating slots for optimizer and scheduler
        self.optimizer = None
        self.optimizing = True
        self.scheduler = None

    def set_opt_val(
        self,
        # static val
        vertex_contact,  # TENSOR[NVERT, ] {0, 1}
        contact_region,  # TENSOR[NVERT, 1], int
        anchor_id,  # TENSOR[NVERT, 4]: int
        anchor_elasti,  # TENSOR[NVERT, 4]
        anchor_padding_mask,  # TENSOR[NVERT, 4] {0, 1}
        # dynamic val: hand
        hand_shape_gt=None,  # TENSOR[10, ]
        hand_tsl_gt=None,  # TENSOR[3, ]
        hand_pose_gt=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        hand_shape_init=None,  # TENSOR[10, ]
        hand_tsl_init=None,  # TENSOR[3, ]
        hand_pose_init=None,  # (LIST[NPROV, ]: int {0..16}, TENSOR[NPROV, 4])
        # dynamic val: obj
        obj_verts_3d_gt=None,
        obj_normals_gt=None,
        # sdf
        sdf_decoder=None,
        sdf_latent=None,
        sdf_center=None,
        sdf_rescale=None,
        # runtime viz
        runtime_vis=None,
    ):
        # ====== clear memory
        self.opt_val = {}
        self.const_val = {}
        self.ctrl_val = {
            "optimize_hand_shape": False,
            "optimize_hand_tsl": False,
            "optimize_hand_pose": False,
            "optimize_obj": False,
        }

        self.sdf_decoder = sdf_decoder
        self.sdf_latent = sdf_latent
        self.sdf_center = sdf_center
        self.sdf_rescale = sdf_rescale

        # ============ process static values >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        vertex_contact = vertex_contact.long()
        anchor_id = anchor_id.long()
        anchor_padding_mask = anchor_padding_mask.long()

        # boolean index contact_region, anchor_id, anchor_elasti && anchor_padding_mask
        obj_contact_region = contact_region[vertex_contact == 1]  # TENSOR[NCONT, ]
        anchor_id = anchor_id[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_elasti = anchor_elasti[vertex_contact == 1, :]  # TENSOR[NCONT, 4]
        anchor_padding_mask = anchor_padding_mask[vertex_contact == 1, :]  # TENSOR[NCONT, 4]

        # boolean mask indexing anchor_id, anchor_elasti && obj_vert_id
        indexed_anchor_id = anchor_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]
        self.const_val["indexed_anchor_id"] = indexed_anchor_id
        self.const_val["indexed_anchor_elasti"] = anchor_elasti[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        vertex_id = torch.arange(anchor_id.shape[0])[:, None].repeat_interleave(
            anchor_padding_mask.shape[1], dim=1
        )  # TENSOR[NCONT, 4]
        self.const_val["indexed_vertex_id"] = vertex_id[anchor_padding_mask == 1]  # TENSOR[NVALID, ]

        tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0]).bool().to(self.device)
        tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
        for tip_anchor_id in tip_anchor_list:
            tip_anchor_mask = tip_anchor_mask | (self.const_val["indexed_anchor_id"] == tip_anchor_id)
        self.const_val["indexed_elasti_k"] = torch.where(
            tip_anchor_mask, torch.Tensor([1.0]).to(self.device), torch.Tensor([0.1]).to(self.device)
        ).to(self.device)

        # hand faces & edges
        self.const_val["hand_faces"] = self.mano_layer.th_faces
        self.const_val["static_verts"] = self.get_static_hand_verts()
        self.const_val["hand_edges"] = HandLoss.get_edge_idx(self.const_val["hand_faces"])
        self.const_val["static_edge_len"] = HandLoss.get_edge_len(
            self.const_val["static_verts"], self.const_val["hand_edges"]
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: hand >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== hand_shape
        if hand_shape_gt is not None and hand_shape_init is not None:
            raise RuntimeError("both hand_shape gt and init are provided")
        elif hand_shape_gt is not None and hand_shape_init is None:
            self.const_val["hand_shape_gt"] = hand_shape_gt
            self.ctrl_val["optimize_hand_shape"] = False
        elif hand_shape_gt is None and hand_shape_init is not None:
            self.opt_val["hand_shape_var"] = hand_shape_init.detach().clone().requires_grad_(True)
            self.const_val["hand_shape_init"] = hand_shape_init
            self.ctrl_val["optimize_hand_shape"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_shape_var"] = torch.zeros(10, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_shape_init"] = torch.zeros(10, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_shape"] = True

        # ====== hand_tsl
        if hand_tsl_gt is not None and hand_tsl_init is not None:
            raise RuntimeError("both hand_tsl gt and init are provided")
        elif hand_tsl_gt is not None and hand_tsl_init is None:
            self.const_val["hand_tsl_gt"] = hand_tsl_gt
            self.ctrl_val["optimize_hand_tsl"] = False
        elif hand_tsl_gt is None and hand_tsl_init is not None:
            self.opt_val["hand_tsl_var"] = hand_tsl_init.detach().clone().requires_grad_(True)
            self.const_val["hand_tsl_init"] = hand_tsl_init
            self.ctrl_val["optimize_hand_tsl"] = True
        else:
            # hand_tsl_gt is None and hand_tsl_init is None:
            self.opt_val["hand_tsl_var"] = torch.zeros(3, dtype=torch.float, requires_grad=True, device=self.device)
            self.const_val["hand_tsl_init"] = torch.zeros(3, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_tsl"] = True

        # ====== hand pose
        # this is complex! need special care
        if hand_pose_gt is not None and hand_pose_init is not None:
            # full gt and init provided
            gt_pose_idx, gt_pose_val = hand_pose_gt
            init_pose_idx, init_pose_val = hand_pose_init
            if len(set(gt_pose_idx).intersection(set(init_pose_idx))) > 0:
                raise RuntimeError("repeat hand_pose gt & init provided")
            if set(gt_pose_idx).union(set(init_pose_idx)) != set(range(16)):
                raise RuntimeError("hand_pose: not enough gt & init")
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is not None and hand_pose_init is None:
            gt_pose_idx, gt_pose_val = hand_pose_gt
            self.const_val["hand_pose_gt_idx"] = gt_pose_idx
            self.const_val["hand_pose_gt_val"] = gt_pose_val
            if set(gt_pose_idx) == set(range(16)):
                # full gt provided
                self.const_val["hand_pose_var_idx"] = []
                self.opt_val["hand_pose_var_val"] = torch.zeros((0, 4), dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = False
            else:
                # partial gt provided
                var_pose_idx = self.get_var_pose_idx(gt_pose_idx)
                n_var_pose = len(var_pose_idx)
                init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_var_idx"] = var_pose_idx
                self.opt_val["hand_pose_var_val"] = torch.tensor(
                    init_val, dtype=torch.float, requires_grad=True, device=self.device
                )
                init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
                self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
                self.ctrl_val["optimize_hand_pose"] = True
        elif hand_pose_gt is None and hand_pose_init is not None:
            # full init provided
            init_pose_idx, init_pose_val = hand_pose_init
            if set(init_pose_idx) != set(range(16)):
                raise RuntimeError("hand_pose: not enough init")
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros(0, dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = init_pose_idx
            self.opt_val["hand_pose_var_val"] = init_pose_val.detach().clone().requires_grad_(True)
            self.const_val["hand_pose_init_val"] = init_pose_val
            self.ctrl_val["optimize_hand_pose"] = True
        else:
            # hand_pose_gt is None and hand_pose_init is None:
            # nothing provided
            self.const_val["hand_pose_gt_idx"] = []
            self.const_val["hand_pose_gt_val"] = torch.zeros((0, 4), dtype=torch.float).to(self.device)
            self.const_val["hand_pose_var_idx"] = list(range(16))
            n_var_pose = 16
            init_val = np.array([[0.9999, 0.0, -0.0101, 0.0]] * n_var_pose).astype(np.float32)
            self.opt_val["hand_pose_var_val"] = torch.tensor(
                init_val, dtype=torch.float, requires_grad=True, device=self.device
            )
            init_val_true = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose).astype(np.float32)
            self.const_val["hand_pose_init_val"] = torch.tensor(init_val_true, dtype=torch.float, device=self.device)
            self.ctrl_val["optimize_hand_pose"] = True
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ dynamic val: obj >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        self.const_val["obj_verts_3d_gt"] = obj_verts_3d_gt[vertex_contact == 1, :]
        self.const_val["obj_normals_gt"] = obj_normals_gt[vertex_contact == 1, :]
        self.ctrl_val["optimize_obj"] = False
        self.const_val["full_obj_verts_3d"] = obj_verts_3d_gt
        self.const_val["full_obj_normals"] = obj_normals_gt
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # ============ construct optimizer & scheduler >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # region
        # ====== optimizer
        if (
            self.ctrl_val["optimize_hand_shape"]
            or self.ctrl_val["optimize_hand_tsl"]
            or self.ctrl_val["optimize_hand_pose"]
            or self.ctrl_val["optimize_obj"]
        ):
            # dispatch lr to different param
            param = []
            if self.ctrl_val["optimize_hand_shape"]:
                param.append({"params": [self.opt_val["hand_shape_var"]]})
            if self.ctrl_val["optimize_hand_tsl"]:
                param.append({"params": [self.opt_val["hand_tsl_var"]], "lr": self.lr})
            if self.ctrl_val["optimize_hand_pose"]:
                param.append({"params": [self.opt_val["hand_pose_var_val"]]})
            if self.ctrl_val["optimize_obj"]:
                param.append({"params": [self.opt_val["obj_rot_var"]]})
                param.append({"params": [self.opt_val["obj_tsl_var"]], "lr": self.lr})

            self.optimizer = torch.optim.Adam(param, lr=self.lr)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=1e-5, mode="min", factor=0.5, patience=20, verbose=False
            )
        else:
            self.optimizing = False
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ====== runtime viz
        self.runtime_vis = runtime_vis

        # ====== verbose
        if self.verbose:
            print("Optimizing: ", self.optimizing)
            pprint(self.ctrl_val)
            pprint(list(self.opt_val.keys()))
            pprint(list(self.const_val.keys()))
            pprint(self.coef_val)

    @staticmethod
    def get_var_pose_idx(sel_pose_idx):
        # gt has 16 pose
        all_pose_idx = set(range(16))
        sel_pose_idx_set = set(sel_pose_idx)
        var_pose_idx = all_pose_idx.difference(sel_pose_idx_set)
        return list(var_pose_idx)

    def get_static_hand_verts(self):
        init_val_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)
        vec_pose = torch.tensor(init_val_pose).reshape(-1).unsqueeze(0).float().to(self.device)
        vec_shape = torch.zeros(1, 10).float().to(self.device)
        mano_out = self.mano_layer(vec_pose, vec_shape)
        v = mano_out.verts.squeeze(0)
        return v

    @staticmethod
    def assemble_pose_vec(gt_idx, gt_pose, var_idx, var_pose):
        idx_tensor = torch.cat((torch.Tensor(gt_idx).long(), torch.Tensor(var_idx).long()))
        pose_tensor = torch.cat((gt_pose, var_pose), dim=0)
        pose_tensor = pose_tensor[torch.argsort(idx_tensor)]
        return pose_tensor

    @staticmethod
    def transf_vectors(vectors, tsl, rot):
        """
        vectors: [K, 3], tsl: [3, ], rot: [3, ]
        return: [K, 3]
        """
        rot_matrix = rodrigues(rot.unsqueeze(0)).squeeze(0).reshape((3, 3))
        vec = (rot_matrix @ vectors.T).T
        vec = vec + tsl
        return vec

    def loss_fn(self, opt_val, const_val, ctrl_val, coef_val):
        var_hand_pose_assembled = self.assemble_pose_vec(
            const_val["hand_pose_gt_idx"],
            const_val["hand_pose_gt_val"],
            const_val["hand_pose_var_idx"],
            opt_val["hand_pose_var_val"],
        )

        # dispatch hand var
        vec_pose = var_hand_pose_assembled.unsqueeze(0)
        if ctrl_val["optimize_hand_shape"]:
            vec_shape = opt_val["hand_shape_var"].unsqueeze(0)
        else:
            vec_shape = const_val["hand_shape_gt"].unsqueeze(0)
        if ctrl_val["optimize_hand_tsl"]:
            vec_tsl = opt_val["hand_tsl_var"].unsqueeze(0)
        else:
            vec_tsl = const_val["hand_tsl_gt"].unsqueeze(0)

        # rebuild hand

        mano_out = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts, rebuild_joints, rebuild_transf = mano_out.verts, mano_out.joints, mano_out.transforms_abs

        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat([torch.zeros(3, 3).to(self.device), vec_tsl.view(3, -1)], dim=1),
                torch.zeros(1, 4).to(self.device),
            ],
            dim=0,
        )
        rebuild_verts_squeezed = rebuild_verts.squeeze(0)

        # rebuild anchor
        rebuild_anchor = self.anchor_layer(rebuild_verts)
        rebuild_anchor = rebuild_anchor.contiguous()  # TENSOR[1, 32, 3]
        rebuild_anchor = rebuild_anchor.squeeze(0)  # TENSOR[32, 3]
        anchor_pos = rebuild_anchor[const_val["indexed_anchor_id"]]  # TENSOR[NVALID, 3]

        # dispatch obj var
        if ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                const_val["obj_verts_3d_can"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_verts = self.transf_vectors(
                const_val["full_obj_verts_3d"],
                opt_val["obj_tsl_var"],
                opt_val["obj_rot_var"],
            )
            full_obj_normals = self.transf_vectors(
                const_val["full_obj_normals"],
                torch.zeros(3, dtype=torch.float, device=self.device),
                opt_val["obj_rot_var"],
            )
        else:
            obj_verts = const_val["obj_verts_3d_gt"]
            full_obj_verts = const_val["full_obj_verts_3d"]
            full_obj_normals = const_val["full_obj_normals"]

        # contact loss
        contact_loss = FieldLoss.contact_loss(
            anchor_pos,
            obj_verts[const_val["indexed_vertex_id"]],
            const_val["indexed_anchor_elasti"],
            const_val["indexed_elasti_k"],
        )

        sdf_model_loss = sdf_loss(
            self.sdf_decoder, self.sdf_latent, rebuild_verts_squeezed, self.sdf_center, self.sdf_rescale
        )

        if ctrl_val["optimize_hand_pose"]:
            # get hand loss
            quat_norm_loss = HandLoss.pose_quat_norm_loss(var_hand_pose_assembled)
            var_hand_pose_normalized = normalize_quaternion(var_hand_pose_assembled)
            pose_reg_loss = HandLoss.pose_reg_loss(
                var_hand_pose_normalized[const_val["hand_pose_var_idx"]], const_val["hand_pose_init_val"]
            )

            b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)

            angle_axis = quaternion_to_angle_axis(var_hand_pose_normalized.reshape((16, 4)))
            angle_axis = angle_axis[1:, :]  # ignore global rot [15, 3]
            axis = angle_axis / torch.norm(angle_axis, dim=1, keepdim=True)
            angle = torch.norm(angle_axis, dim=1, keepdim=False)
            # limit angle
            angle_limit_loss = HandLoss.rotation_angle_loss(angle)

            joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis)
            joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis)
            joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis)

            edge_loss = HandLoss.edge_len_loss(
                rebuild_verts_squeezed, const_val["hand_edges"], const_val["static_edge_len"]
            )

        else:
            quat_norm_loss = torch.Tensor([0.0]).to(self.device)
            pose_reg_loss = torch.Tensor([0.0]).to(self.device)
            angle_limit_loss = torch.Tensor([0.0]).to(self.device)
            joint_b_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_u_axis_loss = torch.Tensor([0.0]).to(self.device)
            joint_l_limit_loss = torch.Tensor([0.0]).to(self.device)
            edge_loss = torch.Tensor([0.0]).to(self.device)
            # pose_reg_loss_to_zero = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_shape"]:
            shape_reg_loss = HandLoss.shape_reg_loss(opt_val["hand_shape_var"], const_val["hand_shape_init"])
        else:
            shape_reg_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_hand_tsl"]:
            hand_tsl_loss = HandLoss.hand_tsl_loss(opt_val["hand_tsl_var"], const_val["hand_tsl_init"])
        else:
            hand_tsl_loss = torch.Tensor([0.0]).to(self.device)

        if ctrl_val["optimize_obj"]:
            obj_transf_loss = ObjectLoss.obj_transf_loss(
                opt_val["obj_tsl_var"], opt_val["obj_rot_var"], const_val["obj_tsl_init"], const_val["obj_rot_init"]
            )
        else:
            obj_transf_loss = torch.Tensor([0.0]).to(self.device)

        loss = (
            # ============= HAND ANATOMICAL LOSS
            1.0 * quat_norm_loss
            + 1.0 * angle_limit_loss
            + 1.0 * edge_loss
            + 0.1 * joint_b_axis_loss
            + 0.1 * joint_u_axis_loss
            + 0.1 * joint_l_limit_loss
            # ============= ELAST POTENTIONAL ENERGY
            + coef_val["lambda_contact_loss"] * contact_loss
            + coef_val["lambda_repulsion_loss"] * sdf_model_loss
            # ============= OFFSET LOSS
            + 1.0 * pose_reg_loss
            + 1.0 * shape_reg_loss
            + 1.0 * hand_tsl_loss
            + 1.0 * obj_transf_loss
        )

        # debug: runtime viz
        if self.runtime_vis:
            if self.ctrl_val["optimize_obj"]:
                full_obj_verts = self.transf_vectors(
                    self.const_val["full_obj_verts_3d"],
                    self.opt_val["obj_tsl_var"].detach(),
                    self.opt_val["obj_rot_var"].detach(),
                )
            else:
                full_obj_verts = self.const_val["full_obj_verts_3d"]

            if not ctrl_val["optimize_hand_pose"]:
                b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transf)  # mend this up
            self.runtime_show(rebuild_verts, b_axis, u_axis, l_axis, rebuild_transf, full_obj_verts)

        return (
            loss,
            {
                "quat_norm_loss": quat_norm_loss.detach().cpu().item(),
                "angle_limit_loss": angle_limit_loss.detach().cpu().item(),
                "edge_loss": edge_loss.detach().cpu().item(),
                "joint_b_axis_loss": joint_b_axis_loss.detach().cpu().item(),
                "joint_u_axis_loss": joint_u_axis_loss.detach().cpu().item(),
                "joint_l_limit_loss": joint_l_limit_loss.detach().cpu().item(),
                "contact_loss": contact_loss.detach().cpu().item(),
                # "repulsion_loss": repulsion_loss.detach().cpu().item(),
                "pose_reg_loss": pose_reg_loss.detach().cpu().item(),
                "hand_tsl_loss": hand_tsl_loss.detach().cpu().item(),
                "obj_transf_loss": obj_transf_loss.detach().cpu().item(),
            },
        )

    def optimize(self, progress=False):
        if progress:
            bar = trange(self.n_iter, position=3)
            bar_hand = trange(0, position=2, bar_format="{desc}")
            bar_contact = trange(0, position=1, bar_format="{desc}")
            bar_axis = trange(0, position=0, bar_format="{desc}")
        else:
            bar = range(self.n_iter)

        loss = torch.Tensor([1000.0]).to(self.device)
        loss_dict = {}
        for _ in bar:
            if self.optimizing:
                self.optimizer.zero_grad()

            loss, loss_dict = self.loss_fn(self.opt_val, self.const_val, self.ctrl_val, self.coef_val)

            if self.optimizing:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)

            if progress:
                bar.set_description("TOTAL LOSS {:4e}".format(loss.item()))
                try:
                    bar_hand.set_description(
                        colored("HAND_REGUL_LOSS: ", "yellow")
                        + "QN={:.3e} PR={:.3e} EG={:.3e}".format(
                            loss_dict["quat_norm_loss"],  # QN
                            loss_dict["pose_reg_loss"],  # PR
                            loss_dict["edge_loss"],  # Edge
                        )
                    )
                except:
                    pass
                try:
                    bar_contact.set_description(
                        colored("HO_CONTACT_LOSS: ", "blue")
                        + "Conta={:.3e}, Repul={:.3e}, OT={:.3e}".format(
                            loss_dict["contact_loss"],  # Conta
                            loss_dict["repulsion_loss"],  # Repul
                            loss_dict["obj_transf_loss"],  # OT
                        )
                    )
                except:
                    pass
                try:
                    bar_axis.set_description(
                        colored("ANGLE_LOSS: ", "cyan")
                        + "AL={:.3e} JB={:.3e} JU={:.3e} JL={:.3e}".format(
                            loss_dict["angle_limit_loss"],  # AL
                            loss_dict["joint_b_axis_loss"],  # JB
                            loss_dict["joint_u_axis_loss"],  # JU
                            loss_dict["joint_l_limit_loss"],  # JL
                        )
                    )
                except:
                    pass
        return loss.item(), loss_dict

    def recover_hand(self, squeeze_out=True):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        vec_pose = vars_hand_pose_normalized.unsqueeze(0)
        if self.ctrl_val["optimize_hand_shape"]:
            vec_shape = self.opt_val["hand_shape_var"].detach().unsqueeze(0)
        else:
            vec_shape = self.const_val["hand_shape_gt"].unsqueeze(0)
        if self.ctrl_val["optimize_hand_tsl"]:
            vec_tsl = self.opt_val["hand_tsl_var"].detach().unsqueeze(0)
        else:
            vec_tsl = self.const_val["hand_tsl_gt"].unsqueeze(0)

        device = vec_pose.device
        mano_out = self.mano_layer(vec_pose, vec_shape)
        rebuild_verts, rebuild_joints, rebuild_transf = mano_out.verts, mano_out.joints, mano_out.transforms_abs

        rebuild_verts = rebuild_verts + vec_tsl
        rebuild_joints = rebuild_joints + vec_tsl
        rebuild_transf = rebuild_transf + torch.cat(
            [
                torch.cat((torch.zeros((3, 3), device=device), vec_tsl.T), dim=1),
                torch.zeros((1, 4), device=device),
            ],
            dim=0,
        )
        if squeeze_out:
            rebuild_verts, rebuild_joints, rebuild_transf = (
                rebuild_verts.squeeze(0),
                rebuild_joints.squeeze(0),
                rebuild_transf.squeeze(0),
            )
        return rebuild_verts, rebuild_joints, rebuild_transf

    def recover_hand_pose(self):
        vars_hand_pose_assembled = self.assemble_pose_vec(
            self.const_val["hand_pose_gt_idx"],
            self.const_val["hand_pose_gt_val"],
            self.const_val["hand_pose_var_idx"],
            self.opt_val["hand_pose_var_val"],
        ).detach()
        vars_hand_pose_normalized = normalize_quaternion(vars_hand_pose_assembled)
        return vars_hand_pose_normalized

    def recover_hand_param(self):
        hand_pose = self.recover_hand_pose()
        if self.ctrl_val["optimize_hand_shape"]:
            hand_shape = self.opt_val["hand_shape_var"].detach()
        else:
            hand_shape = self.const_val["hand_shape_gt"]
        if self.ctrl_val["optimize_hand_tsl"]:
            hand_tsl = self.opt_val["hand_tsl_var"].detach()
        else:
            hand_tsl = self.const_val["hand_tsl_gt"]
        return hand_pose, hand_shape, hand_tsl

    def recover_obj(self):
        if self.ctrl_val["optimize_obj"]:
            obj_verts = self.transf_vectors(
                self.const_val["full_obj_verts_3d"],
                self.opt_val["obj_tsl_var"].detach(),
                self.opt_val["obj_rot_var"].detach(),
            )
        else:
            obj_verts = self.const_val["full_obj_verts_3d"]
        return obj_verts

    def obj_rot_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_rot_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_rot")

    def obj_tsl_np(self):
        if self.ctrl_val["optimize_obj"]:
            res = self.opt_val["obj_tsl_var"].detach().cpu().numpy()
            return res
        else:
            raise RuntimeError("not optimizing obj, cannot get obj_tsl")

    def runtime_show(self, hand_verts, b_axis, u_axis, l_axis, hand_transf, obj_verts):

        hand_transf = hand_transf.detach().cpu().squeeze(0).numpy()


        while True:
            self.runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(
                np.array(hand_verts.detach().cpu().squeeze(0))
            )
            self.runtime_vis["hand_mesh"].compute_vertex_normals()
            self.runtime_vis["obj_mesh"].points = o3d.utility.Vector3dVector(
                np.array(obj_verts.detach().cpu().squeeze(0))
            )
            # self.runtime_vis["obj_mesh"].compute_vertex_normals()
            self.runtime_vis["window"].update_geometry(self.runtime_vis["hand_mesh"])
            self.runtime_vis["window"].update_geometry(self.runtime_vis["obj_mesh"])
            self.runtime_vis["window"].update_renderer()
            if not self.runtime_vis["window"].poll_events():
                break

        return


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [
            [0, -z_unit_Arr[2], z_unit_Arr[1]],
            [z_unit_Arr[2], 0, -z_unit_Arr[0]],
            [-z_unit_Arr[1], z_unit_Arr[0], 0],
        ]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array(
        [
            [0, -z_c_vec[2], z_c_vec[1]],
            [z_c_vec[2], 0, -z_c_vec[0]],
            [-z_c_vec[1], z_c_vec[0], 0],
        ]
    )

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def init_runtime_viz(
    hand_verts,
    hand_faces,
    obj_verts,
    obj_normals,
    contact_info,
    cam_extr=None,
):
    hand_mesh_cur = o3d.geometry.TriangleMesh()
    hand_mesh_cur.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_cur.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh_cur.compute_vertex_normals()

    o3d_obj_pc = o3d.geometry.PointCloud()
    o3d_obj_pc.points = o3d.utility.Vector3dVector(obj_verts)
    o3d_obj_pc.normals = o3d.utility.Vector3dVector(obj_normals)
    if contact_info is not None:
        o3d_obj_pc.colors = o3d.utility.Vector3dVector(create_vertex_color(contact_info, mode="contact_region"))

    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
    vis_cur.add_geometry(o3d_obj_pc)
    vis_cur.add_geometry(hand_mesh_cur)

    vis_cur.poll_events()
    runtime_vis = {
        "hand_mesh": hand_mesh_cur,
        "obj_mesh": o3d_obj_pc,
        "window": vis_cur,
    }
    if cam_extr is not None:
        ctl = runtime_vis["window"].get_view_control()
        parameters = ctl.convert_to_pinhole_camera_parameters()
        parameters.extrinsic = cam_extr
        ctl.convert_from_pinhole_camera_parameters(parameters)


    return runtime_vis


def update_runtime_viz(runtime_vis, hand_verts_curr):
    runtime_vis["hand_mesh"].vertices = o3d.utility.Vector3dVector(hand_verts_curr)
    runtime_vis["hand_mesh"].compute_vertex_normals()

    runtime_vis["window"].update_geometry(runtime_vis["hand_mesh"])

    runtime_vis["window"].poll_events()
    runtime_vis["window"].update_renderer()
    runtime_vis["window"].reset_view_point(True)
