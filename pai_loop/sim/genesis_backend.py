"""Genesis implementation of QuadrupedSim for Unitree Go2."""

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat

from .interface import SimState

FEET_CALF_LINKS = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
# The foot is a fixed child of the calf that Genesis merges into the calf link (see go2.urdf *_foot_joint).
FOOT_OFFSET_IN_CALF = (0.0, 0.0, -0.213)
# Points drawn over the recorded video: the trunk, then per leg the hip, thigh and calf (knee) joints and the foot.
MARKER_LEGS = ["FR", "FL", "RR", "RL"]
MARKER_NAMES = ["base"] + [f"{leg}_{p}" for leg in MARKER_LEGS for p in ("hip", "thigh", "calf", "foot")]


class GenesisGo2Sim:
    def __init__(
        self,
        num_envs: int,
        joint_names: list[str],
        kp: float,
        kd: float,
        init_pos,
        dt: float = 0.02,
        substeps: int = 2,
        camera: dict | None = None,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = gs.device

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False, max_collision_pairs=30, tolerance=1e-5
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=init_pos))
        self.cam = None
        if camera is not None:
            self.cam = self.scene.add_camera(
                res=tuple(camera["res"]), pos=camera["pos"], lookat=camera["lookat"], fov=camera["fov"], GUI=False
            )
        self.scene.build(n_envs=num_envs)

        self.dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in joint_names], dtype=gs.tc_int, device=gs.device
        )
        self.robot.set_dofs_kp([kp] * len(joint_names), self.dof_idx)
        self.robot.set_dofs_kv([kd] * len(joint_names), self.dof_idx)
        self.feet_links = [self.robot.get_link(n).idx_local for n in FEET_CALF_LINKS]
        self.foot_offset = torch.tensor(FOOT_OFFSET_IN_CALF, dtype=gs.tc_float, device=gs.device)

    def step(self, joint_targets):
        self.robot.control_dofs_position(joint_targets, self.dof_idx)
        self.scene.step()

    def get_state(self) -> SimState:
        calf_pos = self.robot.get_links_pos(self.feet_links)
        calf_quat = self.robot.get_links_quat(self.feet_links)
        offset = self.foot_offset.expand(calf_quat.shape[:-1] + (3,))
        feet_pos = calf_pos + transform_by_quat(offset.reshape(-1, 3), calf_quat.reshape(-1, 4)).reshape(calf_pos.shape)
        return SimState(
            base_pos=self.robot.get_pos(),
            base_quat=self.robot.get_quat(),
            base_lin_vel=self.robot.get_vel(),
            base_ang_vel=self.robot.get_ang(),
            dof_pos=self.robot.get_dofs_position(self.dof_idx),
            dof_vel=self.robot.get_dofs_velocity(self.dof_idx),
            feet_pos=feet_pos,
            knee_pos=calf_pos,
        )

    def marker_positions(self):
        """(N, 17, 3) world positions in the order of MARKER_NAMES (the joints are the origins of their child links)."""
        pts = [self.robot.get_pos()]
        for leg in MARKER_LEGS:
            for part in ("hip", "thigh", "calf"):
                pts.append(self.robot.get_link(f"{leg}_{part}").get_pos())
            calf = self.robot.get_link(f"{leg}_calf")
            offset = self.foot_offset.expand(calf.get_quat().shape[:-1] + (3,))
            pts.append(calf.get_pos() + transform_by_quat(offset, calf.get_quat()))
        return torch.stack(pts, dim=1)

    def reset(self, envs_mask, base_pos, base_quat, dof_pos):
        # qpos = [pos(3), quat(4), joints]; the free joint has 6 dofs but 7 qs, so joint q index = dof index + 1.
        qpos = torch.empty((self.num_envs, 7 + len(self.dof_idx)), dtype=gs.tc_float, device=gs.device)
        qpos[:, :3] = base_pos
        qpos[:, 3:7] = base_quat
        qpos[:, self.dof_idx + 1] = dof_pos
        self.robot.set_qpos(qpos, envs_idx=envs_mask, zero_velocity=True)

    def error_mask(self):
        return self.scene.rigid_solver.get_error_envs_mask()

    def start_recording(self, path, fps):
        self.cam.start_recording(save_to_filename=path, fps=fps)

    def render_frame(self, cam_pos, lookat):
        self.cam.set_pose(pos=cam_pos, lookat=lookat)
        return self.cam.render()[0]

    def stop_recording(self):
        self.cam.stop_recording()
